"""
/chat/v2/stream — agent_v2 RAG 챗봇 SSE 스트리밍 라우트.

흐름:

   [질문] → ① Rule-based pre-router (정규식 즉답)
              │
              ├─ smalltalk 즉시 확정 → token streaming (temperature=0.6)
              │
              └─ 애매 → ② Supervisor LLM (structured output) ┬─ smalltalk
                                                              └─ rag → 2-step
                                                                       (a) tool_choice='rag_search'
                                                                            강제 → RAG (Parent-Child)
                                                                       (b) <context>+<question> XML
                                                                            → token streaming (temp=0.0)

긴 단계(LLM/RAG 호출) 사이에 SSE ':ping' heartbeat 를 5초마다 송출 — 프록시
keep-alive timeout 방어용.

LLM 자율 판단(`tool_choice="auto"`)을 봉쇄해야 GGUF chat handler 가 Qwen/Gemma
양쪽 모두 일관되게 동작한다 (auto + Gemma 4 조합은 native function calling
토큰을 tool_calls JSON 으로 변환 못 함).

SSE 이벤트 타입:
  - info       : 진행 상황 메시지
  - rag_info   : RAG/도구 결과 알림
  - text       : 답변 토큰 (점진적 streaming)
  - done       : 종료
  - error      : 오류
  (그 외 ':ping' SSE 주석 형식 heartbeat)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


_chat_model = None  # 전역 LLM 인스턴스 캐시
_HEARTBEAT_INTERVAL = 5.0  # 초 — RAG/LLM 호출 사이 ':ping' 송출 주기

# 답변 streaming 의 sampling 파라미터.
RAG_TEMPERATURE = 0.0       # 사실 검색 답변 — greedy
SMALLTALK_TEMPERATURE = 0.6  # 잡담 — 약간의 자연스러움


def get_chat_model():
    """ChatModel 싱글턴 반환 (최초 호출 시 로드).

    환경변수 CHAT_V2_BACKEND:
      - gemma (기본): Gemma 4 E4B-it GGUF Q4_K_M.
      - qwen / gguf : Qwen2.5-7B GGUF Q4_K_M.
    """
    global _chat_model
    if _chat_model is None:
        import os
        backend = os.environ.get("CHAT_V2_BACKEND", "gemma").lower()
        if backend in ("qwen", "gguf"):
            from app.core.agent_v2.llm_qwen import build_qwen_chat_model_gguf
            logger.info("[chat_v2] Qwen 2.5-7B GGUF Q4_K_M 로드 중 (~2초)")
            _chat_model = build_qwen_chat_model_gguf()
        else:
            from app.core.agent_v2.llm_qwen import build_gemma_chat_model_gguf
            logger.info("[chat_v2] Gemma 4 E4B-it GGUF Q4_K_M 로드 중 (~2초)")
            _chat_model = build_gemma_chat_model_gguf()
    return _chat_model


def _sse(event_type: str, content: str = "", **kw) -> str:
    """SSE 이벤트 1줄 직렬화."""
    payload = {"type": event_type, "content": content, **kw}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ─── Rule-based pre-routing ───────────────────────────────────────────────
# Supervisor LLM 호출 (~1.5초) 을 우회해 즉답 가능한 smalltalk 패턴.
# 보수적으로 잡음 — 애매한 입력은 Supervisor 로 넘김.
_SMALLTALK_PATTERNS = [
    # 인사
    r"^(안녕|안녕하세요|하이|hi+|hello+|반가|반갑)\b",
    # 감사
    r"^(고마워|고마운|감사|땡큐|thank you|thanks)\b",
    # 끝맺음·격려
    r"^(잘\s?자|굿\s?나잇|잘\s?있어|수고|화이팅|파이팅|good\s?bye|bye)\b",
    # 단순 응답
    r"^(네|넵|예|응|어|음|오|아|ㅎㅎ+|ㅋㅋ+|ㅠㅠ+|ok|okay)$",
    # 자기소개 요청
    r"^(너\s?누구|넌\s?누구|누구야|니가\s?누구|자기소개)",
]
_SMALLTALK_RE = re.compile("|".join(_SMALLTALK_PATTERNS), flags=re.IGNORECASE)


def _quick_smalltalk(message: str) -> bool:
    """LLM 호출 없이 smalltalk 으로 즉시 분류 가능한가."""
    msg = (message or "").strip()
    if not msg:
        return True
    # 너무 짧으면 (5자 이하) + 한국어/영어 단순 응답 → smalltalk 후보
    if len(msg) <= 5 and _SMALLTALK_RE.search(msg):
        return True
    # 패턴 매칭 (긴 인사도 포함, 예: "안녕하세요 오늘 날씨 어때")
    if _SMALLTALK_RE.search(msg):
        return True
    return False


# ─── Supervisor LLM (rule-based 가 잡아내지 못한 경우) ───────────────────
class RouteDecision(BaseModel):
    """Supervisor 의 라우팅 결정."""
    route: Literal["rag", "smalltalk"] = Field(
        description=(
            "rag = 헬리콥터 매뉴얼에서 검색이 필요한 사실/정의/절차 질문, "
            "smalltalk = 인사·잡담·고마움 표현 등 RAG 가 불필요한 메시지"
        )
    )
    reason: str = Field(description="라우팅 결정 이유 (한국어 1문장)")


SUPERVISOR_SYSTEM = (
    "당신은 헬리콥터 AI 튜터의 라우터입니다.\n"
    "사용자 메시지를 분석해 두 경로 중 하나로 분류하세요.\n"
    " - rag       : 헬리콥터/항공/매뉴얼 관련 사실·정의·절차 질문\n"
    "                (예: '베르누이가 뭐야', '엔진 시동 순서', 'VRS 정의', 'AGB 가 뭐야')\n"
    "                약어·전문용어는 일단 rag 로 분류.\n"
    " - smalltalk : 인사·잡담·감사·날씨 등 RAG 가 불필요한 메시지\n"
    "                (예: '안녕', '고마워', '잘 자', '오늘 날씨 어때')\n"
    "반드시 RouteDecision 스키마로 응답하세요."
)


SMALLTALK_SYSTEM = (
    "당신은 헬리콥터 AI 튜터의 친근한 비서입니다. "
    "사용자의 짧은 인사/잡담에 한두 문장으로 따뜻하고 자연스럽게 한국어로 응답하세요. "
    "헬리콥터·매뉴얼 관련 질문이 있다면 언제든 물어달라고 가볍게 안내해도 좋습니다."
)


SEARCH_QUERY_SYSTEM = (
    "당신은 검색 쿼리 추출기입니다. 사용자 질문에서 헬리콥터 매뉴얼 검색에 쓸 "
    "핵심 명사 키워드만 뽑아 rag_search 도구의 query 인자로 호출하세요.\n"
    "\n"
    "규칙:\n"
    "- 명사·전문용어만 추출 (조사·동사·서술어·문서형식 단어는 절대 제외)\n"
    "- 금지어: 'PDF', '문서', '매뉴얼', '교재', '알려주세요', '설명', '에 대해', '에 대한'\n"
    "- 2~5단어, 30자 이내\n"
    "\n"
    "예시:\n"
    "- 사용자: '베르누이 정리에 대해 알려주세요' → query: '베르누이 정리'\n"
    "- 사용자: '헬리콥터 시동 절차를 단계별로 설명해주세요' → query: '헬리콥터 시동 절차'\n"
    "- 사용자: 'Vortex Ring State 란 무엇인가요?' → query: 'Vortex Ring State'\n"
    "- 사용자: '동적 롤오버는 언제 발생하나요' → query: '동적 롤오버 발생 조건'\n"
    "\n"
    "답변 생성은 절대 하지 않습니다."
)


# D6/D8: XML 태그 + 질문 유형별 유연한 서식 + 자연스러운 인용
RESEARCHER_SYSTEM = (
    "당신은 헬리콥터 표준 교재 기반 사실 검색 전문가입니다. "
    "<context> 안의 참고 문서를 바탕으로 <question> 에 한국어로 답하세요.\n"
    "\n"
    "[서식]\n"
    "- 질문의 성격에 따라 가독성을 우선해 자유롭게 구성하세요.\n"
    "  · 절차·순서를 묻는 질문은 번호 매기기(1. 2. 3.)\n"
    "  · 정의·요약을 묻는 질문은 1~3문장 평문, 필요 시 글머리 기호(•)\n"
    "  · 원리·메커니즘을 묻는 질문은 흐름이 보이도록 단락 구성\n"
    "\n"
    "[인용]\n"
    "- 답변에 사용한 근거 페이지를 [p.숫자] 형식으로 자연스럽게 표기하세요.\n"
    "- 매 문장 끝마다 강제로 붙이지 말고, 단락 끝 또는 핵심 설명 뒤에 한 번씩.\n"
    "- 같은 페이지를 연속 인용할 때는 묶어서: [p.10, p.52]\n"
    "\n"
    "[금지]\n"
    "- 참고 문서에 없는 내용은 추측하지 말고 '문서에 충분한 정보가 없습니다'라고 답하세요."
)


def _answer_user_message(user_query: str, search_result: str):
    """답변 LLM 에 줄 통합 user 메시지 — XML 태그로 컨텍스트/질문 분리.

    답변 작성 지시는 SystemMessage(RESEARCHER_SYSTEM) 에 두고 여기서는 중복하지 않는다.
    """
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=(
        f"<context>\n{search_result}\n</context>\n\n"
        f"<question>\n{user_query}\n</question>"
    ))


# ─── SSE heartbeat helper ─────────────────────────────────────────────────
async def _drain_with_ping(task: asyncio.Task):
    """task 완료를 기다리며 _HEARTBEAT_INTERVAL 마다 ':ping' 줄을 yield.

    호출 측은 `async for ping in _drain_with_ping(task): yield ping` 후 `task.result()` 사용.
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            # SSE 주석 형식 — 클라이언트는 무시하고 연결만 유지.
            yield ":ping\n\n"


# ─── 메인 SSE generator ────────────────────────────────────────────────────
async def stream_answer(message: str):
    """rule-based pre-router → (옵션) supervisor LLM → smalltalk OR (RAG → 답변).

    SSE 이벤트 yield. voice.py 등에서도 STT 결과를 흘려넣어 재사용 가능.
    """
    import anyio
    from starlette.concurrency import iterate_in_threadpool
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from app.core.agent_v2.tools import rag_search
    from app.core.agent_v2.langfuse_handler import (
        request_span,
        new_callbacks,
        flush as lf_flush,
    )

    try:
        chat_model = get_chat_model()
    except Exception as e:
        yield _sse("error", f"모델 로드 실패: {e}")
        return

    t0 = time.perf_counter()
    answer_buf: list[str] = []
    route = "rag"
    route_source = "supervisor"  # "rule" / "supervisor" / "fallback"

    with request_span("agent-pipeline", input=message) as _span:
        try:
            yield _sse("info", "📥 질문 수신, agent_v2 시작...")

            # ─── 0-A: Rule-based pre-router (LLM 호출 X, 즉답) ───
            if _quick_smalltalk(message):
                route = "smalltalk"
                route_source = "rule"
                yield _sse("info", "🧭 라우팅: smalltalk (규칙 기반 즉답)")
            else:
                # ─── 0-B: Supervisor LLM (애매한 케이스만) ───
                structured = chat_model.with_structured_output(RouteDecision)
                sup_task = asyncio.create_task(anyio.to_thread.run_sync(
                    lambda: structured.invoke(
                        [
                            SystemMessage(content=SUPERVISOR_SYSTEM),
                            HumanMessage(content=message),
                        ],
                        config={"callbacks": new_callbacks()},
                    )
                ))
                async for ping in _drain_with_ping(sup_task):
                    yield ping
                try:
                    decision: RouteDecision = sup_task.result()
                    route = decision.route
                    yield _sse("info", f"🧭 라우팅: {route} ({decision.reason})")
                except Exception as e:
                    logger.warning(f"[chat_v2] Supervisor 분류 실패, RAG 폴백: {e}")
                    route = "rag"
                    route_source = "fallback"
                    yield _sse("info", "🧭 라우팅: rag (Supervisor 실패 폴백)")

            # ─── smalltalk 경로: RAG 스킵, 도구 없이 token streaming ───
            if route == "smalltalk":
                smalltalk_messages = [
                    SystemMessage(content=SMALLTALK_SYSTEM),
                    HumanMessage(content=message),
                ]

                def _stream_smalltalk():
                    return chat_model.stream(
                        smalltalk_messages,
                        config={"callbacks": new_callbacks()},
                        temperature=SMALLTALK_TEMPERATURE,
                    )

                async for chunk in iterate_in_threadpool(_stream_smalltalk()):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        answer_buf.append(str(token))
                        yield _sse("text", str(token))

                elapsed = time.perf_counter() - t0
                yield _sse("info", f"⏱️ 소요 {elapsed:.1f}초")
                yield _sse("done", "")

                full_answer = "".join(answer_buf)
                if _span is not None:
                    try:
                        _span.update(
                            output=full_answer or None,
                            metadata={"route": "smalltalk", "route_source": route_source,
                                      "elapsed_sec": round(elapsed, 1)},
                        )
                        _span.set_trace_io(output=full_answer or None)
                    except Exception:
                        pass
                return

            # ─── rag 경로 (a) 검색 쿼리 추출 (tool_choice='rag_search' 강제) ───
            yield _sse("rag_info", "🔍 RAG 검색 중...")
            search_messages = [
                SystemMessage(content=SEARCH_QUERY_SYSTEM),
                HumanMessage(content=message),
            ]
            llm_search = chat_model.bind_tools([rag_search], tool_choice="rag_search")
            qry_task = asyncio.create_task(anyio.to_thread.run_sync(
                lambda: llm_search.invoke(
                    search_messages, config={"callbacks": new_callbacks()}
                )
            ))
            async for ping in _drain_with_ping(qry_task):
                yield ping
            ai_search: AIMessage = qry_task.result()
            if not ai_search.tool_calls:
                err = ai_search.content or "검색 쿼리 추출 실패"
                yield _sse("text", str(err))
                yield _sse("done", "")
                return

            tc = ai_search.tool_calls[0]
            query = tc["args"].get("query", "")
            yield _sse("rag_info", f"🔎 쿼리: {query}")

            rag_task = asyncio.create_task(anyio.to_thread.run_sync(
                lambda: rag_search.invoke(tc["args"], config={"callbacks": new_callbacks()})
            ))
            async for ping in _drain_with_ping(rag_task):
                yield ping
            search_text = str(rag_task.result())
            yield _sse("rag_info", f"📄 RAG 결과 {len(search_text)}자")

            # ─── rag 경로 (b) 답변 streaming (도구 없음, 자유 텍스트) ───
            answer_messages = [
                SystemMessage(content=RESEARCHER_SYSTEM),
                _answer_user_message(message, search_text),
            ]

            def _stream_iter():
                return chat_model.stream(
                    answer_messages,
                    config={"callbacks": new_callbacks()},
                    temperature=RAG_TEMPERATURE,
                )

            async for chunk in iterate_in_threadpool(_stream_iter()):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    answer_buf.append(str(token))
                    yield _sse("text", str(token))

            elapsed = time.perf_counter() - t0
            yield _sse("info", f"⏱️ 소요 {elapsed:.1f}초")
            yield _sse("done", "")

            full_answer = "".join(answer_buf)
            if _span is not None:
                try:
                    _span.update(
                        output=full_answer or None,
                        metadata={"route": "rag", "route_source": route_source,
                                  "elapsed_sec": round(elapsed, 1)},
                    )
                    _span.set_trace_io(output=full_answer or None)
                except Exception:
                    pass

        except Exception as e:
            logger.exception("[chat_v2] 오류")
            yield _sse("error", f"처리 중 오류: {e}")
        finally:
            try:
                lf_flush()
            except Exception:
                pass


@router.get("/chat/v2/stream")
async def chat_v2_stream(message: str = Query(..., description="사용자 질문")):
    """agent_v2 SSE 라우트."""
    try:
        get_chat_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"모델 로드 실패: {e}")

    return StreamingResponse(
        stream_answer(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 프록시 환경에서 버퍼링 끔
            "Access-Control-Allow-Origin": "*",
        },
    )
