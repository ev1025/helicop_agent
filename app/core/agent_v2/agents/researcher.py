"""
Researcher 에이전트 — 사실/정의 질문에 대해 RAG 검색 후 답변.

도구: rag_search, final_answer
첫 호출은 tool_choice="rag_search" 강제, 두 번째는 tool_choice="final_answer" 강제 (run)
또는 plain text streaming (run_streaming).
이로써 args 누락 / BPE 노이즈 / 도구 미호출 모두 차단 (#002 결과).

System prompt 분리: 검색 쿼리 추출용(SEARCH_QUERY_SYSTEM, agents/__init__.py) ≠ 답변 작성용
(RESEARCHER_SYSTEM, 이 파일). 호출별 역할에 맞는 prompt 만 들어감 — 토큰 절감 + 역할 명료.
"""

from __future__ import annotations

import logging
from typing import Iterator, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.agent_v2.tools import rag_search, final_answer
from app.core.agent_v2.agents import SEARCH_QUERY_SYSTEM

logger = logging.getLogger(__name__)


# 답변 작성용 — 검색이 끝난 시점에 LLM 에 들어가는 system prompt.
# (이전엔 "1) rag_search 2) final_answer" 같은 도구 흐름 지시도 들어가 있었으나,
#  답변 단계에선 이미 검색이 끝났고 final_answer 도구를 streaming 에선 안 쓰므로 제거.)
RESEARCHER_SYSTEM = (
    "당신은 헬리콥터 표준 교재를 기반으로 답변하는 사실 검색 전문가입니다. "
    "참고 문서를 바탕으로 한국어로 간결히 요약해 답변하세요. "
    "참고 문서에 없는 내용은 '문서에 충분한 정보가 없습니다'라고 답하세요."
)


def _answer_user_message(user_query: str, search_result: str) -> HumanMessage:
    """답변 LLM 에 줄 통합 user 메시지 — [참고 문서] + [질문] 단일 블록."""
    return HumanMessage(content=(
        f"[참고 문서]\n{search_result}\n\n"
        f"[질문]\n{user_query}\n\n"
        f"위 참고 문서를 바탕으로 한국어로 간결하게 답변해주세요."
    ))


def run(chat_model, user_query: str, prior_feedback: str | None = None) -> str:
    """Researcher 한 사이클 실행 (non-streaming). 최종 답변 텍스트 반환.

    Args:
        chat_model: ChatModel.
        user_query: 사용자 질문.
        prior_feedback: 이전 retry 의 critic 평가 (있으면 답변 system prompt 에 추가).
                        CHANGES.md #016-A — feedback loop 구현.
    """
    from app.core.agent_v2.langfuse_handler import new_callbacks

    # 1단계: rag_search 강제 — 쿼리 추출 전용 prompt (답변 작성 지시 없음)
    search_messages = [
        SystemMessage(content=SEARCH_QUERY_SYSTEM),
        HumanMessage(content=user_query),
    ]
    llm_search = chat_model.bind_tools([rag_search, final_answer], tool_choice="rag_search")
    ai1: AIMessage = llm_search.invoke(search_messages, config={"callbacks": new_callbacks()})
    if not ai1.tool_calls:
        logger.warning("[Researcher] rag_search 호출 실패")
        return ai1.content or "검색에 실패했습니다."

    tc = ai1.tool_calls[0]
    search_result = rag_search.invoke(tc["args"], config={"callbacks": new_callbacks()})

    # 2단계: final_answer 강제 — Researcher 답변 prompt + 검색 결과 통합 user msg
    answer_sys = RESEARCHER_SYSTEM
    if prior_feedback:
        answer_sys = f"{RESEARCHER_SYSTEM}\n\n[이전 답변에 대한 평가]\n{prior_feedback}"
    answer_messages = [
        SystemMessage(content=answer_sys),
        _answer_user_message(user_query, str(search_result)),
    ]
    llm_answer = chat_model.bind_tools([rag_search, final_answer], tool_choice="final_answer")
    ai2: AIMessage = llm_answer.invoke(answer_messages, config={"callbacks": new_callbacks()})
    if not ai2.tool_calls:
        return ai2.content or "답변 생성 실패"
    return ai2.tool_calls[0]["args"].get("answer", "(빈 답변)")


def run_streaming(
    chat_model,
    user_query: str,
    prior_feedback: str | None = None,
) -> Iterator[Tuple[str, str]]:
    """Streaming 변형 — TTFT 최적화. (event_type, content) tuple 을 yield.

    이벤트 종류:
      - ("rag_info", "...")  RAG 진행 알림
      - ("text", token)      답변 토큰 (점진적)

    run() 과의 차이: 답변 단계에서 final_answer 도구 wrapping 을 안 쓰고
    plain text 로 streaming. JSON 토큰 흘리는 문제 회피.

    Langfuse: 각 LLM 호출이 ambient OTEL context (있으면) 의 자식 span 으로 자동 기록.
    """
    from app.core.agent_v2.langfuse_handler import new_callbacks

    # 1단계: RAG 검색 — 쿼리 추출 전용 prompt
    yield ("rag_info", "🔍 RAG 검색 중...")
    search_messages = [
        SystemMessage(content=SEARCH_QUERY_SYSTEM),
        HumanMessage(content=user_query),
    ]
    llm_search = chat_model.bind_tools([rag_search, final_answer], tool_choice="rag_search")
    ai1: AIMessage = llm_search.invoke(search_messages, config={"callbacks": new_callbacks()})
    if not ai1.tool_calls:
        logger.warning("[Researcher.stream] rag_search 호출 실패")
        yield ("text", ai1.content or "검색에 실패했습니다.")
        return

    tc = ai1.tool_calls[0]
    search_result = rag_search.invoke(tc["args"], config={"callbacks": new_callbacks()})
    yield ("rag_info", f"📄 RAG 결과 {len(str(search_result))}자")

    # 2단계: 답변 streaming — Researcher 답변 prompt + 검색 결과 통합 user msg
    answer_sys = RESEARCHER_SYSTEM
    if prior_feedback:
        answer_sys = f"{RESEARCHER_SYSTEM}\n\n[이전 답변에 대한 평가]\n{prior_feedback}"
    answer_messages = [
        SystemMessage(content=answer_sys),
        _answer_user_message(user_query, str(search_result)),
    ]
    for chunk in chat_model.stream(answer_messages, config={"callbacks": new_callbacks()}):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            yield ("text", str(token))
