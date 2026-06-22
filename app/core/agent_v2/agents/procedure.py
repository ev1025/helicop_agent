"""
Procedure 에이전트 — 절차/단계/방법 질문에 대해 단계별 답변.

Researcher 와 같은 도구 (rag_search, final_answer) 를 쓰지만 답변 system prompt 가
"단계별 형식" 을 강제한다. 검색 쿼리 추출 단계는 SEARCH_QUERY_SYSTEM 공유.
"""

from __future__ import annotations

import logging
from typing import Iterator, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.agent_v2.tools import rag_search, final_answer
from app.core.agent_v2.agents import SEARCH_QUERY_SYSTEM

logger = logging.getLogger(__name__)


# 답변 작성용 — 단계별 형식 강제 (Procedure 의 핵심 차이점).
PROCEDURE_SYSTEM = (
    "당신은 헬리콥터 표준 교재의 절차/단계 질문 전문가입니다. "
    "참고 문서를 바탕으로 반드시 번호 매긴 단계 (1. 2. 3.) 형식으로 답변하세요. "
    "참고 문서에 단계가 명시되지 않았으면 '문서에 명확한 절차가 없습니다'라고 답하세요."
)


def _answer_user_message(user_query: str, search_result: str) -> HumanMessage:
    """답변 LLM 에 줄 통합 user 메시지 — [참고 문서] + [질문] 단일 블록."""
    return HumanMessage(content=(
        f"[참고 문서]\n{search_result}\n\n"
        f"[질문]\n{user_query}\n\n"
        f"위 참고 문서를 바탕으로 번호 매긴 단계 (1. 2. 3.) 형식으로 답변해주세요."
    ))


def run(chat_model, user_query: str, prior_feedback: str | None = None) -> str:
    """CHANGES.md #016-A: prior_feedback 받으면 답변 system prompt 에 추가."""
    from app.core.agent_v2.langfuse_handler import new_callbacks

    # 1단계: 쿼리 추출 (검색 전용 prompt)
    search_messages = [
        SystemMessage(content=SEARCH_QUERY_SYSTEM),
        HumanMessage(content=user_query),
    ]
    llm_search = chat_model.bind_tools([rag_search, final_answer], tool_choice="rag_search")
    ai1: AIMessage = llm_search.invoke(search_messages, config={"callbacks": new_callbacks()})
    if not ai1.tool_calls:
        return ai1.content or "절차 검색 실패"

    tc = ai1.tool_calls[0]
    search_result = rag_search.invoke(tc["args"], config={"callbacks": new_callbacks()})

    # 2단계: 답변 (Procedure 답변 prompt + 검색 결과 통합 user msg)
    answer_sys = PROCEDURE_SYSTEM
    if prior_feedback:
        answer_sys = f"{PROCEDURE_SYSTEM}\n\n[이전 답변에 대한 평가]\n{prior_feedback}"
    answer_messages = [
        SystemMessage(content=answer_sys),
        _answer_user_message(user_query, str(search_result)),
    ]
    llm_answer = chat_model.bind_tools([rag_search, final_answer], tool_choice="final_answer")
    ai2: AIMessage = llm_answer.invoke(answer_messages, config={"callbacks": new_callbacks()})
    if not ai2.tool_calls:
        return ai2.content or "절차 답변 생성 실패"
    return ai2.tool_calls[0]["args"].get("answer", "(빈 답변)")


def run_streaming(
    chat_model,
    user_query: str,
    prior_feedback: str | None = None,
) -> Iterator[Tuple[str, str]]:
    """Streaming 변형. researcher.run_streaming 과 동일 구조, 절차 system prompt."""
    from app.core.agent_v2.langfuse_handler import new_callbacks

    # 1단계: 쿼리 추출
    yield ("rag_info", "🔍 RAG 검색 중 (절차)...")
    search_messages = [
        SystemMessage(content=SEARCH_QUERY_SYSTEM),
        HumanMessage(content=user_query),
    ]
    llm_search = chat_model.bind_tools([rag_search, final_answer], tool_choice="rag_search")
    ai1: AIMessage = llm_search.invoke(search_messages, config={"callbacks": new_callbacks()})
    if not ai1.tool_calls:
        yield ("text", ai1.content or "절차 검색 실패")
        return

    tc = ai1.tool_calls[0]
    search_result = rag_search.invoke(tc["args"], config={"callbacks": new_callbacks()})
    yield ("rag_info", f"📄 RAG 결과 {len(str(search_result))}자")

    # 2단계: 답변 streaming
    answer_sys = PROCEDURE_SYSTEM
    if prior_feedback:
        answer_sys = f"{PROCEDURE_SYSTEM}\n\n[이전 답변에 대한 평가]\n{prior_feedback}"
    answer_messages = [
        SystemMessage(content=answer_sys),
        _answer_user_message(user_query, str(search_result)),
    ]
    for chunk in chat_model.stream(answer_messages, config={"callbacks": new_callbacks()}):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            yield ("text", str(token))
