"""
Supervisor 에이전트 — 사용자 질문을 보고 어느 Worker 로 보낼지 결정한다.

방식: with_structured_output(RouteDecision) 으로 LLM 이 정확한 enum 값을 반환하도록 강제.

분기:
  - "researcher"  : 일반 정의/사실 질문 (베르누이, 양력, VRS 정의 등)
  - "procedure"   : 절차/단계/방법 질문 ("어떻게 작동하나", "순서가 뭐야")
  - "smalltalk"   : RAG 가 필요 없는 잡담 ("안녕", "고마워")
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """라우팅 결정. Supervisor 가 채워서 반환."""
    route: Literal["researcher", "procedure", "smalltalk"] = Field(
        description=(
            "어느 워커로 보낼지: "
            "researcher = 정의·사실 질문, "
            "procedure = 절차/방법/단계 질문, "
            "smalltalk = RAG 불필요한 잡담"
        )
    )
    reason: str = Field(description="라우팅 결정 이유 (한국어 1~2문장)")


SUPERVISOR_SYSTEM_PROMPT = (
    "당신은 헬리콥터 AI 튜터의 라우터입니다.\n"
    "사용자 질문을 분석해서 가장 적합한 워커 에이전트로 보내야 합니다.\n"
    "researcher: 사실/정의 질문 (예: '베르누이가 뭐야', '양력은 왜 생기나')\n"
    "procedure: 절차/단계/방법 질문 (예: '엔진 시동 순서', '비상 착륙 절차')\n"
    "smalltalk: RAG 불필요한 잡담 (예: '안녕', '고마워', '날씨')\n"
    "반드시 RouteDecision 스키마에 맞춰 응답하세요."
)


def route(chat_model, user_query: str) -> RouteDecision:
    """라우팅 결정을 반환.

    Args:
        chat_model: QwenChatModel 인스턴스 (이미 with_structured_output 가능 상태).
        user_query: 사용자 질문.

    Langfuse: 호출 시 ambient OTEL context (있으면) 의 자식 span 으로 자동 기록.
    """
    from app.core.agent_v2.langfuse_handler import new_callbacks

    structured = chat_model.with_structured_output(RouteDecision)
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
    decision: RouteDecision = structured.invoke(messages, config={"callbacks": new_callbacks()})
    logger.info(f"[Supervisor] {user_query[:30]!r} → {decision.route} ({decision.reason})")
    return decision
