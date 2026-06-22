"""
Critic 에이전트 — Worker 의 답변을 평가.

with_structured_output 으로 다음 항목을 LLM 에게 강제 채우게 함:
  - score (0~10)
  - issues (목록, 한국어)
  - needs_retry (bool)

needs_retry 가 True 면 graph.py 가 워커에게 다시 요청 (retry).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Critique(BaseModel):
    """답변 품질 평가."""
    score: int = Field(ge=0, le=10, description="답변 품질 점수 (0=형편없음, 10=완벽)")
    issues: List[str] = Field(description="발견된 문제점 목록 (없으면 빈 리스트)")
    needs_retry: bool = Field(description="워커에게 재시도를 요청할지 여부")
    reason: str = Field(description="평가 이유 (한국어 1~2문장)")


CRITIC_SYSTEM = (
    "당신은 헬리콥터 AI 튜터의 답변 품질 평가자입니다.\n"
    "주어진 (질문, 답변) 쌍을 보고 다음 기준으로 평가하세요:\n"
    "- 정확성: 답변이 사실에 부합하는가\n"
    "- 완전성: 핵심 내용을 다 다루는가\n"
    "- 구체성: 모호하지 않고 구체적인가\n"
    "- 형식: 가독성, 구조\n"
    "score 는 0~10. 6점 미만이면 needs_retry=True 로 설정하세요.\n"
    "반드시 Critique 스키마로 응답하세요."
)


def evaluate(chat_model, user_query: str, answer: str) -> Critique:
    structured = chat_model.with_structured_output(Critique)
    messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=f"질문: {user_query}\n\n답변: {answer}"),
    ]
    critique: Critique = structured.invoke(messages)
    logger.info(
        f"[Critic] score={critique.score} retry={critique.needs_retry} "
        f"issues={critique.issues}"
    )
    return critique
