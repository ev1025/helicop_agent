"""
외부에서 호출하는 entry point.

기존 app/core/orchestrator.Orchestrator.run(user_query) 와 동일한 인터페이스를
LangGraph 기반으로 제공한다. chat.py 가 이 함수를 호출하면 된다.

사용 예:
    from app.core.agent_v2.runner import get_runner

    runner = get_runner()
    answer = runner.run("베르누이 원리로 양력이 어떻게 발생하나요?")
    print(answer)

전역 싱글턴으로 모델을 한 번만 로드한다 (로드 비용이 크기 때문).
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.config import config
from app.core.agent_v2.graph import build_graph
from app.core.agent_v2.llm_qwen import build_qwen_chat_model
from app.core.agent_v2.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "당신은 헬리콥터 표준 교재를 바탕으로 답변하는 항공 AI 튜터입니다.\n"
    "사용자 질문에 답하기 전 반드시 rag_search 도구를 호출해 관련 문서를 찾고, "
    "검색 결과를 바탕으로 final_answer 도구로 최종 답변을 전달하세요.\n"
    "- 추측 금지. 문서에 없는 내용은 '문서에 충분한 정보가 없습니다'라고 답하세요.\n"
    "- 답변은 간결하고 한국어로 작성하세요."
)


class AgentRunner:
    """Qwen + LangGraph 기반 에이전트 실행기 (싱글턴)."""

    def __init__(self, llm_with_tools, app):
        self._llm_with_tools = llm_with_tools
        self._app = app

    def run(self, user_query: str, recursion_limit: int = 10) -> str:
        """단일 질의를 실행하고 최종 답변 텍스트를 반환.

        Args:
            user_query: 사용자 질문.
            recursion_limit: 그래프 노드 최대 순회 횟수 (무한 루프 방지).

        Returns:
            최종 답변 문자열. final_answer 가 호출됐으면 그 내용을,
            도구 호출 없이 LLM 이 답한 경우 LLM 텍스트를 반환.
        """
        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ]
        }
        final_state = self._app.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
        )
        return self._extract_answer(final_state)

    @staticmethod
    def _extract_answer(state) -> str:
        """최종 상태에서 답변 텍스트를 뽑아낸다."""
        messages = state["messages"]
        # 1) final_answer ToolMessage 가 있으면 그 내용
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "final_answer":
                return str(msg.content)
        # 2) 마지막 AIMessage 의 content
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return "답변을 생성할 수 없습니다."


_RUNNER: Optional[AgentRunner] = None


def get_runner(force_reload: bool = False) -> AgentRunner:
    """전역 싱글턴 AgentRunner 를 반환 (없으면 생성)."""
    global _RUNNER
    if _RUNNER is not None and not force_reload:
        return _RUNNER

    # CHANGES.md #008/#010 결과: GGUF 백엔드가 60× 빠름.
    # 환경변수 AGENT_V2_BACKEND=bnb 로 BNB 4bit 강제 가능.
    import os
    backend = os.environ.get("AGENT_V2_BACKEND", "gguf").lower()
    if backend == "bnb":
        model_name = getattr(config, "LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        logger.info(f"[agent_v2] AgentRunner 초기화: BNB 4bit ({model_name})")
        chat = build_qwen_chat_model(model_name=model_name)
    else:
        from app.core.agent_v2.llm_qwen import build_qwen_chat_model_gguf
        logger.info("[agent_v2] AgentRunner 초기화: GGUF Q4_K_M (60× 빠름)")
        chat = build_qwen_chat_model_gguf()

    llm_with_tools = chat.bind_tools(ALL_TOOLS)
    app = build_graph(llm_with_tools)
    _RUNNER = AgentRunner(llm_with_tools=llm_with_tools, app=app)
    logger.info("[agent_v2] AgentRunner 싱글턴 초기화 완료")
    return _RUNNER
