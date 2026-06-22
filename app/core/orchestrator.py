"""
Orchestrator

이 모듈은 대화 흐름을 관리하는 Orchestrator를 정의합니다.
Conversation과 ToolExecutor를 통합하여 멀티턴 대화를 처리합니다.
"""

from typing import Optional
from app.core.conversation import Conversation, ToolCall
from app.core.tool_executor import ToolExecutor


class Orchestrator:
    """
    대화 흐름 관리자

    Conversation과 ToolExecutor를 통합하여 멀티턴 대화를 처리합니다.
    현재는 하드코딩된 플로우로 동작하며, 향후 LLM 통합 예정입니다.

    Example:
        >>> executor = ToolExecutor()
        >>> executor.register_tool(RAGSearchTool())
        >>> executor.register_tool(FinalAnswerTool())
        >>> orchestrator = Orchestrator(executor)
        >>> result = orchestrator.run("양력이란 무엇인가요?")
        >>> print(result)
        [TOOL_RESULT] ...
    """

    def __init__(self, executor: ToolExecutor):
        """
        Orchestrator 초기화

        Args:
            executor: 도구 실행 엔진
        """
        self.executor = executor
        self.conversation = Conversation()

    def run(self, user_query: str, max_turns: int = 10, use_llm: bool = True) -> str:
        """
        사용자 질문을 처리하고 답변을 반환합니다.

        Phase 2: LLM 기반 멀티턴 대화 루프
        1. 사용자 질문 추가
        2. LLM 호출 (도구 스키마 포함)
        3. LLM이 도구를 선택하거나 직접 답변
        4. 도구 실행 및 결과 추가
        5. final_answer가 호출될 때까지 반복

        Args:
            user_query: 사용자 질문
            max_turns: 최대 대화 턴 수 (무한 루프 방지, 기본값: 10)
            use_llm: LLM 사용 여부 (False면 Phase 1 하드코딩 플로우)

        Returns:
            str: 최종 답변

        Raises:
            RuntimeError: max_turns 도달 시
        """
        from app.core.llm import generate_with_tools_local

        # 사용자 질문 추가
        self.conversation.add_user_message(user_query)

        if not use_llm:
            # Phase 1 하드코딩 플로우 (하위 호환성)
            return self._run_hardcoded(user_query)

        # Phase 2: LLM 기반 멀티턴 루프
        current_query = user_query

        for turn in range(1, max_turns + 1):
            # 도구 스키마 가져오기
            tools = self.executor.get_all_schemas()

            # 대화 히스토리 구성 (간단히 문자열로)
            conversation_history = None
            if len(self.conversation.messages) > 1:
                # 이전 메시지들이 있으면 히스토리로 전달
                conversation_history = self.conversation.to_string(include_system=False)

            # LLM 호출
            text, tool_calls = generate_with_tools_local(
                user_message=current_query,
                tools=tools,
                conversation_history=conversation_history
            )

            # Assistant 메시지 추가
            self.conversation.add_assistant_message(content=text, tool_calls=tool_calls)

            # 도구 호출이 없으면 직접 답변 (종료)
            if not tool_calls:
                return text if text else "답변을 생성할 수 없습니다."

            # 도구 실행
            last_result = None
            for tool_call in tool_calls:
                # 도구 실행
                last_result = self.executor.execute(tool_call.name, **tool_call.arguments)
                self.conversation.add_user_message(last_result)

                # final_answer면 종료
                if tool_call.name == "final_answer":
                    return last_result

            # 다음 턴 준비 (마지막 도구 결과가 다음 질문이 됨)
            current_query = last_result if last_result else current_query

        # max_turns 도달
        raise RuntimeError(
            f"최대 대화 턴 수({max_turns})에 도달했습니다. "
            "무한 루프를 방지하기 위해 종료합니다."
        )

    def _run_hardcoded(self, user_query: str) -> str:
        """
        Phase 1 하드코딩된 플로우 (하위 호환성)

        Args:
            user_query: 사용자 질문

        Returns:
            str: 최종 답변
        """
        # Step 1: RAG 검색 (하드코딩)
        rag_tool_call = ToolCall(
            name="rag_search",
            arguments={"query": user_query, "top_k": 5, "reranker_top_k": 2}
        )
        self.conversation.add_assistant_message(content=None, tool_calls=[rag_tool_call])

        # RAG 도구 실행
        rag_result = self.executor.execute(rag_tool_call.name, **rag_tool_call.arguments)
        self.conversation.add_user_message(rag_result)

        # Step 2: 최종 답변 (하드코딩)
        final_tool_call = ToolCall(
            name="final_answer",
            arguments={"answer": f"'{user_query}'에 대한 답변입니다. (하드코딩된 답변)"}
        )
        self.conversation.add_assistant_message(content=None, tool_calls=[final_tool_call])

        # 최종 답변 도구 실행
        final_result = self.executor.execute(final_tool_call.name, **final_tool_call.arguments)
        self.conversation.add_user_message(final_result)

        return final_result

    def get_conversation(self) -> Conversation:
        """
        현재 대화 히스토리를 반환합니다.

        Returns:
            Conversation: 대화 히스토리
        """
        return self.conversation

    def get_turn_count(self) -> int:
        """
        대화 턴 수를 반환합니다.

        Returns:
            int: 대화 턴 수
        """
        return self.conversation.count_turns()

    def reset(self):
        """대화 히스토리를 초기화합니다."""
        self.conversation = Conversation()

    def __repr__(self) -> str:
        """Orchestrator 문자열 표현"""
        return f"<Orchestrator(turns={self.get_turn_count()}, tools={self.executor.get_tool_names()})>"
