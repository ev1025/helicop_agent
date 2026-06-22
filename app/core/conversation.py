"""
ChatGPT/Claude 스타일 대화 구조

이 모듈은 Messages API 기반의 대화 구조를 정의합니다.
- Message: system/user/assistant role을 가진 메시지
- ToolCall: 도구 호출 정보
- Conversation: 대화 히스토리 관리
"""

from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """
    도구 호출 정보

    LLM이 특정 도구를 사용하겠다고 결정했을 때 생성됩니다.

    Attributes:
        id: 도구 호출 고유 ID (선택사항)
        name: 도구 이름 (예: "rag_search", "decompose_question", "final_answer")
        arguments: 도구에 전달할 파라미터 딕셔너리

    Example:
        >>> tool_call = ToolCall(
        ...     name="rag_search",
        ...     arguments={"query": "양력 원리", "intent": "reason", "max_docs": 5}
        ... )
    """
    id: Optional[str] = None
    name: str
    arguments: Dict[str, Any]


class Message(BaseModel):
    """
    대화 메시지

    ChatGPT/Claude Messages API와 호환되는 메시지 구조입니다.

    Attributes:
        role: 메시지 역할 (system: 시스템 지침, user: 사용자/도구결과, assistant: AI응답)
        content: 메시지 내용 (텍스트)
        tool_calls: 도구 호출 목록 (assistant 역할일 때만 사용)

    Example:
        >>> # 사용자 메시지
        >>> msg = Message(role="user", content="양력이란 무엇인가요?")
        >>>
        >>> # 도구 호출을 포함한 어시스턴트 메시지
        >>> msg = Message(
        ...     role="assistant",
        ...     content="RAG 검색이 필요합니다.",
        ...     tool_calls=[ToolCall(name="rag_search", arguments={...})]
        ... )
    """
    role: Literal["system", "user", "assistant"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    def is_tool_call(self) -> bool:
        """도구 호출 메시지인지 확인"""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def is_tool_result(self) -> bool:
        """도구 결과 메시지인지 확인 (user role이며 [TOOL_RESULT]로 시작)"""
        return (
            self.role == "user"
            and self.content is not None
            and self.content.startswith("[TOOL_RESULT]")
        )

    def get_first_tool_call(self) -> Optional[ToolCall]:
        """첫 번째 도구 호출 반환"""
        if self.tool_calls and len(self.tool_calls) > 0:
            return self.tool_calls[0]
        return None


class Conversation(BaseModel):
    """
    대화 히스토리 관리

    여러 메시지를 순차적으로 관리하며, LLM API 호출에 필요한 형식으로 변환할 수 있습니다.

    Attributes:
        messages: 메시지 목록 (시간 순서대로)

    Example:
        >>> conv = Conversation()
        >>> conv.add_system_message("당신은 헬리콥터 전문가입니다.")
        >>> conv.add_user_message("양력이란?")
        >>> conv.add_assistant_message(
        ...     content=None,
        ...     tool_calls=[ToolCall(name="rag_search", arguments={...})]
        ... )
        >>> conv.add_user_message("[TOOL_RESULT] 문서1: ...")
        >>> conv.add_assistant_message(
        ...     content=None,
        ...     tool_calls=[ToolCall(name="final_answer", arguments={...})]
        ... )
    """
    messages: List[Message] = Field(default_factory=list)

    def add_system_message(self, content: str) -> None:
        """
        시스템 메시지 추가 (맨 앞에 위치)

        이미 시스템 메시지가 있으면 교체합니다.

        Args:
            content: 시스템 지침 내용
        """
        if self.messages and self.messages[0].role == "system":
            # 이미 시스템 메시지가 있으면 교체
            self.messages[0] = Message(role="system", content=content)
        else:
            # 없으면 맨 앞에 삽입
            self.messages.insert(0, Message(role="system", content=content))

    def add_user_message(self, content: str) -> None:
        """
        사용자 메시지 추가

        Args:
            content: 사용자 질문 또는 도구 실행 결과
        """
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> None:
        """
        어시스턴트 메시지 추가

        Args:
            content: AI의 텍스트 응답 (사고 과정, 설명 등)
            tool_calls: 실행할 도구 목록

        Note:
            content와 tool_calls 중 최소 하나는 있어야 합니다.
        """
        if content is None and (tool_calls is None or len(tool_calls) == 0):
            raise ValueError("Assistant message must have either content or tool_calls")

        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )

    def get_messages_dict(self) -> List[Dict[str, Any]]:
        """
        LLM API 호출용 딕셔너리 리스트 반환

        None 값은 제외됩니다.

        Returns:
            메시지 딕셔너리 리스트

        Example:
            >>> conv = Conversation()
            >>> conv.add_system_message("You are a helpful assistant.")
            >>> conv.add_user_message("Hello")
            >>> conv.get_messages_dict()
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        """
        return [msg.model_dump(exclude_none=True) for msg in self.messages]

    def get_last_assistant_message(self) -> Optional[Message]:
        """
        마지막 어시스턴트 메시지 반환

        Returns:
            마지막 assistant 역할 메시지, 없으면 None
        """
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def get_last_user_message(self) -> Optional[Message]:
        """
        마지막 사용자 메시지 반환

        Returns:
            마지막 user 역할 메시지, 없으면 None
        """
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    def count_turns(self) -> int:
        """
        대화 턴 수 계산 (assistant 메시지 개수)

        Returns:
            어시스턴트 메시지 개수
        """
        return sum(1 for msg in self.messages if msg.role == "assistant")

    def to_string(self, include_system: bool = False) -> str:
        """
        대화 내용을 사람이 읽기 편한 문자열로 변환

        Args:
            include_system: 시스템 메시지 포함 여부

        Returns:
            포맷팅된 대화 문자열
        """
        lines = []
        for msg in self.messages:
            if msg.role == "system" and not include_system:
                continue

            role_display = {
                "system": "[SYSTEM]",
                "user": "[USER]",
                "assistant": "[ASSISTANT]"
            }[msg.role]

            if msg.content:
                lines.append(f"{role_display} {msg.content[:200]}...")

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"  └─ TOOL: {tc.name}({tc.arguments})")

        return "\n".join(lines)
