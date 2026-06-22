"""
ToolExecutor

이 모듈은 도구 이름으로 도구를 찾아 실행하는 엔진을 정의합니다.
"""

from typing import Dict, Any, Optional, List
from app.core.tools.base import Tool


class ToolExecutor:
    """
    도구 실행 엔진

    도구 이름으로 도구를 찾아 실행합니다.
    여러 도구를 등록하고 이름으로 호출할 수 있습니다.

    Example:
        >>> executor = ToolExecutor()
        >>> executor.register_tool(FinalAnswerTool())
        >>> executor.register_tool(RAGSearchTool())
        >>> result = executor.execute("final_answer", answer="답변입니다")
        >>> print(result)
        [TOOL_RESULT] 답변입니다
    """

    def __init__(self):
        """ToolExecutor 초기화"""
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """
        도구를 등록합니다.

        Args:
            tool: 등록할 도구 인스턴스

        Raises:
            ValueError: 도구 이름이 이미 등록되어 있는 경우
        """
        tool_name = tool.get_schema()["name"]

        if tool_name in self._tools:
            raise ValueError(f"도구 '{tool_name}'가 이미 등록되어 있습니다.")

        self._tools[tool_name] = tool

    def unregister_tool(self, tool_name: str) -> bool:
        """
        도구 등록을 해제합니다.

        Args:
            tool_name: 등록 해제할 도구 이름

        Returns:
            bool: 등록 해제 성공 여부
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        도구를 가져옵니다.

        Args:
            tool_name: 도구 이름

        Returns:
            Optional[Tool]: 도구 인스턴스 (없으면 None)
        """
        return self._tools.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """
        도구가 등록되어 있는지 확인합니다.

        Args:
            tool_name: 도구 이름

        Returns:
            bool: 등록 여부
        """
        return tool_name in self._tools

    def get_all_tools(self) -> Dict[str, Tool]:
        """
        등록된 모든 도구를 반환합니다.

        Returns:
            Dict[str, Tool]: 도구 이름 → 도구 인스턴스 딕셔너리
        """
        return self._tools.copy()

    def get_tool_names(self) -> List[str]:
        """
        등록된 모든 도구 이름을 반환합니다.

        Returns:
            List[str]: 도구 이름 리스트
        """
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        등록된 모든 도구의 스키마를 반환합니다.

        Returns:
            List[Dict[str, Any]]: 도구 스키마 리스트
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs) -> str:
        """
        도구를 실행합니다.

        Args:
            tool_name: 실행할 도구 이름
            **kwargs: 도구 실행 파라미터

        Returns:
            str: 도구 실행 결과

        Raises:
            ValueError: 도구가 등록되어 있지 않은 경우
            ValueError: 파라미터 검증 실패 시
        """
        if tool_name not in self._tools:
            raise ValueError(
                f"도구 '{tool_name}'가 등록되어 있지 않습니다. "
                f"등록된 도구: {list(self._tools.keys())}"
            )

        tool = self._tools[tool_name]

        # 파라미터 검증
        if not tool.validate_parameters(**kwargs):
            schema = tool.get_schema()
            required_params = schema.get("required", [])
            raise ValueError(
                f"도구 '{tool_name}'의 필수 파라미터가 누락되었습니다. "
                f"필수 파라미터: {required_params}"
            )

        # 도구 실행
        return tool.execute(**kwargs)

    def __repr__(self) -> str:
        """ToolExecutor 문자열 표현"""
        return f"<ToolExecutor(tools={list(self._tools.keys())})>"
