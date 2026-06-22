"""
Tool 추상 베이스 클래스

이 모듈은 RAG Orchestrator에서 사용할 도구들의 추상 베이스 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Tool(ABC):
    """
    도구 추상 베이스 클래스

    모든 도구는 이 클래스를 상속받아 구현해야 합니다.

    Example:
        >>> class MyTool(Tool):
        ...     def execute(self, **kwargs):
        ...         return "result"
        ...
        ...     def get_schema(self):
        ...         return {
        ...             "name": "my_tool",
        ...             "description": "My custom tool",
        ...             "parameters": {...}
        ...         }
    """

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        도구를 실행합니다.

        Args:
            **kwargs: 도구별 파라미터

        Returns:
            str: 도구 실행 결과 (텍스트 형식)

        Note:
            결과는 항상 문자열로 반환되어야 합니다.
            [TOOL_RESULT] 프리픽스를 포함할 수 있습니다.
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        도구의 스키마를 반환합니다.

        Returns:
            Dict[str, Any]: 도구 스키마 (name, description, parameters 포함)

        Example:
            >>> {
            ...     "name": "rag_search",
            ...     "description": "RAG 검색 도구",
            ...     "parameters": {
            ...         "query": {
            ...             "type": "string",
            ...             "description": "검색 쿼리"
            ...         }
            ...     },
            ...     "required": ["query"]
            ... }
        """
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """
        파라미터 유효성을 검증합니다.

        Args:
            **kwargs: 검증할 파라미터

        Returns:
            bool: 유효하면 True, 아니면 False

        Note:
            기본 구현은 항상 True를 반환합니다.
            필요한 경우 하위 클래스에서 오버라이드하세요.
        """
        schema = self.get_schema()
        required_params = schema.get("required", [])

        for param in required_params:
            if param not in kwargs:
                return False

        return True

    def __repr__(self) -> str:
        """도구의 문자열 표현"""
        schema = self.get_schema()
        return f"<{self.__class__.__name__}: {schema.get('name', 'unknown')}>"
