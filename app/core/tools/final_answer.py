"""
FinalAnswer 도구

이 모듈은 최종 답변을 생성하는 도구를 정의합니다.
"""

from app.core.tools.base import Tool
from typing import Dict, Any


class FinalAnswerTool(Tool):
    """
    최종 답변 도구

    Orchestrator가 사용자에게 최종 답변을 반환할 때 사용하는 도구입니다.

    Example:
        >>> tool = FinalAnswerTool()
        >>> result = tool.execute(answer="양력은 항공기를 띄우는 힘입니다.")
        >>> print(result)
        [TOOL_RESULT] 양력은 항공기를 띄우는 힘입니다.
    """

    def execute(self, **kwargs) -> str:
        """
        최종 답변을 반환합니다.

        Args:
            **kwargs: 도구 파라미터
                - answer (str): 최종 답변 내용

        Returns:
            str: [TOOL_RESULT] 프리픽스가 붙은 최종 답변

        Raises:
            ValueError: answer 파라미터가 없는 경우
        """
        answer = kwargs.get('answer')

        if answer is None:
            raise ValueError("answer 파라미터가 필요합니다.")

        return f"[TOOL_RESULT] {answer}"

    def get_schema(self) -> Dict[str, Any]:
        """
        도구의 스키마를 반환합니다.

        Returns:
            Dict[str, Any]: 도구 스키마
        """
        return {
            "name": "final_answer",
            "description": "사용자에게 최종 답변을 제공합니다. 모든 필요한 정보를 수집한 후 이 도구를 사용하여 답변을 완성합니다.",
            "parameters": {
                "answer": {
                    "type": "string",
                    "description": "사용자에게 제공할 최종 답변 내용"
                }
            },
            "required": ["answer"]
        }
