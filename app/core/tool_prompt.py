"""
Tool Prompt Builder

로컬 LLM이 도구를 선택할 수 있도록 프롬프트를 구성하는 유틸리티입니다.
"""

from typing import List, Dict, Any


def build_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    도구 스키마를 LLM이 이해할 수 있는 프롬프트 문자열로 변환합니다.

    Args:
        tools: 도구 스키마 리스트 (ToolExecutor.get_all_schemas() 형식)

    Returns:
        str: 도구 설명과 사용 지침이 포함된 프롬프트

    Example:
        >>> from app.core.tool_executor import ToolExecutor
        >>> from app.core.tools import FinalAnswerTool, RAGSearchTool
        >>>
        >>> executor = ToolExecutor()
        >>> executor.register_tool(FinalAnswerTool())
        >>> executor.register_tool(RAGSearchTool())
        >>>
        >>> tools = executor.get_all_schemas()
        >>> prompt = build_tool_prompt(tools)
        >>> print(prompt)
    """
    if not tools:
        return ""

    lines = ["사용 가능한 도구:\n"]

    for i, tool in enumerate(tools, 1):
        name = tool["name"]
        description = tool["description"]
        parameters = tool.get("parameters", {})
        required = tool.get("required", [])

        lines.append(f"{i}. {name}")
        lines.append(f"   설명: {description}")

        if parameters:
            lines.append("   파라미터:")
            for param_name, param_info in parameters.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                is_required = "(필수)" if param_name in required else "(선택)"

                lines.append(f"   - {param_name} {is_required}, {param_type}: {param_desc}")

        lines.append("")  # 빈 줄

    # 사용 지침 추가
    lines.append("도구 사용 방법:")
    lines.append("도구를 사용하려면 다음 JSON 형식으로 응답하세요:")
    lines.append('```json')
    lines.append('{"tool": "도구이름", "arguments": {"인자이름": "값"}}')
    lines.append('```')
    lines.append("")
    lines.append("중요 규칙:")
    lines.append("- 질문에 답변하기 전 반드시 rag_search를 먼저 사용하세요 (필수)")
    lines.append("- rag_search 쿼리는 반드시 한국어로 작성하세요 (영어 금지)")
    lines.append("- 쿼리는 30자 이내로 간결하게 작성하세요 (핵심 키워드만)")
    lines.append("- 최종 답변을 제공할 때는 final_answer를 사용하세요")
    lines.append("- 복합 질문은 핵심 주제만 추출하여 검색하세요")

    return "\n".join(lines)


def parse_tool_call_from_response(response: str) -> tuple[str, dict]:
    """
    로컬 LLM의 응답에서 도구 호출을 파싱합니다.

    LLM이 JSON 형식으로 도구 호출을 응답한 경우, 이를 추출합니다.
    JSON 블록이 없으면 (None, None)을 반환합니다.

    Args:
        response: LLM 응답 텍스트

    Returns:
        (tool_name, arguments) 튜플
        - tool_name: 호출할 도구 이름 (없으면 None)
        - arguments: 도구 인자 딕셔너리 (없으면 None)

    Example:
        >>> response = '''
        ... 양력에 대해 검색하겠습니다.
        ... ```json
        ... {"tool": "rag_search", "arguments": {"query": "양력", "top_k": 5}}
        ... ```
        ... '''
        >>> tool_name, args = parse_tool_call_from_response(response)
        >>> print(tool_name)  # "rag_search"
        >>> print(args)       # {"query": "양력", "top_k": 5}
    """
    import re
    import json

    # 1단계: JSON 블록 찾기 (```json ... ``` 또는 ```{...}```)
    json_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
    match = re.search(json_pattern, response, re.DOTALL)

    json_str = None
    if match:
        json_str = match.group(1)
    else:
        # 2단계: 코드 블록이 없으면 bare JSON 찾기 (LLM이 코드 블록 없이 출력한 경우)
        # "tool"과 "arguments" 키를 포함하는 JSON 객체 찾기
        bare_json_pattern = r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        match = re.search(bare_json_pattern, response, re.DOTALL)
        if match:
            json_str = match.group(0)

    if not json_str:
        # JSON을 찾을 수 없으면 도구 호출 없음
        return None, None

    try:
        # JSON 파싱
        data = json.loads(json_str)

        # 도구 이름과 인자 추출
        tool_name = data.get("tool")
        arguments = data.get("arguments", {})

        if not tool_name:
            return None, None

        return tool_name, arguments

    except json.JSONDecodeError:
        # JSON 파싱 실패
        return None, None


def extract_text_without_tool_call(response: str) -> str:
    """
    LLM 응답에서 도구 호출 JSON 블록을 제외한 텍스트만 추출합니다.

    Args:
        response: LLM 응답 텍스트

    Returns:
        str: JSON 블록이 제거된 텍스트

    Example:
        >>> response = '''
        ... 양력에 대해 검색하겠습니다.
        ... ```json
        ... {"tool": "rag_search", "arguments": {"query": "양력"}}
        ... ```
        ... '''
        >>> text = extract_text_without_tool_call(response)
        >>> print(text)  # "양력에 대해 검색하겠습니다."
    """
    import re

    # 1단계: JSON 코드 블록 제거
    json_pattern = r'```(?:json)?\s*\{[^`]+\}\s*```'
    text = re.sub(json_pattern, '', response, flags=re.DOTALL)

    # 2단계: bare JSON 도구 호출 제거 (코드 블록 없는 경우)
    bare_json_pattern = r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
    text = re.sub(bare_json_pattern, '', text, flags=re.DOTALL)

    # 앞뒤 공백 제거
    return text.strip()
