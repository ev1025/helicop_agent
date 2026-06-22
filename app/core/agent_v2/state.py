"""
LangGraph 상태 정의.

ReAct 루프의 단일 상태는 메시지 리스트 하나뿐이다.
- HumanMessage  : 사용자 질문
- AIMessage     : LLM 응답 (도구 호출 포함 가능)
- ToolMessage   : 도구 실행 결과
- (선택) SystemMessage

reducer (operator.add) 로 새 메시지를 항상 누적한다.
"""

import operator
from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
