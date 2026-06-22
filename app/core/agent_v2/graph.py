"""
LangGraph StateGraph 조립.

흐름:

    [사용자 질문]
          │
          ▼
   ┌──────────────┐
   │   llm 노드   │ ◄────┐
   │ (Qwen 호출)  │      │
   └──────┬───────┘      │
          │              │
          ▼              │
   [도구 호출 있음?]     │
       /    \\           │
     예      아니오      │
      │        │         │
      ▼        ▼         │
 [final_  [응답 텍스트  │
  answer?] 그대로 반환] │
   /  \\                  │
 예    아니오             │
  │      │                │
  ▼      ▼                │
 END   tools 노드 ────────┘
        (rag_search 등 실행)

- 도구가 final_answer 면 그래프 종료 (END)
- 도구가 그 외(rag_search) 면 결과를 메시지에 추가하고 다시 LLM 호출
- LLM 이 도구 호출 없이 텍스트만 답하면 그대로 종료
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.agent_v2.state import AgentState
from app.core.agent_v2.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


def _route_after_llm(state: AgentState) -> Literal["tools", "__end__"]:
    """LLM 응답을 보고 도구 실행으로 갈지 종료할지 결정."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return END

    if not last.tool_calls:
        # LLM 이 도구 호출 없이 일반 텍스트로 답한 경우 → 종료
        return END

    # final_answer 가 호출되면 종료. 도구 자체는 ToolNode 가 한 번 실행 후
    # 다시 LLM 으로 가지 않도록 라우팅을 분기.
    for call in last.tool_calls:
        if call["name"] == "final_answer":
            # ToolNode 가 final_answer 를 실행하도록 일단 tools 로 보낸 뒤
            # _route_after_tool 에서 다시 종료.
            return "tools"

    return "tools"


def _route_after_tool(state: AgentState) -> Literal["llm", "__end__"]:
    """ToolNode 실행 직후 라우팅. final_answer 결과가 있으면 종료."""
    # 마지막 ToolMessage 들 중 final_answer 가 있는지 확인
    # ToolNode 는 같은 step 의 모든 tool_call 을 실행하고 ToolMessage 들을 추가
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) != "tool":
            break
        if getattr(msg, "name", None) == "final_answer":
            return END
    return "llm"


def build_graph(llm_with_tools):
    """LangGraph StateGraph 를 조립하고 컴파일된 앱을 반환.

    Args:
        llm_with_tools: ChatHuggingFace.bind_tools(ALL_TOOLS) 결과.

    Returns:
        compiled graph (LangGraph CompiledGraph).
    """

    def call_llm(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", _route_after_llm)
    graph.add_conditional_edges("tools", _route_after_tool)

    app = graph.compile()
    logger.info("[agent_v2] LangGraph 조립 완료")
    return app
