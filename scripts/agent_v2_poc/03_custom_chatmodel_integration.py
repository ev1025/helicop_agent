"""
QwenChatModel (커스텀 LangChain ChatModel) + LangGraph 통합 검증.

02 스크립트와 동일한 흐름이지만 ChatHuggingFace 대신
app.core.agent_v2.qwen_chat_model.QwenChatModel 사용.

기대 결과:
  - LLM 응답에서 <tool_call> 태그가 정상 파싱되어 AIMessage.tool_calls 채워짐
  - LangGraph ToolNode 가 mock rag_search 를 실행해 ToolMessage 생성
  - 두 번째 LLM 호출에서 final_answer 도구를 호출하여 종료
  - 메시지 누적이 5~6개로 늘어나야 정상 (Sys, User, AI(rag_search), Tool, AI(final_answer), Tool)

실행:
    .venv/Scripts/python.exe scripts/agent_v2_poc/03_custom_chatmodel_integration.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

# ────────────────────────────────────────────────────────────
# Mock 도구 (실제 RAG/Chroma 우회)
# ────────────────────────────────────────────────────────────
from langchain_core.tools import tool


@tool
def rag_search(query: str, top_k: int = 5) -> str:
    """헬리콥터 표준 교재(PDF)에서 사용자 질문과 관련된 문서를 검색한다.

    질문에 답변하기 전에 이 도구를 먼저 호출해야 한다.

    Args:
        query: 검색할 한국어 키워드. 30자 이내 핵심 키워드만.
        top_k: 사용할 문서 수.

    Returns:
        JSON 문자열. {"documents": [...], "count": N} 형식.
    """
    fake_docs = [
        {
            "content": (
                "헬리콥터 메인 로터 블레이드는 에어포일 형상으로 설계되어 있어, "
                "회전 시 베르누이 정리에 따라 블레이드 상부의 공기 속도가 빨라지고 "
                "압력이 낮아진다. 이 압력 차이가 위쪽으로 향하는 양력을 만든다."
            ),
            "page": 17,
            "source": "조종사표준교재.pdf",
            "score": 0.87,
        },
        {
            "content": (
                "피치 각도가 증가하면 받음각이 커져 양력이 비례하여 증가한다. "
                "다만 임계각을 넘으면 실속(stall) 이 발생해 양력이 급격히 감소한다."
            ),
            "page": 19,
            "source": "조종사표준교재.pdf",
            "score": 0.71,
        },
    ]
    payload = {"documents": fake_docs, "count": len(fake_docs)}
    print(f"\n[mock rag_search] query={query!r}, 반환 {len(fake_docs)}개")
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def final_answer(answer: str) -> str:
    """사용자에게 최종 답변을 전달하고 대화를 종료한다.

    참고 문서 검색을 충분히 마친 뒤 호출한다.

    Args:
        answer: 사용자에게 보여줄 최종 답변 (한국어).

    Returns:
        입력한 answer 그대로.
    """
    print(f"\n[final_answer] {len(answer)}자")
    return answer


ALL_TOOLS = [rag_search, final_answer]

# ────────────────────────────────────────────────────────────
# 그래프 + LLM 로드
# ────────────────────────────────────────────────────────────
from app.core.agent_v2.llm_qwen import build_qwen_chat_model
from app.core.agent_v2.graph import build_graph
from app.core.agent_v2 import graph as graph_mod  # ALL_TOOLS 재바인딩 위해

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import BaseMessage
import operator


# 로컬 그래프 (ALL_TOOLS 를 mock 으로 교체하기 위해 직접 조립)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _route_after_llm(state):
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END
    return "tools"


def _route_after_tool(state):
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        if msg.name == "final_answer":
            return END
    return "llm"


SYSTEM_PROMPT = (
    "당신은 헬리콥터 표준 교재를 바탕으로 답변하는 항공 AI 튜터입니다.\n"
    "사용자 질문에 답하기 전 반드시 rag_search 도구를 호출해 관련 문서를 찾고, "
    "검색 결과를 바탕으로 final_answer 도구로 최종 답변을 전달하세요.\n"
    "- 추측 금지. 문서에 없는 내용은 '문서에 충분한 정보가 없습니다'라고 답하세요.\n"
    "- 답변은 한국어로 간결하게 작성하세요."
)


def main():
    print("[1/3] QwenChatModel 로드 중...")
    chat = build_qwen_chat_model(model_name="Qwen/Qwen2.5-7B-Instruct")

    print("[2/3] bind_tools + LangGraph 조립")
    llm_with_tools = chat.bind_tools(ALL_TOOLS)

    def call_llm(state):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(ALL_TOOLS)
    g = StateGraph(AgentState)
    g.add_node("llm", call_llm)
    g.add_node("tools", tool_node)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", _route_after_llm)
    g.add_conditional_edges("tools", _route_after_tool)
    app = g.compile()

    print("[3/3] 질의 실행")
    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "피치 각도가 양력에 어떤 영향을 주나요?",
    ]

    for i, q in enumerate(queries, 1):
        print()
        print("=" * 70)
        print(f"[테스트 {i}] {q}")
        print("=" * 70)

        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=q),
            ]
        }
        try:
            final_state = app.invoke(initial_state, config={"recursion_limit": 8})
        except Exception as e:
            import traceback
            print(f"[ERROR] 그래프 실행 실패: {e}")
            traceback.print_exc()
            continue

        print(f"\n  메시지 누적: {len(final_state['messages'])}개")
        for j, msg in enumerate(final_state["messages"]):
            kind = type(msg).__name__
            preview = str(msg.content)[:140].replace("\n", " ")
            tool_info = ""
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_info = f" [tool_calls={[(tc['name'], list(tc['args'].keys())) for tc in msg.tool_calls]}]"
            elif isinstance(msg, ToolMessage):
                tool_info = f" [name={msg.name}]"
            print(f"  [{j}] {kind}{tool_info}: {preview}")

        # 최종 답변 추출
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "final_answer":
                print(f"\n  ✅ 최종 답변: {msg.content[:300]}...")
                break
        else:
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n  ⚠️  LLM 직답 (도구 미호출): {msg.content[:300]}")
                    break


if __name__ == "__main__":
    main()
