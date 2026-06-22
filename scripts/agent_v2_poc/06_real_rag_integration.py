"""
실제 chroma_db + 임베딩 모델을 연결한 agent_v2 통합 테스트.

전제:
  scripts/agent_v2_poc/05_index_pdf_to_chroma.py 를 한 번 실행해
  chroma_db_new/ 가 만들어져 있어야 함.

흐름:
  1) 임베딩 모델 + Chroma DB 로드 (실제 RAG 활성화)
  2) Qwen 로드 (4-bit) + QwenChatModel
  3) app.core.agent_v2.tools.rag_search (실제 PDF 검색) + final_answer
  4) LangGraph 그래프로 단일 에이전트 ReAct 루프 실행
  5) 동일 질문을 평가용 metric 과 함께 기록 (속도, 응답 길이 등)

실행:
    .venv/Scripts/python.exe scripts/agent_v2_poc/06_real_rag_integration.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

# ────────────────────────────────────────────────────────────
# LangChain 메시지 / LangGraph 임포트
# ────────────────────────────────────────────────────────────
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
import operator


# ────────────────────────────────────────────────────────────
# AgentState (간단 inline)
# ────────────────────────────────────────────────────────────
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
    "당신은 헬리콥터 표준 교재를 기반으로 답변하는 항공 AI 튜터입니다.\n"
    "다음 규칙을 반드시 지키세요.\n"
    "1. 사용자 질문에 답하기 전 반드시 rag_search 도구로 관련 문서를 검색한다.\n"
    "2. 검색 결과를 받으면 그 내용에 근거하여 final_answer 도구로 답변을 전달한다.\n"
    "3. 답변은 한국어로, 핵심만 간결하게.\n"
    "4. 문서에 없는 내용은 추측하지 말고 \"문서에 충분한 정보가 없습니다\"라고 답한다."
)


def main():
    # ────────────────────────────────────────────────
    # [1] RAG 인프라 로드 (임베딩 모델 + Chroma DB)
    # ────────────────────────────────────────────────
    print("=" * 70)
    print("[1/4] 임베딩 모델 + Chroma DB 로드")
    print("=" * 70)
    from app.core import models
    t0 = time.perf_counter()
    models.load_embedding_model()
    t1 = time.perf_counter()
    print(f"  임베딩 로드: {(t1-t0):.2f}초")
    models.load_vector_db()
    t2 = time.perf_counter()
    print(f"  벡터 DB 로드: {(t2-t1):.2f}초")
    print(f"  collection: {models.collection}")
    print(f"  parent_store size: {len(models.parent_store) if models.parent_store else 0}")

    # ────────────────────────────────────────────────
    # [2] Qwen 모델 + QwenChatModel
    # ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("[2/4] Qwen2.5-7B 로드 (4-bit) + QwenChatModel 래핑")
    print("=" * 70)
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model

    t0 = time.perf_counter()
    chat = build_qwen_chat_model(model_name="Qwen/Qwen2.5-7B-Instruct")
    t1 = time.perf_counter()
    print(f"  Qwen 로드: {(t1-t0):.2f}초")

    # ────────────────────────────────────────────────
    # [3] 실제 RAG 도구 + LangGraph 그래프 조립
    # ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("[3/4] 실제 도구 + 그래프 조립")
    print("=" * 70)
    from app.core.agent_v2.tools import ALL_TOOLS  # 실제 rag_search/final_answer

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

    # ────────────────────────────────────────────────
    # [4] 질의 실행 (실제 RAG)
    # ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("[4/4] 질의 실행 (실제 PDF 검색)")
    print("=" * 70)

    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "헬리콥터 메인 로터의 피치 각도가 양력에 미치는 영향은?",
        "Vortex Ring State 란 무엇인가요?",
    ]

    results = []
    for i, q in enumerate(queries, 1):
        print()
        print("-" * 70)
        print(f"[Q{i}] {q}")
        print("-" * 70)

        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=q),
            ]
        }
        t_start = time.perf_counter()
        try:
            final_state = app.invoke(initial_state, config={"recursion_limit": 8})
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            continue
        t_end = time.perf_counter()
        elapsed = t_end - t_start

        # 메시지 요약 출력
        n_msgs = len(final_state["messages"])
        n_tool_calls = sum(
            1 for m in final_state["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
        )
        n_tool_results = sum(
            1 for m in final_state["messages"]
            if isinstance(m, ToolMessage)
        )

        # 최종 답변 추출
        final_answer_text = None
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "final_answer":
                final_answer_text = str(msg.content)
                break
        if final_answer_text is None:
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer_text = str(msg.content)
                    break
        final_answer_text = final_answer_text or "(답변 없음)"

        print(f"  소요: {elapsed:.2f}초")
        print(f"  메시지 수: {n_msgs} (tool_calls={n_tool_calls}, tool_results={n_tool_results})")
        print(f"  답변 길이: {len(final_answer_text)}자")
        print(f"  답변 (300자):")
        for line in final_answer_text[:300].split("\n"):
            print(f"    {line}")

        results.append({
            "query": q,
            "elapsed_sec": round(elapsed, 2),
            "n_messages": n_msgs,
            "n_tool_calls": n_tool_calls,
            "n_tool_results": n_tool_results,
            "answer_length": len(final_answer_text),
            "answer_preview": final_answer_text[:300],
        })

    # 요약
    print()
    print("=" * 70)
    print("실행 요약")
    print("=" * 70)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
