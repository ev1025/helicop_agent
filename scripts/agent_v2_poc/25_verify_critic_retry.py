"""
#016 - Critic retry 메커니즘 동작 검증.

#016 에서 multi_graph 의 critic_node 가 retry_count 증가 + needs_retry 사용하도록
강화. 실제로 retry 가 발동되는지 + 발동 시 답변이 개선되는지 검증.

검증 방법:
  - critic 의 score 임계값을 강제로 9 로 올림 (대부분 케이스에서 retry 발동)
  - 같은 질의를 multi_graph 로 실행
  - retry_count, 최종 답변, score 변화 관찰

대조군: 임계값 6 (기본) 으로 같은 질의 실행 → retry 없이 끝남.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Literal  # nested function type hint 평가용

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


QUERIES = [
    "베르누이 원리로 양력이 어떻게 발생하나요?",
    "Vortex Ring State 란 무엇인가요?",
]


def run_with_threshold(chat_model, threshold: int, q: str) -> dict:
    """multi_graph 의 critic threshold 를 monkey-patch 후 실행."""
    import app.core.agent_v2.multi_graph as mg
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END

    # 새 라우팅 함수 생성 (threshold 가 다름)
    def custom_route(state) -> Literal["researcher", "procedure", "__end__"]:
        if state["critique_score"] >= threshold:
            return END
        if state.get("retry_count", 0) >= mg.MAX_RETRIES:
            return END
        if state["route"] == "researcher":
            return "researcher"
        if state["route"] == "procedure":
            return "procedure"
        return END

    # build_multi_graph 의 라우팅 함수 패치는 어렵 → 직접 노드 만들고 그래프 재조립
    from langgraph.graph import StateGraph
    from app.core.agent_v2.multi_graph import MultiAgentState
    from app.core.agent_v2.agents import (
        supervisor as sv_mod, researcher as rs_mod,
        procedure as pr_mod, critic as cr_mod,
    )
    from langchain_core.messages import AIMessage, SystemMessage

    def supervisor_node(state):
        decision = sv_mod.route(chat_model, state["user_query"])
        return {"route": decision.route,
                "messages": [AIMessage(content=f"[Supervisor] {decision.reason}")]}

    def researcher_node(state):
        prior = state.get("critique_feedback") or None
        ans = rs_mod.run(chat_model, state["user_query"], prior_feedback=prior)
        return {"answer": ans, "messages": [AIMessage(content=f"[Researcher] {ans[:80]}")]}

    def procedure_node(state):
        prior = state.get("critique_feedback") or None
        ans = pr_mod.run(chat_model, state["user_query"], prior_feedback=prior)
        return {"answer": ans, "messages": [AIMessage(content=f"[Procedure] {ans[:80]}")]}

    def smalltalk_node(state):
        from langchain_core.messages import HumanMessage as HM
        ai = chat_model.invoke([
            SystemMessage(content="간결한 한국어로 친근하게 응답하세요."),
            HM(content=state["user_query"]),
        ])
        ans = str(ai.content) if hasattr(ai, "content") else str(ai)
        return {"answer": ans, "messages": [AIMessage(content=f"[Smalltalk] {ans[:80]}")]}

    def critic_node(state):
        if state["route"] == "smalltalk":
            return {"critique_score": 10}
        critique = cr_mod.evaluate(chat_model, state["user_query"], state["answer"])
        will_retry = critique.needs_retry or critique.score < threshold
        feedback_text = ""
        if will_retry:
            issues_str = "\n".join(f"- {it}" for it in (critique.issues or [])) or "- (특정 이슈 없음)"
            feedback_text = (
                f"이전 답변 평가 (점수 {critique.score}/10):\n"
                f"{issues_str}\n"
                f"평가 사유: {critique.reason}\n"
                f"위 문제점을 보완하여 다른 각도로 답변을 다시 작성하세요."
            )
        return {
            "critique_score": critique.score,
            "retry_count": state.get("retry_count", 0) + 1 if will_retry else state.get("retry_count", 0),
            "critique_feedback": feedback_text,
            "messages": [AIMessage(content=f"[Critic] score={critique.score}")],
        }

    g = StateGraph(MultiAgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("researcher", researcher_node)
    g.add_node("procedure", procedure_node)
    g.add_node("smalltalk", smalltalk_node)
    g.add_node("critic", critic_node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["route"], {
        "researcher": "researcher", "procedure": "procedure", "smalltalk": "smalltalk",
    })
    g.add_edge("researcher", "critic")
    g.add_edge("procedure", "critic")
    g.add_edge("smalltalk", "critic")
    g.add_conditional_edges("critic", custom_route,
                            {"researcher": "researcher", "procedure": "procedure", END: END})
    app = g.compile()

    initial = {
        "messages": [HumanMessage(content=q)],
        "user_query": q,
        "route": "", "answer": "", "critique_score": 0, "retry_count": 0,
        "critique_feedback": "",
    }
    t0 = time.perf_counter()
    final = app.invoke(initial, config={"recursion_limit": 12})
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": round(elapsed, 1),
        "route": final.get("route"),
        "score": final.get("critique_score"),
        "retry_count": final.get("retry_count", 0),
        "answer_length": len(final.get("answer") or ""),
        "answer": (final.get("answer") or "")[:200],
    }


def main():
    from app.core import models
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model_gguf

    print("=" * 70)
    print("[1/3] 임베딩 + Chroma + Qwen GGUF 로드")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()
    chat = build_qwen_chat_model_gguf(max_new_tokens=512)

    for thr in [6, 9]:
        print()
        print("=" * 70)
        print(f"[2/3] critic threshold = {thr} (>= threshold 면 종료)")
        print("=" * 70)
        for q in QUERIES:
            print(f"\n  {q}")
            r = run_with_threshold(chat, thr, q)
            print(f"    {r}")

    print()
    print("=" * 70)
    print("[3/3] 해석")
    print("=" * 70)
    print("  threshold=6: Critic score 6+ 면 종료 (대부분 첫 답변에서 끝남)")
    print("  threshold=9: Critic score 9+ 만 통과 (대부분 retry 발동, MAX_RETRIES=2 도달 시 종료)")


if __name__ == "__main__":
    main()
