"""
멀티에이전트 LangGraph 조립.

흐름:
                 ┌────────────┐
   사용자 질문 ─►│ Supervisor │
                 │ (라우팅)   │
                 └─┬──┬──┬───┘
                   │  │  │
          ┌────────┘  │  └────────┐
          ▼           ▼           ▼
      Researcher  Procedure   Smalltalk
          │           │           │
          └─────┬─────┘           │
                ▼                 │
            ┌──────┐              │
            │Critic│              │
            └──┬───┘              │
               │                  │
        retry? │ ───┐             │
       ┌──────┘    │              │
       ▼           │              ▼
   원래 워커     END             END
                                  ↑
                                답변

기존 단일 에이전트 (graph.py) 와는 별도 파일로 둠.
runner.py 가 어느 그래프를 쓸지 선택.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal, Sequence, TypedDict
import operator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.agent_v2.agents import supervisor as sv_mod
from app.core.agent_v2.agents import researcher as rs_mod
from app.core.agent_v2.agents import procedure as pr_mod
from app.core.agent_v2.agents import critic as cr_mod

logger = logging.getLogger(__name__)


class MultiAgentState(TypedDict):
    """멀티에이전트 그래프 상태."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    route: str  # 'researcher' / 'procedure' / 'smalltalk'
    answer: str  # 최종 답변
    critique_score: int  # 0~10
    retry_count: int
    # CHANGES.md #016-A: critic 피드백 → worker 로 전달 (retry 시 답변 개선용)
    critique_feedback: str  # 이전 critic 의 issues+reason 텍스트


MAX_RETRIES = 2


def build_multi_graph(chat_model):
    """멀티에이전트 그래프 컴파일."""

    def supervisor_node(state: MultiAgentState):
        decision = sv_mod.route(chat_model, state["user_query"])
        return {
            "route": decision.route,
            "messages": [AIMessage(content=f"[Supervisor] {decision.reason}")],
        }

    def researcher_node(state: MultiAgentState):
        prior = state.get("critique_feedback") or None
        ans = rs_mod.run(chat_model, state["user_query"], prior_feedback=prior)
        return {
            "answer": ans,
            "messages": [AIMessage(content=f"[Researcher] {ans[:80]}...")],
        }

    def procedure_node(state: MultiAgentState):
        prior = state.get("critique_feedback") or None
        ans = pr_mod.run(chat_model, state["user_query"], prior_feedback=prior)
        return {
            "answer": ans,
            "messages": [AIMessage(content=f"[Procedure] {ans[:80]}...")],
        }

    def smalltalk_node(state: MultiAgentState):
        # 잡담은 RAG 없이 LLM 한 번만
        from langchain_core.messages import HumanMessage
        ai = chat_model.invoke([
            SystemMessage(content="간결한 한국어로 친근하게 응답하세요."),
            HumanMessage(content=state["user_query"]),
        ])
        ans = ai.content if hasattr(ai, "content") else str(ai)
        return {"answer": str(ans), "messages": [AIMessage(content=f"[Smalltalk] {ans[:80]}...")]}

    def critic_node(state: MultiAgentState):
        # smalltalk 은 critic 건너뛰기
        if state["route"] == "smalltalk":
            return {"critique_score": 10}
        critique = cr_mod.evaluate(chat_model, state["user_query"], state["answer"])
        will_retry = critique.needs_retry or critique.score < 6
        # CHANGES.md #016-A: retry 시 worker 가 참고할 feedback 텍스트 구성.
        feedback_text = ""
        if will_retry:
            issues_str = "\n".join(f"- {it}" for it in (critique.issues or [])) or "- (특정 이슈 없음)"
            feedback_text = (
                f"이전 답변에 대한 평가 (점수 {critique.score}/10):\n"
                f"{issues_str}\n"
                f"평가 사유: {critique.reason}\n"
                f"위 문제점을 보완하여 다른 각도로 답변을 다시 작성하세요."
            )
        return {
            "critique_score": critique.score,
            "retry_count": state.get("retry_count", 0) + 1 if will_retry else state.get("retry_count", 0),
            "critique_feedback": feedback_text,
            "messages": [
                AIMessage(content=(
                    f"[Critic] score={critique.score} retry={critique.needs_retry} "
                    f"issues={critique.issues}"
                ))
            ],
        }

    # 라우팅 함수
    def route_after_supervisor(state) -> Literal["researcher", "procedure", "smalltalk"]:
        return state["route"]

    def route_after_critic(state) -> Literal["researcher", "procedure", "__end__"]:
        # CHANGES.md #016: 점수 충분(>=6)이면 즉시 종료. 그 외엔 retry.
        # retry_count 가 MAX_RETRIES 도달 시 더 이상 시도 안 하고 현재 답변으로 종료.
        if state["critique_score"] >= 6:
            return END
        if state.get("retry_count", 0) >= MAX_RETRIES:
            logger.info(f"[multi_graph] MAX_RETRIES({MAX_RETRIES}) 도달, 현재 답변으로 종료")
            return END
        # retry — 같은 워커로 다시
        if state["route"] == "researcher":
            return "researcher"
        if state["route"] == "procedure":
            return "procedure"
        return END

    g = StateGraph(MultiAgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("researcher", researcher_node)
    g.add_node("procedure", procedure_node)
    g.add_node("smalltalk", smalltalk_node)
    g.add_node("critic", critic_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor, {
        "researcher": "researcher",
        "procedure": "procedure",
        "smalltalk": "smalltalk",
    })
    g.add_edge("researcher", "critic")
    g.add_edge("procedure", "critic")
    g.add_edge("smalltalk", "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "researcher": "researcher",
        "procedure": "procedure",
        END: END,
    })

    return g.compile()


def run_multi_agent(chat_model, user_query: str) -> dict:
    """단일 질의에 대해 멀티에이전트 실행. 답변 + 메타 반환."""
    app = build_multi_graph(chat_model)
    initial = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "route": "",
        "answer": "",
        "critique_score": 0,
        "retry_count": 0,
    }
    final_state = app.invoke(initial, config={"recursion_limit": 12})
    return {
        "answer": final_state.get("answer", ""),
        "route": final_state.get("route", ""),
        "score": final_state.get("critique_score", 0),
        "n_messages": len(final_state.get("messages", [])),
    }
