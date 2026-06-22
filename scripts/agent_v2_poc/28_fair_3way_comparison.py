"""
#022 - 진짜 fair 3-way 비교: legacy orchestrator vs LangGraph 단일 vs 멀티.

같은 환경 통일:
  모델       Qwen 7B GGUF Q4_K_M (llama.cpp)
  RAG        Parent-Child + 본문 순서 정렬 + Sentence-Window=1
  도구       rag_search + final_answer
  하드웨어    RTX 4060 8GB

3 엔진 비교:
  A. orchestrator.py (prompt + 정규식 파싱, monkey-patch 로 GGUF 사용)
  B. graph.py        (LangGraph 단일 ReAct, Qwen native tool calling)
  C. multi_graph.py  (LangGraph Supervisor + Workers + Critic + feedback)

이렇게 해야 "기존 수동 구현" vs "LangGraph 도입" 효과를 환경 차이 없이 측정 가능.

법칙: orchestrator 가 호출하는 generate_with_tools_local 만 GGUF 어댑터로 교체.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


QUERIES = [
    {"id": "Q1", "query": "베르누이 원리로 양력이 어떻게 발생하나요?",
     "expected_keywords": ["베르누이", "압력", "속도", "양력"]},
    {"id": "Q2", "query": "Vortex Ring State 란 무엇인가요?",
     "expected_keywords": ["하강", "공기", "양력", "수직"]},
    {"id": "Q3", "query": "동적 롤오버는 어떤 상황에서 발생하나요?",
     "expected_keywords": ["기울", "회전", "지면", "측"]},
    {"id": "Q4", "query": "헬리콥터 시동 절차를 단계별로 설명해주세요.",
     "expected_keywords": ["단계", "확인", "시동"]},
    {"id": "Q5", "query": "오토자이로와 헬리콥터의 차이가 뭔가요?",
     "expected_keywords": ["로터", "동력", "차이"]},
]


def keyword_recall(answer: str, expected: list) -> float:
    if not expected:
        return 0.0
    return sum(1 for k in expected if k in answer) / len(expected)


# ─────────────────────────────────────────────────
# orchestrator 의 generate_with_tools_local 를 GGUF 로 우회.
# 시그니처: (user_message, tools, conversation_history, max_new_tokens) -> (text, tool_calls)
# ─────────────────────────────────────────────────
def make_gguf_generate(gguf_llm):
    """orchestrator 가 호출하는 함수를 GGUF 백엔드로 만든다."""
    from app.core.tool_prompt import build_tool_prompt
    from app.core.conversation import ToolCall

    _TOOL_CALL_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", re.DOTALL)
    _BARE_RE = re.compile(r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[\s\S]*?\}\s*\}', re.DOTALL)

    def fn(user_message, tools, conversation_history=None, max_new_tokens=1024):
        tool_prompt = build_tool_prompt(tools) if tools else ""
        if conversation_history:
            full = f"{conversation_history}\n\n{tool_prompt}\n\n사용자: {user_message}\n\n어시스턴트:"
        else:
            full = f"{tool_prompt}\n\n사용자: {user_message}\n\n어시스턴트:"

        messages = [
            {"role": "system", "content": "당신은 헬리콥터 AI 튜터입니다."},
            {"role": "user", "content": full},
        ]
        try:
            resp = gguf_llm.create_chat_completion(
                messages=messages, max_tokens=max_new_tokens, temperature=0.0,
            )
            text = resp["choices"][0]["message"].get("content") or ""
        except Exception as e:
            return (f"GGUF 호출 실패: {e}", [])

        # 정규식 파싱 (orchestrator 의 원래 로직)
        m = _TOOL_CALL_RE.search(text)
        json_str = None
        if m:
            json_str = m.group(1)
        else:
            m2 = _BARE_RE.search(text)
            if m2:
                json_str = m2.group(0)

        if not json_str:
            return (text, [])
        try:
            data = json.loads(json_str)
            tname = data.get("tool")
            args = data.get("arguments", {}) or {}
            if not isinstance(args, dict):
                args = {}
            if not tname:
                return (text, [])
            return (text, [ToolCall(name=tname, arguments=args)])
        except json.JSONDecodeError:
            return (text, [])

    return fn


# ─────────────────────────────────────────────────
# A. orchestrator (legacy) on GGUF
# ─────────────────────────────────────────────────
def run_orchestrator_gguf(query: str, gguf_llm) -> dict:
    from app.core import llm as llm_mod
    from app.core.orchestrator import Orchestrator
    from app.core.tool_executor import ToolExecutor
    from app.core.tools import RAGSearchTool, FinalAnswerTool

    orig = llm_mod.generate_with_tools_local
    llm_mod.generate_with_tools_local = make_gguf_generate(gguf_llm)
    try:
        executor = ToolExecutor()
        executor.register_tool(RAGSearchTool())
        executor.register_tool(FinalAnswerTool())
        orch = Orchestrator(executor)

        t0 = time.perf_counter()
        try:
            answer = orch.run(query, max_turns=8)
            err = None
        except Exception as e:
            answer = f"[ERROR] {e}"
            err = str(e)
        elapsed = time.perf_counter() - t0
        n_msgs = orch.get_turn_count() or 0
    finally:
        llm_mod.generate_with_tools_local = orig

    return {"engine": "orchestrator_legacy_gguf", "elapsed_sec": round(elapsed, 2),
            "n_messages": n_msgs, "answer": answer, "answer_length": len(answer), "error": err}


# ─────────────────────────────────────────────────
# B. graph.py (LangGraph 단일 ReAct + Qwen native)
# ─────────────────────────────────────────────────
def run_single_langgraph(query: str, chat_model) -> dict:
    from app.core.agent_v2.graph import build_graph
    from app.core.agent_v2.tools import ALL_TOOLS
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    SYS = "헬리콥터 AI 튜터. rag_search 후 final_answer. 한국어 간결."
    llm = chat_model.bind_tools(ALL_TOOLS)
    app = build_graph(llm)

    t0 = time.perf_counter()
    try:
        state = app.invoke({"messages": [SystemMessage(content=SYS), HumanMessage(content=query)]},
                           config={"recursion_limit": 8})
        err = None
    except Exception as e:
        return {"engine": "langgraph_single", "elapsed_sec": round(time.perf_counter()-t0, 2),
                "n_messages": 0, "answer": f"[ERROR] {e}", "answer_length": 0, "error": str(e)}
    elapsed = time.perf_counter() - t0

    answer = ""
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "final_answer":
            answer = str(m.content); break
    else:
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content:
                answer = str(m.content); break
    return {"engine": "langgraph_single", "elapsed_sec": round(elapsed, 2),
            "n_messages": len(state["messages"]), "answer": answer,
            "answer_length": len(answer), "error": None}


# ─────────────────────────────────────────────────
# C. multi_graph.py
# ─────────────────────────────────────────────────
def run_multi_langgraph(query: str, chat_model) -> dict:
    from app.core.agent_v2.multi_graph import run_multi_agent
    t0 = time.perf_counter()
    try:
        r = run_multi_agent(chat_model, query)
        err = None
    except Exception as e:
        return {"engine": "langgraph_multi", "elapsed_sec": round(time.perf_counter()-t0, 2),
                "n_messages": 0, "answer": f"[ERROR] {e}", "answer_length": 0, "error": str(e)}
    elapsed = time.perf_counter() - t0
    return {"engine": "langgraph_multi", "elapsed_sec": round(elapsed, 2),
            "n_messages": r.get("n_messages", 0), "answer": r.get("answer", ""),
            "answer_length": len(r.get("answer", "")),
            "route": r.get("route"), "critic_score": r.get("score"),
            "error": None}


def main():
    print("=" * 70)
    print("[1/3] 임베딩 + Chroma + Qwen GGUF 로드")
    print("=" * 70)
    from app.core import models
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model_gguf

    models.load_embedding_model()
    models.load_vector_db()
    chat = build_qwen_chat_model_gguf(max_new_tokens=512)
    # legacy orchestrator 가 사용할 raw GGUF 도 같은 인스턴스 공유
    raw_llm = chat._llm

    # ─────────────────────────────────────
    print()
    print("=" * 70)
    print(f"[2/3] {len(QUERIES)} 질의 × 3 엔진 측정")
    print("=" * 70)

    rows = []
    for item in QUERIES:
        print(f"\n  [{item['id']}] {item['query']}")
        for runner, name, args in [
            (run_orchestrator_gguf, "A. orchestrator (legacy)", (item["query"], raw_llm)),
            (run_single_langgraph,  "B. langgraph 단일", (item["query"], chat)),
            (run_multi_langgraph,   "C. langgraph 멀티", (item["query"], chat)),
        ]:
            r = runner(*args)
            r["qid"] = item["id"]; r["query"] = item["query"]
            r["keyword_recall"] = round(keyword_recall(r["answer"], item["expected_keywords"]), 2)
            rows.append(r)
            mark = "❌" if r.get("error") else "✅"
            extras = ""
            if r["engine"] == "langgraph_multi":
                extras = f" route={r.get('route')} critic={r.get('critic_score')}"
            print(f"    {mark} {name:25s}  {r['elapsed_sec']:6.2f}s  msgs={r['n_messages']:>2}  "
                  f"len={r['answer_length']:>3}  kw={r['keyword_recall']}{extras}")

    # ─────────────────────────────────────
    print()
    print("=" * 70)
    print("[3/3] 종합")
    print("=" * 70)

    by_engine = {}
    for r in rows:
        by_engine.setdefault(r["engine"], []).append(r)

    print(f"\n{'engine':30s} | {'avg sec':>8s} | {'avg kw':>6s} | {'avg len':>7s} | {'errors':>6s}")
    print("-" * 70)
    summary = {}
    for eng, vrows in by_engine.items():
        n = len(vrows)
        ok = [v for v in vrows if not v.get("error")]
        if not ok:
            print(f"{eng:30s} | (모두 에러)")
            summary[eng] = {"n": n, "errors": n}
            continue
        avg_sec = sum(v["elapsed_sec"] for v in ok) / len(ok)
        avg_kw = sum(v["keyword_recall"] for v in ok) / len(ok)
        avg_len = sum(v["answer_length"] for v in ok) / len(ok)
        n_err = n - len(ok)
        summary[eng] = {"n": n, "n_ok": len(ok), "n_errors": n_err,
                        "avg_elapsed_sec": round(avg_sec, 2),
                        "avg_keyword_recall": round(avg_kw, 3),
                        "avg_answer_length": round(avg_len, 0)}
        print(f"{eng:30s} | {avg_sec:>8.2f} | {avg_kw:>6.3f} | {avg_len:>7.0f} | {n_err:>6d}")

    out = ROOT / "results" / "agent_v2_changes" / "22_fair_3way_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
