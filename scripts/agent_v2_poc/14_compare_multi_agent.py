"""
#006 - 단일 ReAct vs Supervisor+Workers 멀티에이전트 비교.

비교 대상:
  A. 단일 에이전트 (graph.py: rag_search → final_answer 한 사이클)
  B. 멀티에이전트 (multi_graph.py: Supervisor → Worker → Critic [retry] → END)

측정 metric:
  - 평균 응답 시간
  - 평균 LLM 호출 수 (메시지 수 proxy)
  - 답변 길이
  - Critic 점수 (B 만)
  - 라우팅 분포 (B 만 — researcher/procedure/smalltalk)

LLM 호출이 많아 시간 오래 걸림 (예상: 단일 ~6분/쿼리, 멀티 ~12~20분/쿼리).
질의 수를 5개 → 3개로 축소 권장 (테스트 시).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


# 시간 절약을 위해 3개 (사실/절차/잡담 각 1)
QUERIES = [
    ("사실",  "베르누이 원리로 양력이 어떻게 발생하나요?"),
    ("절차",  "헬리콥터 시동을 거는 절차를 단계별로 설명해줘"),
    ("잡담",  "안녕하세요"),
]


def main():
    from app.core import models
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model
    from app.core.agent_v2.multi_graph import run_multi_agent
    from app.core.agent_v2.tools import ALL_TOOLS

    print("=" * 70)
    print("[1/4] 임베딩 + Chroma + Qwen 로드")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()
    chat = build_qwen_chat_model(model_name="Qwen/Qwen2.5-7B-Instruct")

    # 단일 에이전트 그래프 (기존)
    from app.core.agent_v2.graph import build_graph
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

    SINGLE_SYSTEM = (
        "당신은 헬리콥터 표준 교재 기반 AI 튜터입니다. "
        "rag_search 로 검색 후 final_answer 로 답변하세요. 한국어 간결."
    )
    llm_with_tools = chat.bind_tools(ALL_TOOLS)
    single_app = build_graph(llm_with_tools)

    # ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("[2/4] 단일 에이전트 (graph.py)")
    print("=" * 70)
    single_results = []
    for kind, q in QUERIES:
        print(f"\n[{kind}] {q}")
        t0 = time.perf_counter()
        try:
            state = single_app.invoke({
                "messages": [SystemMessage(content=SINGLE_SYSTEM), HumanMessage(content=q)]
            }, config={"recursion_limit": 8})
            elapsed = time.perf_counter() - t0
            ans = ""
            for m in reversed(state["messages"]):
                if isinstance(m, ToolMessage) and m.name == "final_answer":
                    ans = str(m.content); break
            else:
                for m in reversed(state["messages"]):
                    if isinstance(m, AIMessage) and m.content:
                        ans = str(m.content); break
            n_msgs = len(state["messages"])
            print(f"  소요 {elapsed:.1f}s, 메시지 {n_msgs}개, 답변 {len(ans)}자")
            single_results.append({"kind": kind, "query": q, "elapsed_sec": round(elapsed, 1),
                                   "n_messages": n_msgs, "answer_length": len(ans),
                                   "answer_preview": ans[:200]})
        except Exception as e:
            print(f"  [ERROR] {e}")
            single_results.append({"kind": kind, "query": q, "error": str(e)})

    # ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("[3/4] 멀티에이전트 (multi_graph.py)")
    print("=" * 70)
    multi_results = []
    for kind, q in QUERIES:
        print(f"\n[{kind}] {q}")
        t0 = time.perf_counter()
        try:
            r = run_multi_agent(chat, q)
            elapsed = time.perf_counter() - t0
            print(
                f"  소요 {elapsed:.1f}s, route={r['route']}, score={r['score']}, "
                f"메시지 {r['n_messages']}개, 답변 {len(r['answer'])}자"
            )
            multi_results.append({
                "kind": kind, "query": q, "elapsed_sec": round(elapsed, 1),
                "route": r["route"], "score": r["score"],
                "n_messages": r["n_messages"], "answer_length": len(r["answer"]),
                "answer_preview": r["answer"][:200],
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERROR] {e}")
            multi_results.append({"kind": kind, "query": q, "error": str(e)})

    # ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("[4/4] 비교")
    print("=" * 70)

    def avg(rows, key):
        valid = [r[key] for r in rows if key in r and not r.get("error")]
        return sum(valid) / len(valid) if valid else 0

    avg_single_t = avg(single_results, "elapsed_sec")
    avg_multi_t = avg(multi_results, "elapsed_sec")
    avg_single_msgs = avg(single_results, "n_messages")
    avg_multi_msgs = avg(multi_results, "n_messages")
    avg_single_len = avg(single_results, "answer_length")
    avg_multi_len = avg(multi_results, "answer_length")

    print(f"{'metric':24s} | {'single':>8s} | {'multi':>8s}")
    print("-" * 50)
    print(f"{'평균 elapsed (s)':24s} | {avg_single_t:>8.1f} | {avg_multi_t:>8.1f}")
    print(f"{'평균 메시지 수':24s} | {avg_single_msgs:>8.1f} | {avg_multi_msgs:>8.1f}")
    print(f"{'평균 답변 길이':24s} | {avg_single_len:>8.0f} | {avg_multi_len:>8.0f}")

    out = ROOT / "results" / "agent_v2_changes" / "06_multi_agent_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "single": single_results,
        "multi": multi_results,
        "summary": {
            "avg_single_elapsed": round(avg_single_t, 1),
            "avg_multi_elapsed": round(avg_multi_t, 1),
            "avg_single_messages": round(avg_single_msgs, 1),
            "avg_multi_messages": round(avg_multi_msgs, 1),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
