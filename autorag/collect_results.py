"""
모든 benchmark_*/0/summary.csv 를 파싱해서 PROD_RESULT_FINAL.md 작성.

만들어주는 표:
1) 모델별 best 조합 표
   | 모델 | 조합(BM/Vector/Hybrid) | 토크나이저 | reranker | NDCG | MRR | sem_score | g_eval | faithfulness | latency(s/q) |
2) 전체 평가 중 최고순위 5위 (종합점수 = sem_score + g_eval + faithfulness 평균)
3) 노드별 상세 (BM25, vector, hybrid, reranker)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
AUTORAG_DIR = ROOT
OUT_PATH = AUTORAG_DIR / "PROD_RESULT_FINAL.md"


def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if pd.isna(v):
            return "-"
        return f"{v:.{digits}f}"
    return str(v)


def safe_to_model(safe: str) -> str:
    return safe.replace("benchmark_", "").replace("_", "/", 1)


def collect_trial(trial_dir: Path) -> dict[str, Any] | None:
    summary_path = trial_dir / "summary.csv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)

    safe_name = trial_dir.parent.name.replace("benchmark_", "")
    info: dict[str, Any] = {"safe_name": safe_name, "model": safe_to_model(trial_dir.parent.name)}

    # 노드별 상세 csv
    detail: dict[str, dict[str, Any]] = {}
    for node in ["lexical_retrieval", "semantic_retrieval", "hybrid_retrieval", "passage_reranker", "passage_compressor"]:
        node_csv = trial_dir / "retrieve_line" / node / "summary.csv"
        if not node_csv.exists():
            continue
        df = pd.read_csv(node_csv)
        if "is_best" in df.columns and df["is_best"].any():
            best = df[df["is_best"]].iloc[0]
        else:
            best = df.iloc[0]
        # 메트릭 컬럼 (prefix 처리)
        d: dict[str, Any] = {"module": best.get("module_name"), "exec_time": best.get("execution_time")}
        for col in df.columns:
            if any(col.endswith(suffix) or col == suffix for suffix in
                   ["retrieval_f1", "retrieval_recall", "retrieval_precision",
                    "retrieval_ndcg", "retrieval_mrr", "retrieval_map",
                    "retrieval_token_f1", "retrieval_token_recall", "retrieval_token_precision"]):
                short = col.split("retrieval_")[-1] if "retrieval_" in col else col
                d[short] = best.get(col)
        detail[node] = d

    info["detail"] = detail

    # generator 결과
    gen_csv = trial_dir / "generate_line" / "generator" / "summary.csv"
    gen: dict[str, Any] = {}
    if gen_csv.exists():
        gdf = pd.read_csv(gen_csv)
        if "is_best" in gdf.columns and gdf["is_best"].any():
            gbest = gdf[gdf["is_best"]].iloc[0]
        else:
            gbest = gdf.iloc[0]
        for k in ["bleu", "rouge", "sem_score", "g_eval", "bert_score", "deepeval_faithfulness",
                  "average_output_token", "execution_time"]:
            if k in gdf.columns:
                gen[k] = gbest.get(k)
    info["gen"] = gen
    return info


def main() -> None:
    trials = sorted(AUTORAG_DIR.glob("benchmark_*/0"))
    rows: list[dict[str, Any]] = []
    for t in trials:
        info = collect_trial(t)
        if info:
            rows.append(info)
    if not rows:
        print("benchmark_* 결과 없음", file=sys.stderr)
        sys.exit(1)

    md: list[str] = []
    md.append("# AutoRAG Multi-Model 평가 결과")
    md.append("")
    md.append(f"평가 모델 수: **{len(rows)}**")
    md.append("")
    md.append("## 1. 모델별 best 조합 + 메트릭 + 시간")
    md.append("")
    md.append("| 모델명 | 조합 | 토크나이저 | reranker | NDCG | MRR | retrieval_map | sem_score | g_eval | faithfulness | gen latency(s/q) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    ranked: list[tuple[float, dict[str, Any]]] = []

    for r in rows:
        d = r["detail"]
        g = r["gen"]
        # 어떤 retrieval mode 가 ndcg 1등인지
        mode_scores = {}
        for n, label in [("lexical_retrieval", "BM only"),
                         ("semantic_retrieval", "Vector only"),
                         ("hybrid_retrieval", "Hybrid")]:
            v = d.get(n, {}).get("ndcg")
            if isinstance(v, (int, float)) and not pd.isna(v):
                mode_scores[label] = v
        best_mode = max(mode_scores, key=mode_scores.get) if mode_scores else "-"

        rerank = d.get("passage_reranker", {})
        rerank_module = rerank.get("module", "-")
        rerank_ndcg = rerank.get("ndcg")
        rerank_mrr = rerank.get("mrr")
        rerank_map = rerank.get("map")

        gen_latency = g.get("execution_time")
        sem = g.get("sem_score")
        geval = g.get("g_eval")
        faith = g.get("deepeval_faithfulness")

        # 종합점수 (sem + g_eval + faithfulness 의 평균; None 인 경우 0)
        s_components = [v for v in [sem, geval, faith] if isinstance(v, (int, float)) and not pd.isna(v)]
        composite = (sum(s_components) / len(s_components)) if s_components else 0.0
        ranked.append((composite, r))

        md.append(
            f"| {r['model']} | {best_mode} | ko_kiwi | {rerank_module} | "
            f"{fmt(rerank_ndcg)} | {fmt(rerank_mrr)} | {fmt(rerank_map)} | "
            f"{fmt(sem)} | {fmt(geval)} | {fmt(faith)} | {fmt(gen_latency, 2)} |"
        )

    md.append("")
    md.append("## 2. 종합 점수 최고 순위 5위")
    md.append("")
    md.append("종합점수 = sem_score / g_eval / faithfulness 평균 (있는 것만)")
    md.append("")
    md.append("| 순위 | 모델 | 조합 | reranker | sem_score | g_eval | faithfulness | 종합 |")
    md.append("|---|---|---|---|---|---|---|---|")
    ranked.sort(key=lambda x: -x[0])
    for i, (score, r) in enumerate(ranked[:5], 1):
        d = r["detail"]
        g = r["gen"]
        mode_scores = {}
        for n, label in [("lexical_retrieval", "BM only"),
                         ("semantic_retrieval", "Vector only"),
                         ("hybrid_retrieval", "Hybrid")]:
            v = d.get(n, {}).get("ndcg")
            if isinstance(v, (int, float)) and not pd.isna(v):
                mode_scores[label] = v
        best_mode = max(mode_scores, key=mode_scores.get) if mode_scores else "-"
        rerank_module = d.get("passage_reranker", {}).get("module", "-")
        md.append(
            f"| {i} | {r['model']} | {best_mode} | {rerank_module} | "
            f"{fmt(g.get('sem_score'))} | {fmt(g.get('g_eval'))} | "
            f"{fmt(g.get('deepeval_faithfulness'))} | **{fmt(score)}** |"
        )

    md.append("")
    md.append("## 3. 노드별 상세 (모든 모델 평균)")
    md.append("")
    md.append("retrieval/reranker 점수는 LLM 무관하게 동일한 corpus/QA 위에서 측정됨.")
    md.append("")
    md.append("| 노드 | best 모듈 (1차 모델 기준) | NDCG | MRR | retrieval_map | exec_time(s) |")
    md.append("|---|---|---|---|---|---|")
    if rows:
        d = rows[0]["detail"]
        for node, label in [("lexical_retrieval", "lexical (BM25)"),
                            ("semantic_retrieval", "semantic (vector)"),
                            ("hybrid_retrieval", "hybrid"),
                            ("passage_reranker", "reranker")]:
            nd = d.get(node, {})
            md.append(
                f"| {label} | {nd.get('module', '-')} | "
                f"{fmt(nd.get('ndcg'))} | {fmt(nd.get('mrr'))} | "
                f"{fmt(nd.get('map'))} | {fmt(nd.get('exec_time'), 3)} |"
            )

    md.append("")
    md.append("## 4. Raw 결과 위치")
    md.append("")
    for r in rows:
        md.append(f"- `autorag/benchmark_{r['safe_name']}/0/`")
    md.append("")

    OUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"saved: {OUT_PATH}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
