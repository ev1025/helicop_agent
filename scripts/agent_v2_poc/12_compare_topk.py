"""
#004 - vector_search_top_k 30 → 100 + reranker_top_k 7 → 5 비교.

이 비교는 Reranker 가 반드시 ON 이어야 의미가 있다.
- top_k=30 + rerank_top_k=7 (기존)
- top_k=100 + rerank_top_k=5 (제안: 더 넓게 후보 → 더 엄격하게 추리기)

전제:
  - #003-A 에서 결정된 RERANKER_SCORE_THRESHOLD 를 적용 (default 0.2)
  - 둘 다 reranker ON, threshold 동일

측정 metric:
  - 평균 결과 문서 수
  - 평균 elapsed (검색 + 리랭크)
  - 페이지 다양성 (unique pages)
  - top-K 의 평균 점수 (정밀도 proxy)
  - 두 설정 간 페이지 집합 Jaccard
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


QUERIES = [
    "베르누이 원리로 양력이 어떻게 발생하나요?",
    "헬리콥터 메인 로터의 피치 각도가 양력에 미치는 영향",
    "Vortex Ring State 란",
    "동적 롤오버",
    "헬리콥터 비행 원리 종합",
    "양력 4요소 설명",
    "오토자이로와 헬리콥터의 차이",
]

# #003-A 결과로 결정된 값. 환경변수로 override 가능.
THRESHOLD = float(os.environ.get("THRESHOLD", "0.2"))


def run_with_settings(label: str, top_k: int, rerank_top_k: int):
    from app.config import config
    from app.services.rag_service import rag_search_with_rerank

    print(f"\n[{label}] top_k={top_k}, rerank_top_k={rerank_top_k}, threshold={config.RERANKER_SCORE_THRESHOLD}")
    out = []
    for q in QUERIES:
        t0 = time.perf_counter()
        # rag_search_with_rerank 의 인자로 직접 전달
        results = rag_search_with_rerank(q, top_k=top_k, reranker_top_k=rerank_top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pages = [r.get("page") for r in results]
        scores = [round(r.get("score", 0.0), 4) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        out.append({
            "query": q,
            "n_docs": len(results),
            "pages": pages,
            "scores": scores,
            "avg_score": round(avg_score, 4),
            "elapsed_ms": round(elapsed_ms, 1),
        })
        print(f"  '{q[:30]:30s}'  n={len(results):2d}  avg_score={avg_score:.3f}  elapsed={elapsed_ms:5.0f}ms  pages={pages}")
    return out


def main():
    from app.config import config
    from app.core import models

    print("=" * 70)
    print(f"[1/4] 임베딩 + Chroma + Reranker 로드  (THRESHOLD={THRESHOLD})")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()

    saved_use = config.USE_RERANKER
    saved_model = config.RERANKER_MODEL
    saved_thr = config.RERANKER_SCORE_THRESHOLD
    config.USE_RERANKER = True
    # #003-B 결과: BAAI/bge-reranker-base 가 3배 빠르고 점수 분포 거의 동일.
    config.RERANKER_MODEL = "BAAI/bge-reranker-base"
    config.RERANKER_SCORE_THRESHOLD = THRESHOLD
    models.load_reranker_model()

    try:
        print()
        print("=" * 70)
        print("[2/4] 설정 A: 기존 (top_k=30, rerank_top_k=7)")
        print("=" * 70)
        a = run_with_settings("A", 30, 7)

        print()
        print("=" * 70)
        print("[3/4] 설정 B: 제안 (top_k=100, rerank_top_k=5)")
        print("=" * 70)
        b = run_with_settings("B", 100, 5)
    finally:
        config.USE_RERANKER = saved_use
        config.RERANKER_MODEL = saved_model
        config.RERANKER_SCORE_THRESHOLD = saved_thr
        models.reranker_model = None
        models.reranker_tokenizer = None

    # 비교
    print()
    print("=" * 70)
    print("[4/4] 비교")
    print("=" * 70)

    rows = []
    for ra, rb in zip(a, b):
        a_pages = set(p for p in ra["pages"] if isinstance(p, (int, float)))
        b_pages = set(p for p in rb["pages"] if isinstance(p, (int, float)))
        inter = a_pages & b_pages
        union = a_pages | b_pages
        jaccard = len(inter) / len(union) if union else 0.0
        rows.append({
            "query": ra["query"],
            "A_n": ra["n_docs"], "B_n": rb["n_docs"],
            "A_avg_score": ra["avg_score"], "B_avg_score": rb["avg_score"],
            "A_elapsed_ms": ra["elapsed_ms"], "B_elapsed_ms": rb["elapsed_ms"],
            "jaccard_pages": round(jaccard, 3),
        })

    n = len(rows)
    avg_a_n = sum(r["A_n"] for r in rows) / n
    avg_b_n = sum(r["B_n"] for r in rows) / n
    avg_a_score = sum(r["A_avg_score"] for r in rows if r["A_n"]) / max(1, sum(1 for r in rows if r["A_n"]))
    avg_b_score = sum(r["B_avg_score"] for r in rows if r["B_n"]) / max(1, sum(1 for r in rows if r["B_n"]))
    avg_a_ms = sum(r["A_elapsed_ms"] for r in rows) / n
    avg_b_ms = sum(r["B_elapsed_ms"] for r in rows) / n
    avg_jac = sum(r["jaccard_pages"] for r in rows) / n

    print()
    print(f"{'metric':30s} | {'A (30/7)':>10s} | {'B (100/5)':>10s} | delta")
    print("-" * 70)
    print(f"{'평균 결과 수':30s} | {avg_a_n:>10.2f} | {avg_b_n:>10.2f} | {avg_b_n-avg_a_n:+.2f}")
    print(f"{'평균 점수 (정밀도)':30s} | {avg_a_score:>10.3f} | {avg_b_score:>10.3f} | {avg_b_score-avg_a_score:+.3f}")
    print(f"{'평균 elapsed (ms)':30s} | {avg_a_ms:>10.1f} | {avg_b_ms:>10.1f} | {avg_b_ms-avg_a_ms:+.1f}")
    print(f"{'평균 Jaccard (페이지 일치)':30s} | {avg_jac:>10.3f} |            | -")

    out = ROOT / "results" / "agent_v2_changes" / "04_topk_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "threshold_used": THRESHOLD,
        "summary": {
            "avg_A_n": round(avg_a_n, 2), "avg_B_n": round(avg_b_n, 2),
            "avg_A_score": round(avg_a_score, 3), "avg_B_score": round(avg_b_score, 3),
            "avg_A_ms": round(avg_a_ms, 1), "avg_B_ms": round(avg_b_ms, 1),
            "avg_jaccard": round(avg_jac, 3),
        },
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
