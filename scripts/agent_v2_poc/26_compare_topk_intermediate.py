"""
#004-A - top_k 30 / 60 / 100 비교 (절충값 탐색).

#004 결과:
  A (30/7) : 842ms,  avg_score 0.216
  B (100/5): 8586ms, avg_score 0.431  (정밀도 2× but 10× 느림)

→ 60 정도면 정밀도와 속도 균형이 어떨지 확인.

사용:
  reranker = BAAI/bge-reranker-base (#003-B 결과로 가장 빠름)
  threshold = 0.0 (필터 없이 점수 분포만 비교)
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


QUERIES = [
    "베르누이 원리로 양력이 어떻게 발생하나요?",
    "헬리콥터 메인 로터의 피치 각도가 양력에 미치는 영향",
    "Vortex Ring State 란",
    "동적 롤오버",
    "헬리콥터 비행 원리 종합",
    "양력 4요소 설명",
    "오토자이로와 헬리콥터의 차이",
]

# (top_k, rerank_top_k) 매트릭스
SETTINGS = [
    ("A", 30, 7),
    ("M", 60, 6),  # 절충값
    ("B", 100, 5),
]


def main():
    from app.config import config
    from app.core import models
    from app.services.rag_service import rag_search_with_rerank

    print("=" * 70)
    print("[1/3] 임베딩 + Chroma + Reranker(BAAI base) 로드")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()

    saved_use = config.USE_RERANKER
    saved_model = config.RERANKER_MODEL
    saved_thr = config.RERANKER_SCORE_THRESHOLD
    config.USE_RERANKER = True
    config.RERANKER_MODEL = "BAAI/bge-reranker-base"
    config.RERANKER_SCORE_THRESHOLD = 0.0
    models.load_reranker_model()

    by_setting = {}
    try:
        for label, tk, rtk in SETTINGS:
            print()
            print("=" * 70)
            print(f"[{label}] top_k={tk}, rerank_top_k={rtk}")
            print("=" * 70)
            rows = []
            for q in QUERIES:
                t0 = time.perf_counter()
                results = rag_search_with_rerank(q, top_k=tk, reranker_top_k=rtk)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                pages = [r.get("page") for r in results]
                scores = [round(r.get("score", 0.0), 4) for r in results]
                avg_score = sum(scores) / len(scores) if scores else 0
                rows.append({
                    "query": q, "n": len(results),
                    "pages": pages, "avg_score": round(avg_score, 4),
                    "elapsed_ms": round(elapsed_ms, 1),
                })
                print(f"  '{q[:30]:30s}'  n={len(results)}  avg={avg_score:.3f}  {elapsed_ms:5.0f}ms  pages={pages}")

            n = len(rows)
            avg_n = sum(r["n"] for r in rows) / n
            avg_score = sum(r["avg_score"] for r in rows) / n
            avg_ms = sum(r["elapsed_ms"] for r in rows) / n
            by_setting[label] = {
                "top_k": tk, "rerank_top_k": rtk,
                "avg_n": round(avg_n, 2),
                "avg_score": round(avg_score, 4),
                "avg_elapsed_ms": round(avg_ms, 1),
                "rows": rows,
            }
    finally:
        config.USE_RERANKER = saved_use
        config.RERANKER_MODEL = saved_model
        config.RERANKER_SCORE_THRESHOLD = saved_thr
        models.reranker_model = None
        models.reranker_tokenizer = None

    print()
    print("=" * 70)
    print("[종합]")
    print("=" * 70)
    print(f"\n{'label':5s} | {'top_k':>5s} | {'rerank':>6s} | {'avg_n':>5s} | {'avg_score':>9s} | {'avg_ms':>7s}")
    print("-" * 60)
    for label, data in by_setting.items():
        print(f"{label:5s} | {data['top_k']:>5d} | {data['rerank_top_k']:>6d} | "
              f"{data['avg_n']:>5.2f} | {data['avg_score']:>9.4f} | {data['avg_elapsed_ms']:>7.1f}")

    out = ROOT / "results" / "agent_v2_changes" / "04A_topk_intermediate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(by_setting, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
