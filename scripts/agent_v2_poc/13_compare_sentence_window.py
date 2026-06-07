"""
#005 - Sentence-Window 인접 청크 자동 회수 비교 (window=0 vs 1 vs 2).

매칭된 Parent 의 ±N 인접 Parent 를 자동 추가 회수.
같은 주제가 여러 Parent 로 쪼개졌을 때 인접 누락을 방지.

측정 metric:
  - 평균 결과 문서 수 (확장 효과)
  - 페이지 다양성 (unique pages)
  - 같은 페이지에 인접 청크 추가된 케이스 수
  - 컨텍스트 길이 증가량
  - 평균 elapsed (인접 회수는 dict 조회만이라 거의 무비용)
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

WINDOWS = [0, 1, 2]


def main():
    from app.core import models
    from app.services.rag_service import rag_search_with_rerank

    print("=" * 70)
    print("[1/3] 임베딩 + Chroma 로드")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()

    print()
    print("=" * 70)
    print(f"[2/3] {len(WINDOWS)} window × {len(QUERIES)} 질의 측정")
    print("=" * 70)

    rows = []
    for w in WINDOWS:
        print(f"\nwindow = {w}")
        for q in QUERIES:
            t0 = time.perf_counter()
            results = rag_search_with_rerank(q, sentence_window=w)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            pages = [r.get("page") for r in results]
            chunk_ids = [r.get("metadata", {}).get("parent_chunk_id") for r in results]
            adjacent_flags = [r.get("_adjacent", False) for r in results]
            n_adjacent = sum(1 for f in adjacent_flags if f)
            total_len = sum(len(r.get("content", "")) for r in results)

            rows.append({
                "query": q, "window": w,
                "n_docs": len(results),
                "n_adjacent": n_adjacent,
                "total_content_len": total_len,
                "pages": pages,
                "chunk_ids": chunk_ids,
                "elapsed_ms": round(elapsed_ms, 1),
            })
            print(
                f"  '{q[:30]:30s}'  n={len(results):2d}  adj={n_adjacent:2d}  "
                f"len={total_len:5d}  ms={elapsed_ms:5.0f}  pages={pages}"
            )

    # 종합
    def agg(window):
        sub = [r for r in rows if r["window"] == window]
        n = len(sub)
        return {
            "window": window,
            "avg_n_docs": round(sum(r["n_docs"] for r in sub) / n, 2),
            "avg_n_adjacent": round(sum(r["n_adjacent"] for r in sub) / n, 2),
            "avg_total_len": round(sum(r["total_content_len"] for r in sub) / n, 0),
            "avg_elapsed_ms": round(sum(r["elapsed_ms"] for r in sub) / n, 1),
        }

    summary = [agg(w) for w in WINDOWS]

    print()
    print("=" * 70)
    print("[3/3] 종합")
    print("=" * 70)
    print(f"{'window':>6s} | {'avg_n':>6s} | {'avg_adj':>7s} | {'avg_len':>7s} | {'avg_ms':>6s}")
    print("-" * 70)
    for s in summary:
        print(
            f"{s['window']:>6d} | {s['avg_n_docs']:>6.2f} | {s['avg_n_adjacent']:>7.2f} | "
            f"{s['avg_total_len']:>7.0f} | {s['avg_elapsed_ms']:>6.1f}"
        )
    print()
    print("→ window 키울수록 컨텍스트 풍부 ↑, 길이 증가, 무관 청크 포함 위험 ↑")
    print("→ MAX_CONTEXT_LENGTH 한도 안에서 멈추므로 폭주는 없음")

    out = ROOT / "results" / "agent_v2_changes" / "05_sentence_window_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary, "rows": rows
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
