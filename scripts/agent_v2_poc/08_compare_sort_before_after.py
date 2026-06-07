"""
정렬 변경 (commit 0d83945) 전/후 비교.

목적: '_sort_by_document_order' 적용 효과를 같은 검색 결과로 정량 비교.

비교 metric:
  - 페이지 순서 단조 증가율 (sorted: 1.0, 본문 순서)
  - 인접 청크 거리 평균 (sorted 가 작으면 본문 응집도 ↑)
  - 청크 ID 의 spearman correlation (입력 순서 vs 본문 순서)

LLM 답변 품질 비교는 별도 (LLM 시간 오래 걸림). 여기는 RAG 단계의 정렬 효과만 측정.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # 노이즈 줄임


def _is_monotonic_increasing(seq):
    nums = [x for x in seq if isinstance(x, (int, float))]
    if len(nums) < 2:
        return True
    return all(nums[i] <= nums[i+1] for i in range(len(nums)-1))


def _adjacent_distance_mean(seq):
    """page 번호 인접 차이 평균 (작을수록 본문 응집도 ↑)."""
    nums = [x for x in seq if isinstance(x, (int, float))]
    if len(nums) < 2:
        return 0.0
    diffs = [abs(nums[i+1] - nums[i]) for i in range(len(nums)-1)]
    return sum(diffs) / len(diffs)


def _spearman(a, b):
    """간이 Spearman rank correlation (페이지 순서 일치도)."""
    n = len(a)
    if n < 2:
        return 1.0
    rank_a = {v: i for i, v in enumerate(sorted(a))}
    rank_b = {v: i for i, v in enumerate(sorted(b))}
    d2 = sum((rank_a[a[i]] - rank_b[b[i]]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n*n - 1))


def main():
    from app.core import models
    from app.services.rag_service import (
        rag_search_with_rerank,
        _sort_by_document_order,
    )

    print("=" * 70)
    print("[1/3] 임베딩 + Chroma 로드")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()

    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "헬리콥터 메인 로터의 피치 각도가 양력에 미치는 영향",
        "Vortex Ring State 란",
        "동적 롤오버",
        "헬리콥터 비행 원리 종합",
        "양력 4요소 설명",
        "오토자이로와 헬리콥터의 차이",
    ]

    print()
    print("=" * 70)
    print("[2/3] 정렬 전/후 비교")
    print("=" * 70)

    # 정렬 우회를 위해 _sort_by_document_order 를 일시적으로 항등 함수로 monkey-patch.
    # 이렇게 받은 결과가 진짜 "before" (Child 유사도 순으로 만난 Parent 순).
    import app.services.rag_service as rs
    _orig_sort = rs._sort_by_document_order
    rs._sort_by_document_order = lambda x: x  # type: ignore

    raw_results_per_query = {}
    for q in queries:
        raw_results_per_query[q] = rag_search_with_rerank(q)

    # 정렬 복구
    rs._sort_by_document_order = _orig_sort

    rows = []
    for q in queries:
        before_result = raw_results_per_query[q]
        after_result = _orig_sort(list(before_result))

        before_pages = [r.get('page') for r in before_result]
        after_pages = [r.get('page') for r in after_result]

        before_chunk_ids = [r.get('metadata', {}).get('parent_chunk_id') for r in before_result]
        after_chunk_ids = [r.get('metadata', {}).get('parent_chunk_id') for r in after_result]

        # metric 계산
        before_monotonic = _is_monotonic_increasing(before_pages)
        after_monotonic = _is_monotonic_increasing(after_pages)
        before_adj_dist = _adjacent_distance_mean(before_pages)
        after_adj_dist = _adjacent_distance_mean(after_pages)

        rows.append({
            "query": q,
            "before_pages": before_pages,
            "after_pages": after_pages,
            "before_monotonic": before_monotonic,
            "after_monotonic": after_monotonic,
            "before_adj_dist": round(before_adj_dist, 1),
            "after_adj_dist": round(after_adj_dist, 1),
        })

        print()
        print(f"질의: {q}")
        print(f"  before(점수 순): pages={before_pages} | monotonic={before_monotonic} | adj_dist={before_adj_dist:.1f}")
        print(f"  after (본문 순): pages={after_pages} | monotonic={after_monotonic} | adj_dist={after_adj_dist:.1f}")

    print()
    print("=" * 70)
    print("[3/3] 종합")
    print("=" * 70)

    n = len(rows)
    before_mono_count = sum(1 for r in rows if r["before_monotonic"])
    after_mono_count = sum(1 for r in rows if r["after_monotonic"])
    before_avg_adj = sum(r["before_adj_dist"] for r in rows) / n
    after_avg_adj = sum(r["after_adj_dist"] for r in rows) / n

    print(f"단조 증가 비율           : before {before_mono_count}/{n}  →  after {after_mono_count}/{n}")
    print(f"인접 페이지 거리 평균    : before {before_avg_adj:.1f}    →  after {after_avg_adj:.1f}")
    print()
    print("→ after 가 본문 순서로 항상 정렬되어 LLM 이 흐름을 그대로 따라갈 수 있음.")
    delta = before_avg_adj - after_avg_adj
    pct = (delta / before_avg_adj * 100) if before_avg_adj else 0
    print(f"→ 인접 페이지 평균 거리 {delta:.1f}p 감소 ({pct:.0f}% 응집도 향상).")

    # 결과 저장
    out = ROOT / "results" / "agent_v2_changes" / "01_sort_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": {
            "n_queries": n,
            "before_monotonic_count": before_mono_count,
            "after_monotonic_count": after_mono_count,
            "before_avg_adj_dist": round(before_avg_adj, 2),
            "after_avg_adj_dist": round(after_avg_adj, 2),
        },
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
