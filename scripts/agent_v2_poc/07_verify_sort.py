"""
RAG 검색 결과 본문 순서 정렬 검증 (LLM 안 씀, GPU 점유 영향 X).

기존: 유사도 점수 순 → 본문 흐름 깨짐
수정: 페이지 → parent_chunk_id 순 정렬 (rag_service._sort_by_document_order)

확인:
  - 정렬 후 페이지 번호가 오름차순인가
  - 같은 페이지 내에서는 parent_chunk_id 오름차순인가
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def main():
    from app.core import models
    from app.services.rag_service import rag_search_with_rerank

    print("[1/3] 임베딩 + Chroma 로드")
    models.load_embedding_model()
    models.load_vector_db()

    print("[2/3] 여러 질의로 검색해서 정렬 결과 확인")
    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "헬리콥터 메인 로터의 피치 각도가 양력에 미치는 영향",
        "Vortex Ring State 란",
        "동적 롤오버",
        "헬리콥터 비행 원리 종합",
    ]

    print()
    print("=" * 70)
    print("정렬 결과 (페이지 → parent_chunk_id 순서로 나와야 함)")
    print("=" * 70)
    all_pass = True
    for q in queries:
        results = rag_search_with_rerank(q)
        pages = []
        chunk_ids = []
        for r in results:
            pages.append(r.get('page'))
            chunk_ids.append(r.get('metadata', {}).get('parent_chunk_id'))

        # 정렬 검증: 'Unknown' 제외하고 numeric만 비교
        numeric_pages = [(p, c) for p, c in zip(pages, chunk_ids) if isinstance(p, (int, float))]
        is_sorted = all(numeric_pages[i] <= numeric_pages[i+1] for i in range(len(numeric_pages)-1))

        marker = "✅" if is_sorted else "❌"
        if not is_sorted:
            all_pass = False
        print(f"\n{marker} '{q[:40]}...'")
        print(f"   pages       : {pages}")
        print(f"   chunk_ids   : {chunk_ids}")

    print()
    print("=" * 70)
    if all_pass:
        print("✅ 모든 질의에서 본문 순서로 정렬됨")
    else:
        print("❌ 일부 질의에서 정렬 실패")
    print("=" * 70)


if __name__ == "__main__":
    main()
