"""
실제 PDF 를 chroma_db 로 인덱싱.

agent_v2 통합 테스트 전에 한 번만 돌리면 됨.

수행:
  1) sentence-transformers/all-MiniLM-L6-v2 임베딩 모델 다운로드/로드
  2) 조종사표준교재 PDF 를 PyPDFLoader 로 적재
  3) Parent-Child 청킹 (config 의 child=300, parent=1500)
  4) Child 청크들을 Chroma 벡터 DB 에 인덱싱
  5) parent_store 를 디스크에 직렬화

실행:
    .venv/Scripts/python.exe scripts/agent_v2_poc/05_index_pdf_to_chroma.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


def main():
    from app.config import config
    from app.core import models

    print("=" * 60)
    print("PDF → Chroma 인덱싱")
    print("=" * 60)
    print(f"  PDF              : {config.PDF_PATH}")
    print(f"  EMBEDDING_MODEL  : {config.EMBEDDING_MODEL}")
    print(f"  VECTOR_DB_PATH   : {config.VECTOR_DB_PATH}")
    print(f"  CHILD chunk      : {getattr(config, 'CHILD_CHUNK_SIZE', '?')} / overlap {getattr(config, 'CHILD_CHUNK_OVERLAP', '?')}")
    print(f"  PARENT chunk     : {getattr(config, 'PARENT_CHUNK_SIZE', '?')} / overlap {getattr(config, 'PARENT_CHUNK_OVERLAP', '?')}")
    print(f"  PARENT 모드      : {getattr(config, 'PARENT_DOCUMENT_MODE', False)}")

    print()
    print("[1/2] 임베딩 모델 로드 (필요 시 ~80MB 다운로드)")
    models.load_embedding_model()
    print(f"      embedding_model 타입: {type(models.embedding_model).__name__}")

    print()
    print("[2/2] PDF 적재 + 청킹 + 벡터 인덱싱")
    models.load_vector_db()
    print(f"      collection: {models.collection}")
    print(f"      parent_store size: {len(models.parent_store) if models.parent_store else 0}")

    print()
    print("=" * 60)
    print("완료. 이제 agent_v2 가 실제 RAG 검색 가능.")
    print("=" * 60)


if __name__ == "__main__":
    main()
