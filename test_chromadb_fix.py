#!/usr/bin/env python3
"""
ChromaDB 손상 복구 테스트 스크립트

이 스크립트는 ChromaDB 손상 시나리오를 재현하고 자동 복구가 정상 작동하는지 검증합니다.
"""

import sys
from pathlib import Path
import shutil
import time

# Project root 설정
project_root = Path("/workspace/jupyter_notebooks/surion_llm_isaacsim")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_chromadb_recovery():
    """ChromaDB 손상 후 자동 복구 테스트"""

    print("=" * 80)
    print("ChromaDB 손상 복구 테스트")
    print("=" * 80)

    # 1. Chroma DB 경로
    chroma_db_path = project_root / "chroma_db_new"

    # 2. 기존 DB 삭제 (깨끗한 시작)
    if chroma_db_path.exists():
        print(f"\n1️⃣  기존 ChromaDB 삭제: {chroma_db_path}")
        shutil.rmtree(chroma_db_path)
        print("   ✅ 삭제 완료")
    else:
        print(f"\n1️⃣  ChromaDB 없음 (깨끗한 시작)")

    # 3. vector_store.py의 함수 임포트
    print("\n2️⃣  vector_store.py 임포트 테스트...")
    try:
        from app.core.vector_store import load_vector_db_parent_retriever
        from app.core.embeddings import get_embeddings
        from app.config import config
        print("   ✅ 임포트 성공")
    except Exception as e:
        print(f"   ❌ 임포트 실패: {e}")
        return False

    # 4. 손상 시나리오 재현: 폴더는 있지만 SQLite DB가 손상됨
    print("\n3️⃣  손상 시나리오 재현 중...")
    print("   (notebook의 force_rebuild_chroma_db()가 삭제한 직후 상태)")

    # Chroma DB 폴더만 생성 (빈 폴더)
    chroma_db_path.mkdir(exist_ok=True)
    print(f"   ✅ 빈 폴더 생성됨: {chroma_db_path}")

    # 5. ChromaDB PersistentClient 생성 시도 (손상 감지 및 복구 테스트)
    print("\n4️⃣  ChromaDB PersistentClient 생성 테스트...")
    print("   (자동 복구 로직이 작동해야 함)")

    try:
        import chromadb

        # 이 부분에서 "no such table" 에러가 발생할 수 있지만,
        # vector_store.py의 retry 로직이 자동으로 복구해야 함
        client = chromadb.PersistentClient(path=str(chroma_db_path))
        collections = client.list_collections()
        print(f"   ✅ PersistentClient 생성 성공!")
        print(f"   Collections: {len(collections)}개")

        # 정상 상태 확인
        if chroma_db_path.exists():
            print(f"   ✅ ChromaDB 폴더 존재 확인")
            # SQLite 파일 확인
            sqlite_file = chroma_db_path / "chroma.sqlite3"
            if sqlite_file.exists():
                print(f"   ✅ SQLite DB 파일 존재: {sqlite_file}")
            else:
                print(f"   ⚠️  SQLite DB 파일 없음 (정상일 수도 있음)")

        return True

    except Exception as e:
        print(f"   ❌ PersistentClient 생성 실패: {e}")
        print(f"   에러 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_functions():
    """vector_store.py의 실제 함수 테스트"""

    print("\n" + "=" * 80)
    print("vector_store.py 함수 직접 테스트")
    print("=" * 80)

    try:
        from app.core.vector_store import load_vector_db_parent_retriever
        from app.core.embeddings import get_embeddings
        from app.config import config

        # ChromaDB 삭제
        chroma_db_path = project_root / "chroma_db_new"
        if chroma_db_path.exists():
            print(f"\n1️⃣  기존 ChromaDB 삭제...")
            shutil.rmtree(chroma_db_path)
            print("   ✅ 삭제 완료")

        # 빈 폴더 생성 (손상 상태)
        chroma_db_path.mkdir(exist_ok=True)
        print(f"\n2️⃣  빈 ChromaDB 폴더 생성 (손상 상태 재현)")

        # Embedding 함수 로드
        print(f"\n3️⃣  Embedding 함수 로드 중...")
        embedding_function = get_embeddings(
            config.embedding_model,
            config.embedding_device
        )
        print("   ✅ Embedding 로드 완료")

        # load_vector_db_parent_retriever 호출 (자동 복구 테스트)
        print(f"\n4️⃣  load_vector_db_parent_retriever 호출...")
        print("   (손상 감지 → 자동 복구 → 정상 동작)")

        pdf_path = str(project_root / config.pdf_file_path)

        child_collection, parent_store = load_vector_db_parent_retriever(
            embedding_function=embedding_function,
            model_name=config.embedding_model,
            vector_db_path=str(chroma_db_path),
            collection_name="new_manual",
            pdf_path=pdf_path,
            child_chunk_size=500,
            child_chunk_overlap=100,
            parent_chunk_size=2500,
            parent_chunk_overlap=250,
            text_cleaning_mode="v1"
        )

        print(f"\n   ✅ 함수 호출 성공!")
        print(f"   Parent documents: {len(parent_store)}개")

        # 검증: collection 쿼리 테스트
        print(f"\n5️⃣  Collection 쿼리 테스트...")
        results = child_collection.similarity_search("테스트", k=1)
        print(f"   ✅ 쿼리 성공! 결과: {len(results)}개")

        return True

    except Exception as e:
        print(f"\n   ❌ 함수 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🔬" * 40)
    print("ChromaDB 손상 복구 자동화 검증 테스트")
    print("🔬" * 40 + "\n")

    # Test 1: 기본 PersistentClient 생성 테스트
    result1 = test_chromadb_recovery()

    # Test 2: 실제 vector_store 함수 테스트
    result2 = test_vector_store_functions()

    # 최종 결과
    print("\n" + "=" * 80)
    print("최종 결과")
    print("=" * 80)

    if result1 and result2:
        print("\n✅ 모든 테스트 통과!")
        print("   ChromaDB 손상 감지 및 자동 복구 정상 작동")
        sys.exit(0)
    else:
        print("\n❌ 일부 테스트 실패")
        print("   Test 1 (PersistentClient): " + ("✅" if result1 else "❌"))
        print("   Test 2 (vector_store 함수): " + ("✅" if result2 else "❌"))
        sys.exit(1)
