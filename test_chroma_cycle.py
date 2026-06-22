#!/usr/bin/env python3
"""
ChromaDB 생성/삭제 반복 테스트
Phase 4와 동일한 흐름으로 12가지 chunk 설정을 순차적으로 테스트
"""

import sys
from pathlib import Path
import shutil
import time

# Project root
project_root = Path("/workspace/jupyter_notebooks/surion_llm_isaacsim")
sys.path.insert(0, str(project_root))

from app.core.embeddings import get_embeddings
from app.core.vector_store import create_parent_document_vectordb
import json

# Config 로드
with open(project_root / "config" / "models.json") as f:
    models_config = json.load(f)

# 테스트 설정들 (Phase 4와 동일)
TEST_CONFIGS = [
    {'name': 'child300_co30_parent1500_po150', 'child_size': 300, 'child_overlap': 30, 'parent_size': 1500, 'parent_overlap': 150},
    {'name': 'child300_co30_parent2500_po250', 'child_size': 300, 'child_overlap': 30, 'parent_size': 2500, 'parent_overlap': 250},
    {'name': 'child300_co100_parent1500_po150', 'child_size': 300, 'child_overlap': 100, 'parent_size': 1500, 'parent_overlap': 150},
    {'name': 'child300_co100_parent2500_po250', 'child_size': 300, 'child_overlap': 100, 'parent_size': 2500, 'parent_overlap': 250},
    {'name': 'child400_co30_parent1500_po150', 'child_size': 400, 'child_overlap': 30, 'parent_size': 1500, 'parent_overlap': 150},
    {'name': 'child400_co30_parent2500_po250', 'child_size': 400, 'child_overlap': 30, 'parent_size': 2500, 'parent_overlap': 250},
    {'name': 'child400_co100_parent1500_po150', 'child_size': 400, 'child_overlap': 100, 'parent_size': 1500, 'parent_overlap': 150},
    {'name': 'child400_co100_parent2500_po250', 'child_size': 400, 'child_overlap': 100, 'parent_size': 2500, 'parent_overlap': 250},
    {'name': 'child500_co30_parent1500_po150', 'child_size': 500, 'child_overlap': 30, 'parent_size': 1500, 'parent_overlap': 150},
    {'name': 'child500_co30_parent2500_po250', 'child_size': 500, 'child_overlap': 30, 'parent_size': 2500, 'parent_overlap': 250},
    {'name': 'child500_co100_parent1500_po150', 'child_size': 500, 'child_overlap': 100, 'parent_size': 1500, 'parent_overlap': 150},
    {'name': 'child500_co100_parent2500_po250', 'child_size': 500, 'child_overlap': 100, 'parent_size': 2500, 'parent_overlap': 250},
]

def delete_chroma_db():
    """ChromaDB 완전 삭제"""
    db_path = project_root / "chroma_db_new"
    if db_path.exists():
        print(f"  🗑️  ChromaDB 삭제 중: {db_path}")
        shutil.rmtree(db_path)
        time.sleep(0.5)  # 파일 시스템 sync
        print("     ✅ 삭제 완료")
    else:
        print("     ℹ️  ChromaDB 없음 (이미 삭제됨)")

def create_and_test_chroma(cfg):
    """특정 chunk 설정으로 ChromaDB 생성 및 테스트"""
    print(f"\n{'='*80}")
    print(f"테스트: {cfg['name']}")
    print(f"  Child: {cfg['child_size']} (overlap={cfg['child_overlap']})")
    print(f"  Parent: {cfg['parent_size']} (overlap={cfg['parent_overlap']})")
    print(f"{'='*80}")

    start = time.time()

    try:
        # 1. Embedding 로드
        print("\n1️⃣  Embedding 모델 로딩...")
        embedding_model = models_config["embedding_model"]
        embedding_device = models_config.get("embedding_device", "cuda:0")

        embedding_function = get_embeddings(embedding_model, embedding_device)
        print("   ✅ Embedding 로딩 완료")

        # 2. ChromaDB 생성
        print("\n2️⃣  ChromaDB 생성 중...")
        print(f"   Child chunk: {cfg['child_size']}, Parent chunk: {cfg['parent_size']}")

        pdf_path = "/workspace/jupyter_notebooks/surion_llm_isaacsim/조종사표준교재(비행이론_헬리콥터).pdf"
        db_path = str(project_root / "chroma_db_new")

        child_collection, parent_store = create_parent_document_vectordb(
            embedding_function=embedding_function,
            model_name=embedding_model,
            vector_db_path=db_path,
            collection_name="new_manual",
            pdf_path=pdf_path,
            child_chunk_size=cfg['child_size'],
            child_chunk_overlap=cfg['child_overlap'],
            parent_chunk_size=cfg['parent_size'],
            parent_chunk_overlap=cfg['parent_overlap'],
            text_cleaning_mode="v1"
        )

        print(f"   ✅ ChromaDB 생성 완료!")
        print(f"      Parent docs: {len(parent_store)}개")

        # 3. 간단한 쿼리 테스트
        print("\n3️⃣  쿼리 테스트...")
        results = child_collection.similarity_search("헬리콥터", k=3)
        print(f"   ✅ 쿼리 성공! 결과: {len(results)}개")

        # 4. 통계
        duration = time.time() - start
        print(f"\n✅ 성공! 소요 시간: {duration:.1f}초")

        return True

    except Exception as e:
        duration = time.time() - start
        print(f"\n❌ 실패! 에러: {e}")
        print(f"   소요 시간: {duration:.1f}초")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("ChromaDB 생성/삭제 반복 테스트 (Phase 4 시뮬레이션)")
    print("="*80)
    print(f"총 {len(TEST_CONFIGS)}개 설정 테스트")
    print()

    # 초기 삭제
    print("🔄 초기 ChromaDB 삭제...")
    delete_chroma_db()

    success_count = 0
    fail_count = 0

    for i, cfg in enumerate(TEST_CONFIGS, 1):
        print(f"\n\n{'#'*80}")
        print(f"진행: {i}/{len(TEST_CONFIGS)} ({i/len(TEST_CONFIGS)*100:.1f}%)")
        print(f"{'#'*80}")

        # 생성 및 테스트
        success = create_and_test_chroma(cfg)

        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"\n⚠️  {cfg['name']} 실패! 계속 진행...")

        # 삭제 (다음 테스트 준비)
        print(f"\n4️⃣  ChromaDB 삭제 (다음 테스트 준비)...")
        delete_chroma_db()

        # 진행 상황
        print(f"\n📊 현재까지: 성공 {success_count}, 실패 {fail_count}")
        remaining = len(TEST_CONFIGS) - i
        print(f"   남은 테스트: {remaining}개")

    # 최종 결과
    print("\n\n" + "="*80)
    print("최종 결과")
    print("="*80)
    print(f"✅ 성공: {success_count}/{len(TEST_CONFIGS)}")
    print(f"❌ 실패: {fail_count}/{len(TEST_CONFIGS)}")

    if fail_count == 0:
        print("\n🎉 모든 테스트 통과! Phase 4 진행 가능!")
        return 0
    else:
        print(f"\n⚠️  {fail_count}개 테스트 실패. 로그를 확인하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
