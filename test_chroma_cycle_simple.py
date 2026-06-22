#!/usr/bin/env python3
"""
ChromaDB 생성/삭제 반복 테스트 (간단 버전)
Phase 4와 동일한 흐름으로 12가지 chunk 설정을 순차적으로 테스트
"""

import sys
import os
from pathlib import Path
import shutil
import time
import json

# ChromaDB 환경변수 설정 (SQLite readonly 문제 해결 시도)
os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
os.environ['SQLITE_TMPDIR'] = '/tmp'

# Project root
project_root = Path("/workspace/jupyter_notebooks/surion_llm_isaacsim")
sys.path.insert(0, str(project_root))

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

def update_rag_config(cfg):
    """RAG config 파일 업데이트"""
    rag_config_path = project_root / "config" / "rag.json"
    with open(rag_config_path, 'r') as f:
        rag_config = json.load(f)

    rag_config['child_chunk_size'] = cfg['child_size']
    rag_config['child_chunk_overlap'] = cfg['child_overlap']
    rag_config['parent_chunk_size'] = cfg['parent_size']
    rag_config['parent_chunk_overlap'] = cfg['parent_overlap']

    with open(rag_config_path, 'w') as f:
        json.dump(rag_config, f, ensure_ascii=False, indent=2)

def delete_chroma_db():
    """ChromaDB 완전 삭제"""
    db_path = project_root / "chroma_db_new"
    if db_path.exists():
        print(f"  🗑️  ChromaDB 삭제 중...")
        shutil.rmtree(db_path)
        time.sleep(0.5)  # 파일 시스템 sync
        print("     ✅ 삭제 완료")
    else:
        print("     ℹ️  ChromaDB 없음")

def create_and_test_chroma(cfg):
    """특정 chunk 설정으로 ChromaDB 생성 및 테스트"""
    print(f"\n{'='*80}")
    print(f"테스트: {cfg['name']}")
    print(f"  Child: {cfg['child_size']} (overlap={cfg['child_overlap']})")
    print(f"  Parent: {cfg['parent_size']} (overlap={cfg['parent_overlap']})")
    print(f"{'='*80}")

    start = time.time()

    try:
        # 1. RAG config 업데이트
        print("\n1️⃣  RAG config 업데이트...")
        update_rag_config(cfg)
        print("   ✅ Config 업데이트 완료")

        # 2. Config reload
        print("\n2️⃣  Config reload...")
        from app.config import config
        config._load_configs()
        print("   ✅ Config reload 완료")

        # 3. models 모듈 임포트 (embedding 포함)
        print("\n3️⃣  Models 모듈 로딩...")
        from app.core import models

        # 기존 모델 언로드
        models._vector_db = None
        models._parent_store = None

        # Vector DB만 로드 (embedding 자동 로드됨)
        print("\n4️⃣  Vector DB 생성 중...")
        models.load_vector_db()
        print("   ✅ Vector DB 생성 완료!")

        # 5. 간단한 테스트
        print("\n5️⃣  쿼리 테스트...")
        collection, parent_store = models.get_vector_db()
        results = collection.similarity_search("헬리콥터", k=3)
        print(f"   ✅ 쿼리 성공! 결과: {len(results)}개, Parent: {len(parent_store)}개")

        # 6. 통계
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
    failed_configs = []

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
            failed_configs.append(cfg['name'])
            print(f"\n⚠️  {cfg['name']} 실패! 계속 진행...")

        # 삭제 (다음 테스트 준비)
        print(f"\n6️⃣  ChromaDB 삭제 (다음 테스트 준비)...")
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

    if failed_configs:
        print(f"\n실패한 설정:")
        for name in failed_configs:
            print(f"  - {name}")

    if fail_count == 0:
        print("\n🎉 모든 테스트 통과! Phase 4 진행 가능!")
        return 0
    else:
        print(f"\n⚠️  {fail_count}개 테스트 실패. 로그를 확인하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
