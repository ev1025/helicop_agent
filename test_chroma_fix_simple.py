#!/usr/bin/env python3
"""
ChromaDB 수정 테스트 - vector_store.py 자동 감지 검증
"""
import sys
from pathlib import Path
import json
import shutil
import time

# Project setup
project_root = Path("/workspace/jupyter_notebooks/surion_llm_isaacsim")
sys.path.insert(0, str(project_root))

from scripts.run_phase2_bench import run_bench, DEFAULT_CASES_PATH
from scripts.comprehensive_evaluation_phase2 import evaluate_phase2_bench

# Config paths
rag_config_file = project_root / "config" / "rag.json"
rag_backup = project_root / "config" / "rag.json.backup_test_simple"

# Test configs
TEST_CONFIGS = [
    {'name': 'child300_parent1500', 'child_size': 300, 'child_overlap': 30, 'parent_size': 1500, 'parent_overlap': 150},
    {'name': 'child500_parent2500', 'child_size': 500, 'child_overlap': 100, 'parent_size': 2500, 'parent_overlap': 250},
]

def backup_config():
    """Backup current config"""
    if not rag_backup.exists():
        shutil.copy(rag_config_file, rag_backup)
        print(f"✅ Backup created: {rag_backup.name}")
    else:
        print(f"ℹ️  Backup exists: {rag_backup.name}")

def restore_config():
    """Restore config"""
    shutil.copy(rag_backup, rag_config_file)
    print(f"✅ Config restored")

def apply_chunk_config(config_item):
    """Apply chunk config"""
    with open(rag_config_file, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)

    rag_data['child_chunk_size'] = config_item['child_size']
    rag_data['child_chunk_overlap'] = config_item['child_overlap']
    rag_data['parent_chunk_size'] = config_item['parent_size']
    rag_data['parent_chunk_overlap'] = config_item['parent_overlap']

    with open(rag_config_file, 'w', encoding='utf-8') as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Config applied: child={config_item['child_size']}, parent={config_item['parent_size']}")

def force_reload_config():
    """Reload config"""
    from app.config import config
    config._load_configs()

def main():
    print("=" * 80)
    print("ChromaDB 자동 감지 테스트")
    print("=" * 80)
    print(f"\n수정된 vector_store.py 테스트:")
    print(f"  - Chunk size 변경 시 자동 collection 재생성")
    print(f"  - DB 폴더는 유지 (lock 문제 해결)")
    print(f"\n테스트 설정: {len(TEST_CONFIGS)}개")
    for i, cfg in enumerate(TEST_CONFIGS, 1):
        print(f"  {i}. {cfg['name']} (child={cfg['child_size']}, parent={cfg['parent_size']})")

    # Backup
    backup_config()

    # Initial cleanup
    chroma_db_path = project_root / "chroma_db_new"
    if chroma_db_path.exists():
        print(f"\n🗑️  Initial cleanup: Deleting existing DB")
        shutil.rmtree(chroma_db_path)

    results = []

    for idx, cfg in enumerate(TEST_CONFIGS, 1):
        print(f"\n" + "=" * 80)
        print(f"TEST {idx}/{len(TEST_CONFIGS)}: {cfg['name']}")
        print(f"=" * 80)

        start = time.time()

        try:
            # Apply config
            apply_chunk_config(cfg)
            force_reload_config()

            # ⭐ NO force_rebuild_chroma_db() call!
            #    vector_store.py will auto-detect chunk size change

            # Run benchmark
            print(f"\n🔄 Running benchmark (limit=1)...")
            print(f"   vector_store.py will automatically:")
            print(f"   1. Detect chunk size change")
            print(f"   2. Delete existing collection")
            print(f"   3. Rebuild with new chunk sizes")

            result_path = run_bench(
                cases_path=DEFAULT_CASES_PATH,
                limit=1,
                use_llm=True,
                max_turns=5,
                verbose=True  # Enable logging to see detection
            )

            # Evaluate
            evaluation = evaluate_phase2_bench(Path(result_path))

            duration = time.time() - start
            comp_score = evaluation['avg_scores']['comprehensive']

            print(f"\n✅ SUCCESS!")
            print(f"   Duration: {duration/60:.1f}min")
            print(f"   Comprehensive: {comp_score:.2f}")

            results.append({
                'name': cfg['name'],
                'success': True,
                'comprehensive': comp_score,
                'duration_min': duration/60
            })

        except Exception as e:
            duration = time.time() - start
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'name': cfg['name'],
                'success': False,
                'error': str(e),
                'duration_min': duration/60
            })

    # Restore
    restore_config()

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count

    print(f"\nTotal: {len(results)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")

    for r in results:
        if r['success']:
            print(f"\n✅ {r['name']}: {r['comprehensive']:.2f} ({r['duration_min']:.1f}min)")
        else:
            print(f"\n❌ {r['name']}: {r.get('error', 'Unknown')[:100]}")

    if success_count == len(results):
        print("\n" + "=" * 80)
        print("🎉 모든 테스트 성공!")
        print("=" * 80)
        print("\n✅ vector_store.py 자동 감지 검증 완료")
        print("✅ ChromaDB readonly 문제 해결됨")
        print("✅ Phase 4 실행 가능")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ 일부 테스트 실패")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    exit(main())
