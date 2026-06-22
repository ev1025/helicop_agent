#!/usr/bin/env python3
"""
Phase 2 다중 설정 벤치마크 테스트 스크립트

여러 RAG 설정을 테스트하고 결과를 비교합니다.
"""

import sys
from pathlib import Path
import json
import shutil
from datetime import datetime
from collections import Counter
import importlib

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

# 벤치마크 및 평가 스크립트 import
from scripts.run_phase2_bench import run_bench, load_cases, DEFAULT_CASES_PATH
from scripts.comprehensive_evaluation_phase2 import evaluate_phase2_bench

print(f"✅ 모듈 import 성공")

# RAG 설정 파일 경로
RAG_CONFIG_PATH = project_root / "config" / "rag.json"

# 테스트할 설정 목록
test_configs = [
    # 현재 설정 (baseline)
    (
        "baseline_th0.7_k10_rk3",
        {
            "rerank_score_threshold": 0.7,
            "vector_search_top_k": 10,
            "rerank_top_k": 3
        }
    ),
    # Threshold 낮춤 (Parent chunk용)
    (
        "low_th0.5_k10_rk3",
        {
            "rerank_score_threshold": 0.5,
            "vector_search_top_k": 10,
            "rerank_top_k": 3
        }
    ),
    # 더 많은 후보 검색
    (
        "more_cand_th0.7_k20_rk5",
        {
            "rerank_score_threshold": 0.7,
            "vector_search_top_k": 20,
            "rerank_top_k": 5
        }
    ),
    # 낮은 threshold + 더 많은 후보
    (
        "balanced_th0.5_k20_rk5",
        {
            "rerank_score_threshold": 0.5,
            "vector_search_top_k": 20,
            "rerank_top_k": 5
        }
    ),
    # Aggressive (많은 문서)
    (
        "aggressive_th0.5_k30_rk7",
        {
            "rerank_score_threshold": 0.5,
            "vector_search_top_k": 30,
            "rerank_top_k": 7
        }
    ),
    # Reference branch 설정
    (
        "reference_th0.7_k50_rk5",
        {
            "rerank_score_threshold": 0.7,
            "vector_search_top_k": 50,
            "rerank_top_k": 5
        }
    ),
]

print(f"\n테스트할 설정: {len(test_configs)}개")
print(f"예상 소요 시간: {len(test_configs) * 17}분 (설정당 평균 17분)")
print("\n설정 목록:")
for i, (name, config) in enumerate(test_configs, 1):
    print(f"  {i}. {name}")
    print(f"     threshold={config['rerank_score_threshold']}, "
          f"top_k={config['vector_search_top_k']}, "
          f"rerank_top_k={config['rerank_top_k']}")


def backup_config():
    """현재 설정을 백업"""
    backup_path = RAG_CONFIG_PATH.with_suffix('.json.backup')
    shutil.copy(RAG_CONFIG_PATH, backup_path)
    print(f"✅ 설정 백업: {backup_path}")
    return backup_path


def restore_config(backup_path):
    """설정 복원"""
    shutil.copy(backup_path, RAG_CONFIG_PATH)
    print(f"✅ 설정 복원: {RAG_CONFIG_PATH}")


def update_rag_config(changes):
    """RAG 설정 업데이트"""
    with open(RAG_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config.update(changes)

    with open(RAG_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ RAG 설정 업데이트: {changes}")


def reload_models():
    """모듈 reload (설정 변경 반영)"""
    from app import config as app_config
    from app.core import models

    importlib.reload(app_config.config)
    importlib.reload(models)

    print(f"✅ 모듈 reload 완료")


def main():
    # 설정 백업
    backup_path = backup_config()

    # 결과 저장
    results_summary = []

    try:
        for i, (config_name, config_changes) in enumerate(test_configs, 1):
            print("\n" + "=" * 70)
            print(f"[{i}/{len(test_configs)}] 설정: {config_name}")
            print("=" * 70)
            print(f"변경 사항: {config_changes}")

            # 설정 업데이트
            update_rag_config(config_changes)

            # 모델 재로딩 (설정 반영)
            reload_models()

            # 벤치마크 실행
            print(f"\n⏱️  벤치마크 실행 중... (예상 15-20분)")
            result_path = run_bench(
                cases_path=DEFAULT_CASES_PATH,
                limit=None,  # 전체 실행
                use_llm=True,  # Phase 2 (LLM 기반)
                max_turns=5,
                verbose=True
            )

            # 결과 파일 이름 변경 (설정 정보 포함)
            result_file = Path(result_path)
            new_name = f"phase2_bench_{config_name}.json"
            new_path = result_file.parent / new_name

            # 기존 파일이 있으면 삭제
            if new_path.exists():
                new_path.unlink()

            shutil.move(result_path, new_path)
            print(f"✅ 결과 저장: {new_path}")

            # 종합 평가
            print(f"\n📊 종합 평가 중...")
            eval_result = evaluate_phase2_bench(new_path)

            # 결과 요약 저장
            results_summary.append({
                'config_name': config_name,
                'config': config_changes,
                'result_path': str(new_path),
                'scores': eval_result['avg_scores'],
                'grade_distribution': eval_result['grade_distribution']
            })

            print(f"\n✅ [{i}/{len(test_configs)}] 완료: {config_name}")
            print(f"   종합 점수: {eval_result['avg_scores']['comprehensive']:.2f}점")
            print(f"   RAG 품질: {eval_result['avg_scores']['rag_quality']:.2f}점")

    finally:
        # 설정 복원
        print("\n" + "=" * 70)
        print("설정 복원")
        print("=" * 70)
        restore_config(backup_path)
        reload_models()

    print("\n" + "=" * 70)
    print("전체 벤치마크 완료!")
    print("=" * 70)

    # 결과 분석
    print_results_summary(results_summary)

    # 결과 저장
    save_results_summary(results_summary)


def print_results_summary(results_summary):
    """결과 요약 출력"""
    print("\n" + "=" * 80)
    print("설정별 종합 점수 비교")
    print("=" * 80)

    # 종합 점수로 정렬
    sorted_results = sorted(results_summary,
                           key=lambda x: x['scores']['comprehensive'],
                           reverse=True)

    print(f"\n{'순위':<6} {'설정 이름':<30} {'종합 점수':<12} {'RAG 품질':<12} {'등급':<10}")
    print("-" * 80)

    for i, result in enumerate(sorted_results, 1):
        comp_score = result['scores']['comprehensive']
        rag_score = result['scores']['rag_quality']

        # 등급 계산
        grade = 'S' if comp_score >= 90 else 'A' if comp_score >= 80 else \
                'B' if comp_score >= 70 else 'C' if comp_score >= 60 else \
                'D' if comp_score >= 50 else 'F'

        marker = '🏆' if i == 1 else ''
        print(f"{i:<6} {result['config_name']:<30} {comp_score:>10.2f} {rag_score:>12.2f}  {grade:<10} {marker}")

    # 최고 성능 설정
    best_config = sorted_results[0]
    print(f"\n" + "=" * 80)
    print(f"🏆 최고 성능 설정: {best_config['config_name']}")
    print("=" * 80)
    print(f"설정:")
    for key, value in best_config['config'].items():
        print(f"  {key}: {value}")
    print(f"\n성능:")
    print(f"  종합 점수: {best_config['scores']['comprehensive']:.2f}점")
    print(f"  RAG 품질: {best_config['scores']['rag_quality']:.2f}점")
    print(f"  조사/괄호: {best_config['scores']['grammatical_particles']:.2f}점")
    print(f"  반복 문구: {best_config['scores']['exact_repetition']:.2f}점")
    print(f"  중복 표현: {best_config['scores']['semantic_redundancy']:.2f}점")
    print(f"  장황함: {best_config['scores']['verbosity']:.2f}점")


def save_results_summary(results_summary):
    """요약 결과를 JSON 파일로 저장"""
    summary_path = project_root / "results" / f"phase2_config_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 최고 성능 설정
    sorted_results = sorted(results_summary,
                           key=lambda x: x['scores']['comprehensive'],
                           reverse=True)
    best = sorted_results[0]

    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'test_count': len(results_summary),
        'best_config': {
            'name': best['config_name'],
            'config': best['config'],
            'scores': best['scores']
        },
        'all_results': results_summary
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 요약 결과 저장: {summary_path}")


if __name__ == "__main__":
    main()
