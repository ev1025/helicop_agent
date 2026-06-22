#!/usr/bin/env python3
"""모든 평가 지표 추출 및 교차 분석"""

import json
from pathlib import Path

# 결과 디렉토리
grid_search_dir = Path("/home/user/surion_llm/results/grid_search")

# 모든 config 파일 로드
all_results = []
for config_file in sorted(grid_search_dir.glob("config_*.json")):
    with open(config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config = data['config']
    eval_data = data['evaluation']

    result = {
        'idx': data['config_idx'],
        'phase': config['phase'],
        'name': config['name'],
        'similarity_threshold': config['multi_query']['similarity_threshold'],
        'dedup_strategy': config['multi_query'].get('dedup_strategy', 'hybrid'),
        'prompt_version': config['prompt_version'],
        'repetition_penalty': config['llm_params']['repetition_penalty'],
        'temperature': config['llm_params']['temperature'],
        # 전체 평가 지표
        'grammatical_particles': eval_data['avg_scores']['grammatical_particles'],
        'exact_repetition': eval_data['avg_scores']['exact_repetition'],
        'semantic_redundancy': eval_data['avg_scores']['semantic_redundancy'],
        'verbosity': eval_data['avg_scores']['verbosity'],
        'rag_quality': eval_data['avg_scores']['rag_quality'],
        'comprehensive': eval_data['avg_scores']['comprehensive']
    }
    all_results.append(result)

# 정렬 및 출력
print("=" * 120)
print("전체 평가 지표 종합")
print("=" * 120)
print(f"\n총 {len(all_results)}개 조합 분석\n")

# 각 지표별 통계
metrics = ['grammatical_particles', 'exact_repetition', 'semantic_redundancy', 'verbosity', 'rag_quality', 'comprehensive']
print("지표별 통계:")
print("-" * 120)
for metric in metrics:
    values = [r[metric] for r in all_results]
    avg = sum(values) / len(values)
    min_val = min(values)
    max_val = max(values)
    print(f"{metric:25s}: 평균 {avg:6.2f} | 최소 {min_val:6.2f} | 최대 {max_val:6.2f}")

# 교차 분석: Top 5 by each metric
print("\n" + "=" * 120)
print("각 지표별 Top 5 조합")
print("=" * 120)

for metric in metrics:
    print(f"\n【{metric}】 Top 5:")
    print("-" * 120)
    sorted_results = sorted(all_results, key=lambda x: x[metric], reverse=True)[:5]
    for i, r in enumerate(sorted_results, 1):
        print(f"{i}. Config #{r['idx']:03d} - {r['name']:30s}")
        print(f"   {metric}: {r[metric]:6.2f} | 종합: {r['comprehensive']:6.2f} | 반복: {r['exact_repetition']:6.2f} | RAG: {r['rag_quality']:6.2f}")
        print(f"   설정: sim={r['similarity_threshold']}, dedup={r['dedup_strategy']}, prompt={r['prompt_version']}, rep={r['repetition_penalty']}, temp={r['temperature']}")

# 종합 점수 Top 10 with all metrics
print("\n" + "=" * 120)
print("🏆 종합 점수 Top 10 (전체 지표 포함)")
print("=" * 120)
sorted_by_comp = sorted(all_results, key=lambda x: x['comprehensive'], reverse=True)[:10]

for i, r in enumerate(sorted_by_comp, 1):
    print(f"\n{i}. Config #{r['idx']:03d} - {r['name']}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   종합: {r['comprehensive']:6.2f} ⭐")
    print(f"   조사오류: {r['grammatical_particles']:6.2f} | 반복문구: {r['exact_repetition']:6.2f} | 의미중복: {r['semantic_redundancy']:6.2f}")
    print(f"   장황함:   {r['verbosity']:6.2f} | RAG품질:  {r['rag_quality']:6.2f}")
    print(f"   설정: sim={r['similarity_threshold']}, dedup={r['dedup_strategy']}, prompt={r['prompt_version']}, rep={r['repetition_penalty']}, temp={r['temperature']}")

# 최고의 균형 조합 찾기 (모든 지표가 평균 이상)
print("\n" + "=" * 120)
print("✨ 모든 지표가 평균 이상인 균형 잡힌 조합")
print("=" * 120)

avg_values = {metric: sum(r[metric] for r in all_results) / len(all_results) for metric in metrics}
balanced = []
for r in all_results:
    if all(r[metric] >= avg_values[metric] for metric in metrics):
        balanced.append(r)

balanced_sorted = sorted(balanced, key=lambda x: x['comprehensive'], reverse=True)

print(f"\n총 {len(balanced)}개 조합이 모든 지표 평균 이상 달성")
if balanced_sorted:
    print("\nTop 5 균형 조합:")
    for i, r in enumerate(balanced_sorted[:5], 1):
        print(f"\n{i}. Config #{r['idx']:03d} - {r['name']}")
        print(f"   종합: {r['comprehensive']:6.2f}")
        print(f"   조사: {r['grammatical_particles']:6.2f} (평균 {avg_values['grammatical_particles']:.2f})")
        print(f"   반복: {r['exact_repetition']:6.2f} (평균 {avg_values['exact_repetition']:.2f})")
        print(f"   의미: {r['semantic_redundancy']:6.2f} (평균 {avg_values['semantic_redundancy']:.2f})")
        print(f"   장황: {r['verbosity']:6.2f} (평균 {avg_values['verbosity']:.2f})")
        print(f"   RAG:  {r['rag_quality']:6.2f} (평균 {avg_values['rag_quality']:.2f})")

# 최종 추천
print("\n" + "=" * 120)
print("🎯 최종 추천 조합")
print("=" * 120)

best_overall = sorted_by_comp[0]
print(f"\n🥇 최고 종합 점수: Config #{best_overall['idx']:03d} - {best_overall['name']}")
print(f"   종합: {best_overall['comprehensive']:.2f}")
print(f"   조사오류: {best_overall['grammatical_particles']:.2f}")
print(f"   반복문구: {best_overall['exact_repetition']:.2f}")
print(f"   의미중복: {best_overall['semantic_redundancy']:.2f}")
print(f"   장황함:   {best_overall['verbosity']:.2f}")
print(f"   RAG품질:  {best_overall['rag_quality']:.2f}")
print(f"\n   설정:")
print(f"   - similarity_threshold: {best_overall['similarity_threshold']}")
print(f"   - dedup_strategy: {best_overall['dedup_strategy']}")
print(f"   - prompt_version: {best_overall['prompt_version']}")
print(f"   - repetition_penalty: {best_overall['repetition_penalty']}")
print(f"   - temperature: {best_overall['temperature']}")

if balanced_sorted and balanced_sorted[0]['idx'] != best_overall['idx']:
    best_balanced = balanced_sorted[0]
    print(f"\n🥈 최고 균형 조합: Config #{best_balanced['idx']:03d} - {best_balanced['name']}")
    print(f"   (모든 지표가 평균 이상이면서 종합 점수 최고)")
    print(f"   종합: {best_balanced['comprehensive']:.2f}")

print("\n" + "=" * 120)
