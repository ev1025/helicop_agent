#!/usr/bin/env python3
"""Grid Search 결과 분석"""

import pandas as pd
from pathlib import Path

# CSV 로드
csv_path = Path(__file__).parent / "results" / "grid_search_summary.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("Grid Search 결과 종합 분석")
print("=" * 80)

print(f"\n총 {len(df)}개 조합 테스트 완료")
print(f"평균 소요 시간: {df['duration_min'].mean():.1f}분")
print(f"총 소요 시간: {df['duration_min'].sum() / 60:.1f}시간")

# 반복 문구 80점 이상 달성 조합
print("\n" + "=" * 80)
print("🎯 반복 문구 80점 이상 달성 조합")
print("=" * 80)
df_80plus = df[df['exact_repetition'] >= 80.0].sort_values('exact_repetition', ascending=False)
print(f"\n총 {len(df_80plus)}개 조합이 80점 이상 달성! ({len(df_80plus)/len(df)*100:.1f}%)")

# Top 10 (반복 문구 기준)
print("\n" + "=" * 80)
print("🏆 Top 10 조합 (반복 문구 점수 기준)")
print("=" * 80)
top10_rep = df.nlargest(10, 'exact_repetition')
for idx, row in top10_rep.iterrows():
    print(f"\n#{row['idx']:02d} - {row['name']}")
    print(f"   반복 문구: {row['exact_repetition']:.2f} | 종합: {row['comprehensive']:.2f} | RAG: {row['rag_quality']:.2f}")
    print(f"   설정: sim={row['similarity_threshold']}, dedup={row['dedup_strategy']}, prompt={row['prompt_version']}")
    print(f"         rep_penalty={row['repetition_penalty']}, temp={row['temperature']}")

# Top 10 (종합 점수 기준)
print("\n" + "=" * 80)
print("🏆 Top 10 조합 (종합 점수 기준)")
print("=" * 80)
top10_comp = df.nlargest(10, 'comprehensive')
for idx, row in top10_comp.iterrows():
    print(f"\n#{row['idx']:02d} - {row['name']}")
    print(f"   종합: {row['comprehensive']:.2f} | 반복 문구: {row['exact_repetition']:.2f} | RAG: {row['rag_quality']:.2f}")
    print(f"   설정: sim={row['similarity_threshold']}, dedup={row['dedup_strategy']}, prompt={row['prompt_version']}")

# 최종 추천 (80점 이상 + 종합 점수 최고)
print("\n" + "=" * 80)
print("✨ 최종 추천 조합")
print("=" * 80)

if len(df_80plus) > 0:
    best = df_80plus.nlargest(1, 'comprehensive').iloc[0]
    print(f"\n🥇 반복 문구 80점 달성 + 종합 점수 최고")
    print(f"   Config #{best['idx']:03d}: {best['name']}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   📊 점수:")
    print(f"      - 반복 문구: {best['exact_repetition']:.2f} ⭐")
    print(f"      - 종합: {best['comprehensive']:.2f}")
    print(f"      - 의미적 중복: {best['semantic_redundancy']:.2f}")
    print(f"      - RAG 품질: {best['rag_quality']:.2f}")
    print(f"   ⚙️  설정:")
    print(f"      - similarity_threshold: {best['similarity_threshold']}")
    print(f"      - dedup_strategy: {best['dedup_strategy']}")
    print(f"      - max_sub_queries: {int(best['max_sub_queries'])}")
    print(f"      - top_k_per_query: {int(best['top_k_per_query'])}")
    print(f"      - final_top_k: {int(best['final_top_k'])}")
    print(f"      - prompt_version: {best['prompt_version']}")
    print(f"      - repetition_penalty: {best['repetition_penalty']}")
    print(f"      - temperature: {best['temperature']}")

# Phase별 최고 조합
print("\n" + "=" * 80)
print("📈 Phase별 최고 조합")
print("=" * 80)

phase_names = {
    1: "중복 제거 전략",
    2: "Multi-Query 파라미터",
    3: "프롬프트 전략",
    4: "LLM 생성 파라미터"
}

for phase in [1, 2, 3, 4]:
    phase_df = df[df['phase'] == phase]
    if len(phase_df) > 0:
        best_phase = phase_df.nlargest(1, 'exact_repetition').iloc[0]
        print(f"\nPhase {phase} ({phase_names[phase]})")
        print(f"   최고: {best_phase['name']}")
        print(f"   반복 문구: {best_phase['exact_repetition']:.2f} | 종합: {best_phase['comprehensive']:.2f}")

# 인사이트 분석
print("\n" + "=" * 80)
print("💡 핵심 인사이트")
print("=" * 80)

# Phase 1: dedup_strategy 영향
phase1 = df[df['phase'] == 1]
print("\n1️⃣ 중복 제거 전략 영향:")
for strategy in ['string_only', 'embedding_only', 'hybrid', 'hybrid_strict']:
    strategy_df = phase1[phase1['dedup_strategy'] == strategy]
    avg_rep = strategy_df['exact_repetition'].mean()
    avg_comp = strategy_df['comprehensive'].mean()
    print(f"   {strategy:16s}: 반복={avg_rep:.2f}, 종합={avg_comp:.2f}")

# Phase 1: similarity_threshold 영향
print("\n2️⃣ Similarity Threshold 영향:")
for sim_th in [0.7, 0.75, 0.8, 0.85, 0.9]:
    sim_df = phase1[phase1['similarity_threshold'] == sim_th]
    avg_rep = sim_df['exact_repetition'].mean()
    avg_comp = sim_df['comprehensive'].mean()
    print(f"   {sim_th:.2f}: 반복={avg_rep:.2f}, 종합={avg_comp:.2f}")

# Phase 3: prompt version 영향
phase3 = df[df['phase'] == 3]
print("\n3️⃣ 프롬프트 버전 영향:")
for prompt_ver in ['v1', 'v2', 'v3']:
    prompt_df = phase3[phase3['prompt_version'] == prompt_ver]
    avg_rep = prompt_df['exact_repetition'].mean()
    avg_sem = prompt_df['semantic_redundancy'].mean()
    avg_comp = prompt_df['comprehensive'].mean()
    print(f"   {prompt_ver}: 반복={avg_rep:.2f}, 의미중복={avg_sem:.2f}, 종합={avg_comp:.2f}")

# Phase 4: repetition_penalty 영향
phase4 = df[df['phase'] == 4]
print("\n4️⃣ Repetition Penalty 영향:")
for rep_pen in [1.0, 1.1, 1.2, 1.3]:
    rep_df = phase4[phase4['repetition_penalty'] == rep_pen]
    avg_rep = rep_df['exact_repetition'].mean()
    avg_comp = rep_df['comprehensive'].mean()
    print(f"   {rep_pen:.1f}: 반복={avg_rep:.2f}, 종합={avg_comp:.2f}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
