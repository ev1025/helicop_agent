#!/usr/bin/env python3
"""
Phase 2 종합 평가 스크립트 - evaluation_guidelines.md의 5가지 기준 적용

평가 항목:
1. 조사 단독 출현 및 빈 괄호
2. 반복 문구
3. 유사 표현 중복
4. 장황함
5. RAG 기반 답변 품질
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# ============================================================================
# 1. 조사 단독 출현 및 빈 괄호 평가
# ============================================================================

def evaluate_grammatical_particles(answer: str) -> Tuple[float, Dict[str, Any]]:
    """
    조사 단독 출현 및 빈 괄호 평가 (100점 만점)

    Returns:
        (점수, 상세정보)
    """
    # 조사 패턴
    particles = ['은 ', '는 ', '이 ', '가 ', '을 ', '를 ', '에 ', '의 ', '와 ', '과 ', '도 ', '만 ', '부터 ', '까지 ']

    particle_count = 0
    particle_positions = []

    for particle in particles:
        # 문장 시작에 조사
        if answer.startswith(particle):
            particle_count += 1
            particle_positions.append(f"시작: '{particle.strip()}'")

        # 연속 공백 후 조사 (단독 사용 패턴)
        pattern = r'\s{2,}' + re.escape(particle)
        matches = re.finditer(pattern, answer)
        for match in matches:
            particle_count += 1
            particle_positions.append(f"위치 {match.start()}: '{particle.strip()}'")

    # 빈 괄호 검출
    empty_brackets_patterns = [
        r'\(\s*\)',   # ()
        r'\[\s*\]',   # []
    ]

    empty_bracket_count = 0
    empty_bracket_positions = []

    for pattern in empty_brackets_patterns:
        matches = re.finditer(pattern, answer)
        for match in matches:
            empty_bracket_count += 1
            empty_bracket_positions.append(f"위치 {match.start()}: '{match.group()}'")

    # 점수 계산
    score = 100 - (particle_count * 15) - (empty_bracket_count * 10)
    score = max(0, score)

    details = {
        'particle_count': particle_count,
        'empty_bracket_count': empty_bracket_count,
        'particle_positions': particle_positions[:5],
        'empty_bracket_positions': empty_bracket_positions[:5],
        'total_issues': particle_count + empty_bracket_count,
    }

    return score, details


# ============================================================================
# 2. 반복 문구 평가
# ============================================================================

def evaluate_exact_repetition(answer: str) -> Tuple[float, Dict[str, Any]]:
    """
    반복 문구 평가 (100점 만점)
    """
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', answer) if len(s.strip()) >= 10]
    phrases = answer.split()

    # 문장 반복 검출 (10자 이상)
    sentence_counts = {}
    for sent in sentences:
        if len(sent) >= 10:
            sentence_counts[sent] = sentence_counts.get(sent, 0) + 1

    duplicate_sentences = sum(count - 1 for count in sentence_counts.values() if count > 1)
    repeated_sentences = [sent[:50] + '...' if len(sent) > 50 else sent
                          for sent, count in sentence_counts.items() if count > 1]

    # 구절 반복 검출 (5자 이상, 3회 이상)
    phrase_counts = {}
    for phrase in phrases:
        if len(phrase) >= 5:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    duplicate_phrases = sum(max(0, count - 2) for count in phrase_counts.values() if count >= 3)
    repeated_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 3]

    # 점수 계산
    score = 100 - (duplicate_sentences * 20) - (duplicate_phrases * 10)
    score = max(0, score)

    details = {
        'duplicate_sentence_count': duplicate_sentences,
        'duplicate_phrase_count': duplicate_phrases,
        'repeated_sentences': repeated_sentences[:3],
        'repeated_phrases': repeated_phrases[:5],
        'total_duplicates': duplicate_sentences + duplicate_phrases,
    }

    return score, details


# ============================================================================
# 3. 유사 표현 중복 평가
# ============================================================================

def evaluate_semantic_redundancy(answer: str) -> Tuple[float, Dict[str, Any]]:
    """
    유사 표현 중복 평가 (100점 만점)
    """
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', answer) if len(s.strip()) >= 10]

    if len(sentences) < 2:
        return 100.0, {'similar_pair_count': 0, 'similar_pairs': []}

    # 간단한 키워드 추출
    def extract_keywords(text):
        words = re.findall(r'[가-힣]{2,}', text)
        return set(words)

    similar_pairs = []
    similar_pair_count = 0

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sent1, sent2 = sentences[i], sentences[j]

            keywords1 = extract_keywords(sent1)
            keywords2 = extract_keywords(sent2)

            if not keywords1 or not keywords2:
                continue

            # Jaccard 유사도
            intersection = len(keywords1 & keywords2)
            union = len(keywords1 | keywords2)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0.5 and len(sent1) >= 10 and len(sent2) >= 10:
                similar_pair_count += 1
                similar_pairs.append({
                    'sentence1': sent1[:50] + '...' if len(sent1) > 50 else sent1,
                    'sentence2': sent2[:50] + '...' if len(sent2) > 50 else sent2,
                    'similarity': round(similarity, 2),
                })

    # 점수 계산
    score = 100 - (similar_pair_count * 15)
    score = max(0, score)

    details = {
        'similar_pair_count': similar_pair_count,
        'similar_pairs': similar_pairs[:3],
    }

    return score, details


# ============================================================================
# 4. 장황함 평가
# ============================================================================

def evaluate_verbosity(answer_length: int, ideal_length: int) -> Tuple[float, Dict[str, Any]]:
    """
    장황함 평가 (100점 만점)
    """
    if ideal_length == 0:
        return 50.0, {'length_ratio': 0, 'penalty': 'ideal_length is 0'}

    ratio = answer_length / ideal_length

    if 0.8 <= ratio <= 1.2:
        score = 100
    elif 0.6 <= ratio < 0.8:
        score = 100 - ((0.8 - ratio) / 0.2) * 20
    elif 1.2 < ratio <= 1.5:
        score = 100 - ((ratio - 1.2) / 0.3) * 20
    elif 1.5 < ratio <= 2.0:
        score = 80 - ((ratio - 1.5) / 0.5) * 20
    elif 2.0 < ratio <= 3.0:
        score = 60 - ((ratio - 2.0) / 1.0) * 30
    elif ratio > 3.0:
        score = max(0, 30 - ((ratio - 3.0) / 2.0) * 30)
    else:  # ratio < 0.6
        score = max(0, 80 - ((0.6 - ratio) / 0.6) * 80)

    details = {
        'length_ratio': round(ratio, 2),
        'answer_length': answer_length,
        'ideal_length': ideal_length,
    }

    return score, details


# ============================================================================
# 5. RAG 기반 답변 품질 평가
# ============================================================================

def evaluate_rag_quality(
    f1: float,
    precision: float,
    recall: float,
    answer: str,
    rag_content: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    RAG 기반 답변 품질 평가 (100점 만점)

    Args:
        f1: F1 score
        precision: Precision
        recall: Recall
        answer: 생성된 답변
        rag_content: RAG 검색 결과 텍스트 (conversation_history에서 추출)
    """
    # F1 점수
    f1_score = min(40, f1 * 100 * 0.4)

    # Precision 점수
    precision_score = min(15, precision * 100 * 0.15)

    # Recall 점수
    recall_score = min(15, recall * 100 * 0.15)

    # RAG 활용도
    if rag_content:
        rag_keywords = set(re.findall(r'[가-힣]{2,}', rag_content))
        answer_keywords = set(re.findall(r'[가-힣]{2,}', answer))

        if rag_keywords:
            rag_utilization = len(answer_keywords & rag_keywords) / len(rag_keywords)
        else:
            rag_utilization = 0
    else:
        rag_utilization = 0

    rag_score = min(30, rag_utilization * 100 * 0.3)

    # 출처 태그 보너스
    source_tag_bonus = 5 if re.search(r'\[문서\s?\d+', answer) else 0

    # 환각 페널티
    hallucination_penalty = 0
    if not rag_content and len(answer) > 200:
        hallucination_penalty = 10

    # 종합 점수
    total_score = f1_score + precision_score + recall_score + rag_score + source_tag_bonus - hallucination_penalty
    total_score = min(100, max(0, total_score))

    details = {
        'f1_contribution': round(f1_score, 2),
        'precision_contribution': round(precision_score, 2),
        'recall_contribution': round(recall_score, 2),
        'rag_utilization_contribution': round(rag_score, 2),
        'source_tag_bonus': source_tag_bonus,
        'hallucination_penalty': hallucination_penalty,
        'rag_content_length': len(rag_content) if rag_content else 0,
    }

    return total_score, details


# ============================================================================
# 종합 평가
# ============================================================================

def comprehensive_evaluate(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 케이스 종합 평가 (Phase 2 벤치마크 구조)

    Args:
        case_data: Phase 2 벤치마크 케이스 딕셔너리
    """
    case_id = case_data.get('id')
    answer = case_data.get('answer', '')
    ideal = case_data.get('ideal_answer', '')
    metrics = case_data.get('metrics', {})
    conversation = case_data.get('conversation_history', [])

    # 메트릭 추출
    f1 = metrics.get('f1_score', 0.0)
    precision = metrics.get('precision', 0.0)
    recall = metrics.get('recall', 0.0)

    # RAG 검색 결과 추출 (conversation_history에서)
    rag_content = ""
    for msg in conversation:
        if msg.get('role') == 'user' and msg.get('content', '').startswith('[TOOL_RESULT]'):
            rag_content += msg.get('content', '')

    # 5가지 평가 수행
    particle_score, particle_details = evaluate_grammatical_particles(answer)
    repetition_score, repetition_details = evaluate_exact_repetition(answer)
    redundancy_score, redundancy_details = evaluate_semantic_redundancy(answer)
    verbosity_score, verbosity_details = evaluate_verbosity(len(answer), len(ideal))
    rag_score, rag_details = evaluate_rag_quality(f1, precision, recall, answer, rag_content)

    # 종합 점수 (가중 평균)
    comprehensive_score = (
        particle_score * 0.10 +
        repetition_score * 0.10 +
        redundancy_score * 0.10 +
        verbosity_score * 0.15 +
        rag_score * 0.55
    )

    # 등급 결정
    if comprehensive_score >= 90:
        grade = 'S'
    elif comprehensive_score >= 80:
        grade = 'A'
    elif comprehensive_score >= 70:
        grade = 'B'
    elif comprehensive_score >= 60:
        grade = 'C'
    elif comprehensive_score >= 50:
        grade = 'D'
    else:
        grade = 'F'

    return {
        'case_id': case_id,
        'grade': grade,
        'scores': {
            'grammatical_particles': round(particle_score, 2),
            'exact_repetition': round(repetition_score, 2),
            'semantic_redundancy': round(redundancy_score, 2),
            'verbosity': round(verbosity_score, 2),
            'rag_quality': round(rag_score, 2),
            'comprehensive': round(comprehensive_score, 2),
        },
        'details': {
            'grammatical_particles': particle_details,
            'exact_repetition': repetition_details,
            'semantic_redundancy': redundancy_details,
            'verbosity': verbosity_details,
            'rag_quality': rag_details,
        },
        'raw_metrics': {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'answer_length': len(answer),
            'ideal_length': len(ideal),
        }
    }


def evaluate_phase2_bench(bench_path: Path) -> Dict[str, Any]:
    """
    전체 Phase 2 벤치마크 평가

    Args:
        bench_path: phase2_bench_XXXX.json 파일 경로
    """
    with open(bench_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    run_id = data.get('run_id')
    cases = data.get('cases', [])

    results = []
    for case in cases:
        if not case.get('success'):
            continue
        case_result = comprehensive_evaluate(case)
        results.append(case_result)

    # 평균 점수 계산
    if results:
        avg_scores = {
            'grammatical_particles': sum(r['scores']['grammatical_particles'] for r in results) / len(results),
            'exact_repetition': sum(r['scores']['exact_repetition'] for r in results) / len(results),
            'semantic_redundancy': sum(r['scores']['semantic_redundancy'] for r in results) / len(results),
            'verbosity': sum(r['scores']['verbosity'] for r in results) / len(results),
            'rag_quality': sum(r['scores']['rag_quality'] for r in results) / len(results),
            'comprehensive': sum(r['scores']['comprehensive'] for r in results) / len(results),
        }
    else:
        avg_scores = {}

    # 등급 분포
    grade_counts = {}
    for r in results:
        grade = r['grade']
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    return {
        'bench_id': run_id,
        'bench_file': str(bench_path),
        'use_llm': data.get('use_llm'),
        'total_cases': len(cases),
        'evaluated_cases': len(results),
        'avg_scores': {k: round(v, 2) for k, v in avg_scores.items()} if avg_scores else {},
        'grade_distribution': grade_counts,
        'case_results': results,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_evaluation_phase2.py <phase2_bench_XXXX.json>")
        sys.exit(1)

    bench_path = Path(sys.argv[1])

    result = evaluate_phase2_bench(bench_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
