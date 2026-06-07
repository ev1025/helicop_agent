"""
Keyword-based Search Service

BM25 알고리즘을 이용한 키워드 기반 검색을 제공합니다.
Semantic 검색이 실패했을 때 Fallback으로 사용됩니다.
"""

import logging
import re
from typing import List, Dict, Any
from collections import Counter
import math

logger = logging.getLogger(__name__)


def extract_korean_keywords(text: str, min_length: int = 2) -> List[str]:
    """
    한국어 텍스트에서 핵심 키워드 추출

    Args:
        text: 입력 텍스트
        min_length: 최소 키워드 길이 (기본값: 2)

    Returns:
        List[str]: 추출된 키워드 리스트

    Note:
        - 한글 명사/동사/형용사 패턴 추출
        - 조사, 특수문자 제거
        - 영어 단어 포함 (3자 이상)
    """
    keywords = []

    # 한글 단어 추출 (2자 이상)
    korean_words = re.findall(r'[가-힣]{' + str(min_length) + r',}', text)
    keywords.extend(korean_words)

    # 영어 단어 추출 (3자 이상, 대소문자 무시)
    english_words = re.findall(r'[a-zA-Z]{3,}', text)
    keywords.extend([w.lower() for w in english_words])

    # 숫자 포함 단어 추출
    alphanumeric = re.findall(r'[a-zA-Z가-힣]+\d+|\d+[a-zA-Z가-힣]+', text)
    keywords.extend(alphanumeric)

    return keywords


def compute_bm25_score(
    query_keywords: List[str],
    doc_keywords: List[str],
    avg_doc_length: float,
    total_docs: int,
    doc_frequencies: Dict[str, int],
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """
    BM25 점수 계산

    Args:
        query_keywords: 쿼리 키워드 리스트
        doc_keywords: 문서 키워드 리스트
        avg_doc_length: 전체 문서의 평균 길이
        total_docs: 전체 문서 수
        doc_frequencies: 각 키워드의 문서 빈도 (DF)
        k1: BM25 파라미터 (기본값: 1.5)
        b: BM25 파라미터 (기본값: 0.75)

    Returns:
        float: BM25 점수

    Note:
        BM25 = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
        - IDF: Inverse Document Frequency
        - f(qi, D): 쿼리 키워드 qi의 문서 D에서의 빈도
        - |D|: 문서 D의 길이
        - avgdl: 평균 문서 길이
    """
    doc_length = len(doc_keywords)
    doc_keyword_counts = Counter(doc_keywords)

    score = 0.0

    for query_kw in set(query_keywords):
        # 문서에 키워드가 없으면 스킵
        if query_kw not in doc_keyword_counts:
            continue

        # TF: 문서 내 키워드 빈도
        tf = doc_keyword_counts[query_kw]

        # DF: 키워드를 포함하는 문서 수
        df = doc_frequencies.get(query_kw, 1)

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

        # 문서 길이 정규화
        norm = 1 - b + b * (doc_length / avg_doc_length)

        # BM25 점수
        score += idf * (tf * (k1 + 1)) / (tf + k1 * norm)

    return score


def keyword_search(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 10,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    키워드 기반 BM25 검색 수행

    Args:
        query: 검색 쿼리
        documents: 검색할 문서 리스트 (각 문서는 'content' 키 필수)
        top_k: 반환할 문서 수 (기본값: 10)
        min_score: 최소 BM25 점수 (기본값: 0.0)

    Returns:
        List[Dict[str, Any]]: BM25 점수 순으로 정렬된 문서 리스트

    Example:
        >>> docs = [
        ...     {"content": "헬리콥터 메인 로터 시스템", "id": 1},
        ...     {"content": "꼬리 로터 anti-torque 시스템", "id": 2}
        ... ]
        >>> results = keyword_search("메인 로터 구성요소", docs, top_k=2)
        >>> print(results[0]['bm25_score'])
        2.45
    """
    if not documents:
        logger.warning("검색할 문서가 없습니다")
        return []

    # 1단계: 쿼리 키워드 추출
    query_keywords = extract_korean_keywords(query)

    if not query_keywords:
        logger.warning(f"쿼리에서 키워드를 추출할 수 없습니다: '{query}'")
        return []

    logger.info(f"쿼리 키워드: {query_keywords}")

    # 2단계: 문서 키워드 추출 및 통계 계산
    doc_keywords_list = []
    doc_frequencies = Counter()  # 각 키워드가 몇 개의 문서에 나타나는지

    for doc in documents:
        content = doc.get('content', '')
        keywords = extract_korean_keywords(content)
        doc_keywords_list.append(keywords)

        # DF 계산 (각 키워드당 문서 빈도)
        unique_keywords = set(keywords)
        for kw in unique_keywords:
            doc_frequencies[kw] += 1

    # 평균 문서 길이 계산
    total_docs = len(documents)
    avg_doc_length = sum(len(kw_list) for kw_list in doc_keywords_list) / total_docs

    logger.info(f"전체 문서: {total_docs}개, 평균 길이: {avg_doc_length:.1f} 키워드")

    # 3단계: BM25 점수 계산
    scored_documents = []

    for i, doc in enumerate(documents):
        doc_keywords = doc_keywords_list[i]

        bm25_score = compute_bm25_score(
            query_keywords=query_keywords,
            doc_keywords=doc_keywords,
            avg_doc_length=avg_doc_length,
            total_docs=total_docs,
            doc_frequencies=doc_frequencies
        )

        # 최소 점수 필터링
        if bm25_score >= min_score:
            # 원본 문서에 BM25 점수 추가
            scored_doc = doc.copy()
            scored_doc['bm25_score'] = bm25_score
            scored_doc['score'] = bm25_score  # 통일된 점수 필드
            scored_documents.append(scored_doc)

    # 4단계: BM25 점수로 정렬
    scored_documents.sort(key=lambda x: x['bm25_score'], reverse=True)

    # Top-K 선택
    top_documents = scored_documents[:top_k]

    logger.info(f"키워드 검색 결과: {len(top_documents)}개 문서 (전체: {len(scored_documents)}개)")

    if top_documents:
        logger.info(f"최고 점수: {top_documents[0]['bm25_score']:.3f}")
        logger.info(f"최저 점수: {top_documents[-1]['bm25_score']:.3f}")

    return top_documents
