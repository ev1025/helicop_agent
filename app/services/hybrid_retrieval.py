"""
Hybrid Retrieval Service

Semantic Search + Keyword Search를 결합한 Hybrid Retrieval을 제공합니다.
Semantic 검색이 부족한 결과를 반환할 때 Keyword 검색으로 보완합니다.
"""

import logging
from typing import List, Dict, Any, Set
from app.services.rag_service import rag_search_with_rerank, rerank_documents, get_all_documents_from_db
from app.services.multi_query import multi_query_retrieval, should_use_multi_query
from app.services.keyword_search import keyword_search
from app.config import config

logger = logging.getLogger(__name__)


def hybrid_retrieval(
    question: str,
    min_results_threshold: int = 5,
    use_multi_query: bool = None,
    final_top_k: int = None,
    keyword_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Hybrid Retrieval 수행 (Semantic + Keyword)

    1. Semantic 검색 먼저 시도 (Multi-Query or Single Query)
    2. 결과가 min_results_threshold 미만이면 Keyword 검색 추가
    3. 두 결과를 Merge하고 Rerank

    Args:
        question: 사용자 질문
        min_results_threshold: Semantic 검색 최소 결과 수 (이하면 Keyword 추가, 기본값: 5)
        use_multi_query: Multi-Query 사용 여부 (None이면 자동 감지)
        final_top_k: 최종 반환할 문서 수 (기본값: config.RERANKER_TOP_K)
        keyword_weight: Keyword 검색 결과의 가중치 (기본값: 0.5)

    Returns:
        List[Dict[str, Any]]: 최종 선택된 문서 리스트

    Note:
        - Semantic 검색이 충분한 결과를 반환하면 Keyword 검색 스킵
        - Keyword 검색은 전체 DB에서 BM25로 검색
        - 최종 결과는 원본 질문으로 Rerank
    """
    if final_top_k is None:
        final_top_k = config.RERANKER_TOP_K

    logger.info(f"Hybrid Retrieval 시작: '{question}'")

    # 1단계: Semantic 검색 수행
    logger.info("1단계: Semantic 검색 수행")

    # Multi-Query 사용 여부 자동 감지
    if use_multi_query is None:
        use_multi_query = should_use_multi_query(question)

    if use_multi_query:
        logger.info("Multi-Query Retrieval 사용")
        semantic_results = multi_query_retrieval(
            question=question,
            max_sub_queries=3,
            top_k_per_query=5,
            final_top_k=10  # 많이 가져와서 Keyword와 Merge
        )
    else:
        logger.info("단일 쿼리 검색 사용")
        semantic_results = rag_search_with_rerank(
            query=question,
            top_k=10,
            reranker_top_k=10  # 많이 가져와서 Keyword와 Merge
        )

    logger.info(f"Semantic 검색 결과: {len(semantic_results)}개 문서")

    # 2단계: Keyword 검색 필요 여부 판단
    need_keyword_search = len(semantic_results) < min_results_threshold

    if not need_keyword_search:
        logger.info(f"Semantic 검색 결과 충분 ({len(semantic_results)} >= {min_results_threshold}), Keyword 검색 스킵")

        # 최종 Rerank만 수행
        if len(semantic_results) > final_top_k:
            document_contents = [doc['content'] for doc in semantic_results]

            # Reranker 사용 가능 여부 체크
            if config.USE_RERANKER and models.reranker_model is not None:
                reranked_results = rerank_documents(
                    query=question,
                    documents=document_contents,
                    top_k=final_top_k
                )
            else:
                # Reranker 없으면 상위 N개만 선택
                reranked_results = [(doc, 1.0) for doc in document_contents[:final_top_k]]

            # Rerank 점수로 업데이트
            final_documents = []
            for reranked_content, reranker_score in reranked_results:
                for doc in semantic_results:
                    if doc['content'][:100] == reranked_content[:100]:
                        final_doc = doc.copy()
                        final_doc['score'] = reranker_score
                        final_documents.append(final_doc)
                        break

            return final_documents
        else:
            return semantic_results[:final_top_k]

    # 3단계: Keyword 검색 수행
    logger.info(f"2단계: Keyword Fallback (Semantic 결과 부족: {len(semantic_results)} < {min_results_threshold})")

    try:
        # 전체 DB에서 문서 가져오기
        all_documents = get_all_documents_from_db()

        if not all_documents:
            logger.warning("DB에서 문서를 가져올 수 없음, Semantic 결과만 반환")
            return semantic_results[:final_top_k]

        logger.info(f"전체 DB 문서: {len(all_documents)}개")

        # BM25 키워드 검색
        keyword_results = keyword_search(
            query=question,
            documents=all_documents,
            top_k=10,
            min_score=0.0
        )

        logger.info(f"Keyword 검색 결과: {len(keyword_results)}개 문서")

    except Exception as e:
        logger.error(f"Keyword 검색 오류: {e}")
        logger.warning("Keyword 검색 실패, Semantic 결과만 반환")
        return semantic_results[:final_top_k]

    # 4단계: 두 결과 Merge (중복 제거)
    logger.info("3단계: Semantic + Keyword 결과 Merge")

    merged_documents = []
    seen_contents: Set[str] = set()

    # Semantic 결과 추가 (높은 우선순위, 가중치 1.0)
    for doc in semantic_results:
        content = doc['content']
        content_key = content[:100]

        if content_key not in seen_contents:
            merged_doc = doc.copy()
            # Semantic 점수는 그대로 유지
            merged_documents.append(merged_doc)
            seen_contents.add(content_key)

    # Keyword 결과 추가 (낮은 우선순위, 가중치 조정)
    for doc in keyword_results:
        content = doc['content']
        content_key = content[:100]

        if content_key not in seen_contents:
            merged_doc = doc.copy()
            # BM25 점수를 가중치로 조정
            if 'bm25_score' in merged_doc:
                merged_doc['score'] = merged_doc['bm25_score'] * keyword_weight
            merged_documents.append(merged_doc)
            seen_contents.add(content_key)

    logger.info(f"Merge 결과: {len(merged_documents)}개 문서 (Semantic: {len(semantic_results)}, Keyword 추가: {len(merged_documents) - len(semantic_results)})")

    # 5단계: 최종 Rerank
    logger.info(f"4단계: 원본 질문으로 최종 Rerank (top_k={final_top_k})")

    if len(merged_documents) <= final_top_k:
        # Rerank 필요없음
        return merged_documents

    # 문서 내용만 추출
    document_contents = [doc['content'] for doc in merged_documents]

    # Reranker 사용 가능 여부 체크
    if config.USE_RERANKER and models.reranker_model is not None:
        # Rerank 수행
        reranked_results = rerank_documents(
            query=question,
            documents=document_contents,
            top_k=final_top_k
        )
    else:
        # Reranker 없으면 상위 N개만 선택
        reranked_results = [(doc, 1.0) for doc in document_contents[:final_top_k]]

    # Rerank 점수로 문서 정보 업데이트
    final_documents = []
    for reranked_content, reranker_score in reranked_results:
        # 원본 문서 찾기
        original_doc = None
        for doc in merged_documents:
            if doc['content'][:100] == reranked_content[:100]:
                original_doc = doc
                break

        if original_doc:
            # Rerank 점수로 업데이트
            final_doc = original_doc.copy()
            final_doc['score'] = reranker_score
            final_documents.append(final_doc)

    # Threshold 필터링
    filtered_documents = [
        doc for doc in final_documents
        if doc['score'] >= config.RERANKER_SCORE_THRESHOLD
    ]

    logger.info(f"Hybrid Retrieval 최종 결과: {len(filtered_documents)}개 문서 (Threshold {config.RERANKER_SCORE_THRESHOLD} 이상)")

    return filtered_documents
