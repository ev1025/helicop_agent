"""
Multi-Query Retrieval 서비스

복합 질문을 여러 서브 쿼리로 분해하여 검색 품질을 향상시킵니다.
"""

import logging
import re
import torch
from typing import List, Dict, Any, Set
from app.core import models
from app.services.rag_service import rag_search_with_rerank, rerank_documents, simplify_query
from app.config import config

logger = logging.getLogger(__name__)


def generate_sub_queries(question: str, max_queries: int = 3) -> List[str]:
    """
    LLM을 사용하여 복합 질문을 여러 서브 쿼리로 분해

    Args:
        question: 사용자 질문
        max_queries: 생성할 최대 쿼리 수

    Returns:
        List[str]: 서브 쿼리 리스트

    Example:
        >>> generate_sub_queries("오토자이로와 헬리콥터의 차이점과 오토자이로의 기여는?")
        ["오토자이로 헬리콥터 차이", "오토자이로 기여", "오토자이로 개발"]
    """
    # 프롬프트 구성
    prompt = f"""다음 질문을 2-3개의 간단한 검색 쿼리로 분해해주세요.

질문: {question}

요구사항:
1. 각 쿼리는 한국어로 작성하세요
2. 각 쿼리는 20자 이내로 작성하세요
3. 핵심 키워드만 포함하세요 (조사, 불필요한 단어 제거)
4. 질문의 모든 측면을 커버하도록 쿼리를 생성하세요
5. 각 쿼리를 한 줄씩 작성하세요

출력 형식 (번호 없이 쿼리만):
<쿼리1>
<쿼리2>
<쿼리3>
"""

    try:
        # LLM으로 서브 쿼리 생성
        # LLM은 항상 GPU 사용 (config에 USE_LLM_GPU 없음)
        device = models.get_device(True)

        # 토큰화
        inputs = models.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # 생성
        models.model.eval()
        with torch.no_grad():
            outputs = models.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=models.tokenizer.eos_token_id
            )

        # 디코딩
        generated_text = models.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 프롬프트 이후의 응답 부분만 추출
        if prompt in generated_text:
            response = generated_text.split(prompt)[-1].strip()
        else:
            # 응답에서 쿼리 부분만 추출 (프롬프트가 포함되지 않은 경우)
            response = generated_text.strip()

        # 서브 쿼리 파싱
        sub_queries = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()

            # 번호나 기호 제거
            line = re.sub(r'^[\d\.\-\*\•\→]+\s*', '', line)
            line = re.sub(r'^[가-힣]\.\s*', '', line)  # "가. " 같은 패턴 제거

            # 빈 줄이나 너무 짧은 줄 제거
            if len(line) < 3:
                continue

            # "쿼리", "검색어" 같은 메타 정보 제거
            if '쿼리' in line or '검색어' in line or '질문' in line:
                # "쿼리1: 오토자이로" -> "오토자이로" 추출
                parts = line.split(':', 1)
                if len(parts) == 2:
                    line = parts[1].strip()
                else:
                    continue

            # 따옴표 제거
            line = line.strip('"\'""''')

            # 유효한 쿼리만 추가
            if len(line) >= 3 and len(line) <= 50:
                # 쿼리 단순화 적용
                simplified = simplify_query(line, max_length=25)
                sub_queries.append(simplified)

            # 최대 개수 도달
            if len(sub_queries) >= max_queries:
                break

        # 최소 1개의 쿼리는 있어야 함
        if not sub_queries:
            # LLM이 실패한 경우, 원본 질문을 단순화하여 사용
            logger.warning("LLM 서브 쿼리 생성 실패, 원본 질문 사용")
            sub_queries = [simplify_query(question, max_length=25)]

        logger.info(f"생성된 서브 쿼리: {sub_queries}")
        return sub_queries[:max_queries]

    except Exception as e:
        logger.error(f"서브 쿼리 생성 오류: {e}")
        # 오류 발생 시 원본 질문을 단순화하여 사용
        return [simplify_query(question, max_length=25)]


def _compute_content_similarity(content1: str, content2: str) -> float:
    """
    두 문서 내용의 의미적 유사도 계산 (Embedding 사용)

    Args:
        content1: 첫 번째 문서 내용
        content2: 두 번째 문서 내용

    Returns:
        float: 코사인 유사도 (0~1)
    """
    try:
        # 기존 get_embeddings 함수 사용 (AutoModel과 호환)
        from app.core.embeddings import get_embeddings

        # 내용이 너무 길면 앞 500자만 사용
        text1 = content1[:500] if len(content1) > 500 else content1
        text2 = content2[:500] if len(content2) > 500 else content2

        # 임베딩 생성 (기존 코드와 동일한 방식)
        embeddings_list = get_embeddings(
            [text1, text2],
            models.embedding_model,
            models.embedding_tokenizer,
            models.get_device(config.USE_EMBEDDING_GPU),
            model_name=config.EMBEDDING_MODEL,
            is_query=False
        )

        # 코사인 유사도 계산
        emb1 = torch.tensor(embeddings_list[0])
        emb2 = torch.tensor(embeddings_list[1])

        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()

        return similarity

    except Exception as e:
        logger.warning(f"유사도 계산 오류: {e}")
        return 0.0


def multi_query_retrieval(
    question: str,
    max_sub_queries: int = 3,
    top_k_per_query: int = 5,
    final_top_k: int = None,
    similarity_threshold: float = 0.7,
    dedup_strategy: str = 'embedding_only'
) -> List[Dict[str, Any]]:
    """
    Multi-Query Retrieval 수행

    1. 질문을 여러 서브 쿼리로 분해
    2. 각 서브 쿼리로 RAG 검색 수행
    3. 결과 통합 및 중복 제거 (의미적 유사도 기반)
    4. 최종 Rerank 수행

    Args:
        question: 사용자 질문
        max_sub_queries: 생성할 최대 서브 쿼리 수 (기본값: 3)
        top_k_per_query: 각 쿼리당 검색할 문서 수 (기본값: 5)
        final_top_k: 최종 선택할 문서 수 (기본값: config.RERANKER_TOP_K)
        similarity_threshold: 중복 판단 유사도 임계값 (기본값: 0.75, 낮춤)
        dedup_strategy: 중복 제거 전략 (기본값: 'hybrid')
            - 'string_only': 앞 100자만 비교 (가장 빠름)
            - 'embedding_only': 임베딩 유사도만 사용
            - 'hybrid': 앞 100자 + 임베딩 (현재 방식)
            - 'hybrid_strict': 앞 200자 + 임베딩

    Returns:
        List[Dict[str, Any]]: 최종 선택된 문서 리스트

    Note:
        - 서브 쿼리별로 검색 후 결과를 통합
        - 중복 문서 제거 (내용 기준, 의미적 유사도 사용)
        - 최종적으로 원본 질문으로 Rerank 수행
    """
    if final_top_k is None:
        final_top_k = config.RERANKER_TOP_K

    logger.info(f"Multi-Query Retrieval 시작: '{question}'")

    # 1단계: 서브 쿼리 생성
    sub_queries = generate_sub_queries(question, max_queries=max_sub_queries)

    if not sub_queries:
        logger.warning("서브 쿼리가 생성되지 않음, 일반 RAG 검색으로 폴백")
        return rag_search_with_rerank(question, top_k=10, reranker_top_k=final_top_k)

    logger.info(f"총 {len(sub_queries)}개의 서브 쿼리 생성됨")

    # 2단계: 각 서브 쿼리로 검색 수행
    all_documents = []

    for i, sub_query in enumerate(sub_queries, 1):
        logger.info(f"서브 쿼리 {i}/{len(sub_queries)}: '{sub_query}'")

        # 서브 쿼리로 RAG 검색
        results = rag_search_with_rerank(
            query=sub_query,
            top_k=10,
            reranker_top_k=top_k_per_query
        )

        # 중복 제거하면서 결과 추가 (전략에 따라)
        for doc in results:
            content = doc['content']
            is_duplicate = False

            # 기존 문서들과 중복 체크
            for existing_doc in all_documents:
                existing_content = existing_doc['content']

                # 전략별 중복 체크
                if dedup_strategy == 'string_only':
                    # 앞 100자만 비교
                    if content[:100] == existing_content[:100]:
                        is_duplicate = True
                        logger.debug(f"중복 제거 (앞 100자 일치): {content[:50]}...")
                        break

                elif dedup_strategy == 'embedding_only':
                    # 임베딩 유사도만 사용
                    similarity = _compute_content_similarity(content, existing_content)
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        logger.debug(f"중복 제거 (유사도 {similarity:.3f}): {content[:50]}...")
                        break

                elif dedup_strategy == 'hybrid_strict':
                    # 앞 200자 체크 + 임베딩
                    if content[:200] == existing_content[:200]:
                        is_duplicate = True
                        logger.debug(f"중복 제거 (앞 200자 일치): {content[:50]}...")
                        break

                    similarity = _compute_content_similarity(content, existing_content)
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        logger.debug(f"중복 제거 (유사도 {similarity:.3f}): {content[:50]}...")
                        break

                else:  # 'hybrid' (기본)
                    # 앞 100자 + 임베딩
                    if content[:100] == existing_content[:100]:
                        is_duplicate = True
                        logger.debug(f"중복 제거 (앞 100자 일치): {content[:50]}...")
                        break

                    similarity = _compute_content_similarity(content, existing_content)
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        logger.debug(f"중복 제거 (유사도 {similarity:.3f}): {content[:50]}...")
                        break

            if not is_duplicate:
                all_documents.append(doc)

        logger.info(f"  → 검색 결과: {len(results)}개 (누적: {len(all_documents)}개, 중복 제거: {len(results) - len([d for d in all_documents if d in results])}개)")

    if not all_documents:
        logger.warning("모든 서브 쿼리에서 검색 결과 없음")
        return []

    logger.info(f"총 {len(all_documents)}개의 고유 문서 수집됨 (의미적 중복 제거 완료)")

    # 3단계: 원본 질문으로 최종 Rerank
    logger.info(f"원본 질문으로 최종 Rerank 수행: top_k={final_top_k}")

    # 문서 내용만 추출
    document_contents = [doc['content'] for doc in all_documents]

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
        for doc in all_documents:
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

    logger.info(f"Multi-Query 최종 결과: {len(filtered_documents)}개 문서 (Threshold {config.RERANKER_SCORE_THRESHOLD} 이상)")

    # 본문 순서(page → parent_chunk_id) 로 재정렬하여 LLM 이 흐름을 잘 파악하도록 함
    from app.services.rag_service import _sort_by_document_order
    filtered_documents = _sort_by_document_order(filtered_documents)
    pages_after = [d.get('page') for d in filtered_documents]
    logger.info(f"Multi-Query 본문 순서 정렬 후 페이지: {pages_after}")

    return filtered_documents


def should_use_multi_query(question: str) -> bool:
    """
    질문이 Multi-Query Retrieval을 사용해야 하는지 판단

    Args:
        question: 사용자 질문

    Returns:
        bool: Multi-Query 사용 여부

    Note:
        다음 패턴이 있으면 Multi-Query 사용:
        - "그리고", "및", "또한", "또는" 같은 접속사
        - "차이", "비교", "vs" 같은 비교 표현
        - 질문이 60자 이상
        - 여러 개의 질문 (?, !가 2개 이상)
    """
    # 패턴 1: 접속사 (며, ~고 추가)
    conjunctions = ['그리고', '및', '또한', '또는', '혹은', '그러나', '하지만', '며', '면서', '이며', '이고']
    if any(conj in question for conj in conjunctions):
        return True

    # 패턴 2: 비교 표현
    comparison_patterns = ['차이', '비교', 'vs', '대비', '상반', '반대']
    if any(pattern in question for pattern in comparison_patterns):
        return True

    # 패턴 3: 긴 질문 (60자 → 55자로 낮춤)
    if len(question) >= 55:
        return True

    # 패턴 4: 여러 개의 질문
    question_marks = question.count('?') + question.count('!')
    if question_marks >= 2:
        return True

    return False
