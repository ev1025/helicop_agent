import torch
import logging
import json
from typing import List, Tuple, Dict, Any
from app.config import config
from app.core import models
from app.utils import text_utils
from app.core.agent_v2.langfuse_handler import maybe_span, end_span
import time

logger = logging.getLogger(__name__)

def rerank_documents(query: str, documents: List[str], top_k: int = None) -> List[Tuple[str, float]]:
    """
    검색된 문서들을 쿼리와의 관련성에 따라 재순위화
    
    Args:
        query: 사용자 쿼리
        documents: 재순위화할 문서 리스트
        reranker_model: 재순위화 모델
        reranker_tokenizer: 재순위화 토크나이저
        device: 연산 디바이스
        top_k: 반환할 상위 문서 수
    
    Returns:
        List[Tuple[str, float]]: (문서, 관련도 점수) 튜플 리스트 (내림차순 정렬)
    
    Note:
        초기 검색 결과를 더 정교한 모델로 재평가하여 정확도 향상
    """
    # top_k 기본값 설정
    if top_k is None:
        top_k = config.RERANKER_TOP_K
    
    # 리랭킹할 문서 확인
    if not documents:
        return []
    
    device = models.get_device(config.USE_RERANKER_GPU)
    
    # 쿼리-문서 쌍 생성
    query_doc_pairs = [[query, doc] for doc in documents]
    
    # 토큰화
    features = models.reranker_tokenizer(
        query_doc_pairs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=512      # 재순위화는 상대적으로 짧은 길이 사용
    ).to(device)
    
    # 관련성 점수 계산
    models.reranker_model.eval()  # 평가 모드
    with torch.no_grad():
        logits = models.reranker_model(**features).logits
        scores = torch.sigmoid(logits).squeeze().cpu().tolist()     # 0-1 사이 점수로 변환
    
    # 단일 문서인 경우 리스트로 변환
    if isinstance(scores, float):
        scores = [scores]
    
    # (문서, 점수) 쌍 생성 후 점수 내림차순 정렬
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return doc_score_pairs[:top_k]


def rag_search_with_rerank(query: str, top_k: int = None, reranker_top_k: int = None,
                           sentence_window: int = 1):
    """
    RAG 검색 + Re-rank 파이프라인 (벡터 검색 -> 재순위화 -> 컨텍스트 구성)

    Parent-Child Retrieval 지원:
    - Parent store가 있으면: Child로 검색 → Parent 반환 (정밀 검색 + 풍부한 컨텍스트)
    - Parent store가 없으면: 일반 검색 (기존 로직)

    Args:
        query: 사용자 질문
        top_k: 벡터 검색에서 가져올 문서 수
        reranker_top_k: 재순위화 후 최종 선택할 문서 수
        sentence_window: 매칭 Parent 의 ±N 인접도 자동 회수 (기본 1, 0=미사용).
                         CHANGES.md #005 결과 — window=1 무비용으로 평균 1.4개 인접
                         Parent 추가 회수, 응집도 향상. 0 으로 명시하면 비활성화.

    Returns:
        list: 선택된 문서 정보 리스트 (내용, 점수, 메타데이터 포함)

    Note:
        1단계: 벡터 유사도로 후보 문서 검색
        2단계: 재순위화 모델로 정밀 평가
        3단계: 고품질 문서만 선별하여 컨텍스트 구성
    """
    logger.info("벡터 검색 시작")
    # top_k 기본값 지정
    if top_k is None:
        top_k = config.VECTOR_SEARCH_TOP_K
    if reranker_top_k is None:
        reranker_top_k = config.RERANKER_TOP_K

    try:
        if not models.collection:
            logger.warning("Collection이 초기화되지 않았습니다")
            return []

        # Parent-Child 모드 확인
        use_parent_child = models.parent_store is not None and len(models.parent_store) > 0

        if use_parent_child:
            logger.info("Parent-Child Retrieval 모드로 검색")
            results = _rag_search_parent_child(query, top_k, reranker_top_k)
        else:
            logger.info("일반 RAG 검색 모드")
            results = _rag_search_normal(query, top_k, reranker_top_k)

        # Sentence-Window: 매칭 Parent 의 ±N 인접도 회수 (선택) — CHANGES.md #4
        if sentence_window > 0 and results:
            _n_before = len(results)
            _s_sw = maybe_span("rag.sentence_window",
                               input={"n_before": _n_before, "window": sentence_window})
            results = _expand_with_adjacent_parents(results, window=sentence_window)
            end_span(_s_sw, output={"n_after": len(results), "added": len(results) - _n_before,
                                    "pages": [r.get("page") for r in results]})

        return results

    except Exception as e:
        logger.error(f"RAG 검색 오류: {e}")
        return []


def _rag_search_normal(query: str, top_k: int, reranker_top_k: int):
    """
    일반 RAG 검색 (Parent-Child 미사용)

    Args:
        query: 사용자 질문
        top_k: 벡터 검색에서 가져올 문서 수
        reranker_top_k: 재순위화 후 최종 선택할 문서 수

    Returns:
        list: 선택된 문서 정보 리스트
    """
    # 1단계: Vector 검색
    start = time.perf_counter()
    search_results = models.collection.similarity_search(query, k=top_k * 2)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"--> 벡터검색 수행시간: {elapsed:.1f}ms")

    if not search_results:
        logger.info("Vector 검색 결과: 없음")
        return []
    else:
        logger.info(f"--> Vector 검색 결과: {len(search_results)}개 문서")

    # 2단계: Re-rank (reranker 활성화 시에만)
    documents = [doc.page_content for doc in search_results]

    if config.USE_RERANKER and models.reranker_model is not None:
        start = time.perf_counter()
        reranked_results = rerank_documents(query, documents, reranker_top_k)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"--> 리랭킹 수행시간: {elapsed:.1f}ms")
    else:
        logger.info("--> 리랭킹 미사용 (vector 검색 결과만 사용)")
        reranked_results = [(doc, 1.0) for doc in documents[:reranker_top_k]]

    # 3단계: 문서 선별 및 컨텍스트 구성
    start = time.perf_counter()
    final_results = []
    total_length = 0

    for doc, score in reranked_results:
        if score >= config.RERANKER_SCORE_THRESHOLD:  # 유사도 임계값 이상 문서 선별
            if total_length + len(doc) > config.MAX_CONTEXT_LENGTH:
                remaining_length = config.MAX_CONTEXT_LENGTH - total_length
                if remaining_length > config.CONTEXT_MIN_REMAINING_LENGTH:
                    doc = doc[:remaining_length] + "..."
                else:
                    break

            # 원본 메타데이터 찾기
            original_doc = None
            for result in search_results:
                if result.page_content.startswith(doc[:100]):   # 앞부분 100자로 매칭
                    original_doc = result
                    break

            # 문서 정보 구성
            doc_info = {
                'content': doc,
                'score': score,
                'metadata': original_doc.metadata if original_doc else {},
                'page': original_doc.metadata.get('page', 'Unknown') if original_doc else 'Unknown',
                'source': original_doc.metadata.get('source', '헬리콥터 교재') if original_doc else '헬리콥터 교재'
            }
            final_results.append(doc_info)
            total_length += len(doc)

            # 최대 길이 도달 시, 중단
            if total_length >= config.MAX_CONTEXT_LENGTH:
                break

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"컨텍스트 구성 수행시간: {elapsed:.1f}ms")
    logger.info(f"--> 최종 RAG 결과: {len(final_results)}개 문서, 총 길이: {total_length}")
    final_results = _sort_by_document_order(final_results)
    return final_results


def _sort_by_document_order(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """회수된 문서를 원본 본문 순서(page → parent_chunk_id)로 정렬.

    유사도 점수 순으로 LLM 에 넘기면 본문 흐름이 깨져 약한 LLM 이 답변 흐름을
    잘못 잡을 수 있다 (분석정리.md 9.2/약점1 참조). 페이지가 비교 가능한
    값이면 page → parent_chunk_id 순으로 정렬, 둘 다 없으면 원본 순서 유지.
    """
    def _key(item):
        page = item.get('page', 'Unknown')
        # 'Unknown' 같은 값은 맨 뒤로 보냄
        page_sort = (1, 0) if not isinstance(page, (int, float)) else (0, page)
        chunk_id = item.get('metadata', {}).get('parent_chunk_id', 0)
        return (page_sort, chunk_id)
    return sorted(results, key=_key)


def _expand_with_adjacent_parents(
    results: List[Dict[str, Any]],
    window: int = 1,
    max_total_length: int = None,
) -> List[Dict[str, Any]]:
    """매칭된 Parent 의 ±window 인접 Parent 를 자동으로 추가 회수 (Sentence-Window).

    분석정리.md §6 약점 2 / §10.5 / CHANGES.md #005 참조.
    같은 주제가 여러 Parent 로 쪼개진 상황에서 일부만 검색되고 인접 Parent 가
    누락되어 답변이 단편적이 되는 문제를 완화한다.

    Args:
        results: 본문 순서 정렬까지 끝난 검색 결과 (각 항목에 metadata.parent_chunk_id 필요).
        window: 매칭 Parent 의 앞뒤 N 개 인접을 추가 (예: 1 → 매칭 #42 면 #41, #43 추가).
        max_total_length: 컨텍스트 총 길이 한도. 초과 시 추가 중단. None 이면 config.MAX_CONTEXT_LENGTH.

    Returns:
        확장된 결과 (본문 순서로 다시 정렬됨). window<=0 이면 입력 그대로.
    """
    if window <= 0 or not results:
        return results
    if not models.parent_store:
        return results

    if max_total_length is None:
        max_total_length = config.MAX_CONTEXT_LENGTH

    matched_chunk_ids = set()
    for r in results:
        cid = r.get('metadata', {}).get('parent_chunk_id')
        if isinstance(cid, int):
            matched_chunk_ids.add(cid)

    if not matched_chunk_ids:
        return results

    # ±window 확장
    extended_chunk_ids = set()
    for cid in matched_chunk_ids:
        for offset in range(-window, window + 1):
            extended_chunk_ids.add(cid + offset)

    new_chunk_ids = extended_chunk_ids - matched_chunk_ids
    if not new_chunk_ids:
        return results

    total_len = sum(len(r.get('content', '')) for r in results)
    added = []
    for cid in sorted(new_chunk_ids):
        parent_key = f"parent_{cid}"
        if parent_key not in models.parent_store:
            continue  # 인덱스 범위 밖
        parent_data = models.parent_store[parent_key]
        content = parent_data.get('content', '')
        if total_len + len(content) > max_total_length:
            continue  # 컨텍스트 한도 초과면 추가 안 함
        meta = parent_data.get('metadata', {})
        added.append({
            'content': content,
            'score': 0.0,  # 인접 회수는 점수 없음 (마커)
            'metadata': meta,
            'page': meta.get('page', 'Unknown'),
            'source': meta.get('source', '헬리콥터 교재'),
            'parent_id': parent_key,
            '_adjacent': True,  # 인접 회수 여부 마커
        })
        total_len += len(content)

    if added:
        logger.info(f"--> Sentence-Window(window={window}): {len(added)}개 인접 Parent 추가")
        return _sort_by_document_order(results + added)
    return results


def _rag_search_parent_child(query: str, top_k: int, reranker_top_k: int):
    """
    Parent-Child Retrieval 검색

    1. Child chunk로 벡터 검색 (정밀 검색)
    2. Child에서 parent_id 추출
    3. Parent chunk 가져오기
    4. Parent chunk rerank
    5. Parent chunk 반환 (풍부한 컨텍스트)

    Args:
        query: 사용자 질문
        top_k: Child 벡터 검색에서 가져올 문서 수
        reranker_top_k: Parent rerank 후 최종 선택할 문서 수

    Returns:
        list: 선택된 Parent 문서 정보 리스트
    """
    # ── Langfuse: child 벡터검색 → parent 확대 단계 (CHANGES.md #1 Parent-Child) ──
    _s_cp = maybe_span("rag.child_to_parent", input={"query": query, "child_k": top_k * 2})

    # 1단계: Child chunk로 벡터 검색
    start = time.perf_counter()
    child_search_results = models.collection.similarity_search(query, k=top_k * 2)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"--> Child 벡터검색 수행시간: {elapsed:.1f}ms")

    if not child_search_results:
        logger.info("Child 검색 결과: 없음")
        end_span(_s_cp, output={"n_child": 0, "n_parent": 0})
        return []

    logger.info(f"--> Child 검색 결과: {len(child_search_results)}개 문서")

    # 2단계: Child에서 parent_id 추출 및 Parent chunk 가져오기
    start = time.perf_counter()
    parent_docs = []
    seen_parent_ids = set()

    for child_doc in child_search_results:
        parent_id = child_doc.metadata.get('parent_id')
        if not parent_id:
            logger.warning(f"parent_id가 없는 Child chunk 발견: {child_doc.metadata}")
            continue

        if parent_id in seen_parent_ids:
            continue  # 중복 제거

        if parent_id in models.parent_store:
            parent_data = models.parent_store[parent_id]
            parent_docs.append({
                'content': parent_data['content'],
                'metadata': parent_data['metadata'],
                'parent_id': parent_id
            })
            seen_parent_ids.add(parent_id)
        else:
            logger.warning(f"parent_store에서 {parent_id}를 찾을 수 없음")

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"--> Parent 문서 가져오기: {len(parent_docs)}개 ({elapsed:.1f}ms)")
    # Langfuse trace 에 child / parent 청크 본문 미리보기 노출 — 검색 품질 디버깅용.
    _child_previews = [
        {
            "page": d.metadata.get("page"),
            "parent_id": d.metadata.get("parent_id"),
            "preview": (d.page_content or "")[:120].replace("\n", " "),
        }
        for d in child_search_results
    ]
    _parent_previews = [
        {
            "page": d["metadata"].get("page"),
            "parent_id": d["parent_id"],
            "preview": (d["content"] or "")[:200].replace("\n", " "),
        }
        for d in parent_docs
    ]
    end_span(_s_cp, output={"n_child": len(child_search_results), "n_parent": len(parent_docs),
                            "parent_pages": [d["metadata"].get("page") for d in parent_docs],
                            "child_chunks": _child_previews,
                            "parent_chunks": _parent_previews})

    if not parent_docs:
        logger.warning("Parent 문서를 가져올 수 없습니다")
        return []

    # 3단계: Parent chunk rerank (reranker 활성화 시에만) — CHANGES.md #3 top_k 튜닝
    _reranker_on = bool(config.USE_RERANKER and models.reranker_model is not None)
    _s_rr = maybe_span("rag.parent_rerank",
                       input={"n_parent": len(parent_docs), "reranker_on": _reranker_on,
                              "reranker_top_k": reranker_top_k})
    parent_contents = [doc['content'] for doc in parent_docs]

    if _reranker_on:
        start = time.perf_counter()
        reranked_parents = rerank_documents(query, parent_contents, reranker_top_k)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"--> Parent 리랭킹 수행시간: {elapsed:.1f}ms")
    else:
        logger.info("--> Parent 리랭킹 미사용 (vector 검색 결과만 사용)")
        reranked_parents = [(content, 1.0) for content in parent_contents[:reranker_top_k]]
    end_span(_s_rr, output={
        "n_after": len(reranked_parents),
        "top_scores": [round(float(s), 3) for _, s in reranked_parents[:5]],
        # 최종 선별된 parent 청크 본문 미리보기 (검색 품질 디버깅용)
        "selected_chunks": [
            {
                "rank": i + 1,
                "score": round(float(s), 3),
                "preview": (c or "")[:200].replace("\n", " "),
            }
            for i, (c, s) in enumerate(reranked_parents)
        ],
    })

    # 4단계: 최종 문서 선별 및 컨텍스트 구성
    start = time.perf_counter()
    final_results = []
    total_length = 0

    for parent_content, score in reranked_parents:
        if score >= config.RERANKER_SCORE_THRESHOLD:
            # 원본 Parent 문서 찾기
            original_parent = None
            for pdoc in parent_docs:
                if pdoc['content'] == parent_content:
                    original_parent = pdoc
                    break

            if not original_parent:
                continue

            # 길이 제한 체크
            if total_length + len(parent_content) > config.MAX_CONTEXT_LENGTH:
                remaining_length = config.MAX_CONTEXT_LENGTH - total_length
                if remaining_length > config.CONTEXT_MIN_REMAINING_LENGTH:
                    parent_content = parent_content[:remaining_length] + "..."
                else:
                    break

            # 문서 정보 구성
            metadata = original_parent['metadata']
            doc_info = {
                'content': parent_content,
                'score': score,
                'metadata': metadata,
                'page': metadata.get('page', 'Unknown'),
                'source': metadata.get('source', '헬리콥터 교재'),
                'parent_id': original_parent['parent_id']
            }
            final_results.append(doc_info)
            total_length += len(parent_content)

            # 최대 길이 도달 시, 중단
            if total_length >= config.MAX_CONTEXT_LENGTH:
                break

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"Parent 컨텍스트 구성 수행시간: {elapsed:.1f}ms")
    logger.info(f"--> 최종 Parent RAG 결과: {len(final_results)}개 문서, 총 길이: {total_length}")
    final_results = _sort_by_document_order(final_results)
    pages_after = [r.get('page') for r in final_results]
    logger.info(f"--> 본문 순서 정렬 후 페이지: {pages_after}")
    return final_results


def format_rag_results(rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    RAG 검색 결과를 포맷팅하여 문서 요약 및 컨텍스트 생성

    Args:
        rag_results: RAG 검색 결과 리스트

    Returns:
        dict: 포맷팅된 결과
            - use_rag: RAG 사용 여부
            - doc_summaries: 문서 요약 리스트
            - detailed_info: 문서 상세 정보 문자열
            - rag_context: LLM에 전달할 컨텍스트
    """
    use_rag = len(rag_results) > 0

    if not use_rag:
        return {
            'use_rag': False,
            'doc_summaries': [],
            'detailed_info': '',
            'rag_context': ''
        }

    # 문서 요약 생성
    doc_summaries = []
    for i, result in enumerate(rag_results):
        page_info = f"페이지 {result['page']}" if result['page'] != 'Unknown' else "페이지 정보 없음"
        score_info = f"관련도: {result['score']:.3f}"
        source_name = result['source'].split('/')[-1].replace('.pdf', '') if text_utils.safe_string_check(result['source'], '/') else result['source']
        doc_summaries.append(f"• {source_name} ({page_info}, {score_info})")

    detailed_info = "\n".join(doc_summaries)

    # RAG 컨텍스트 구성
    rag_context = "\n\n".join([
        f"[문서 {i+1} - {result['source']} 페이지 {result['page']}] {result['content']}"
        for i, result in enumerate(rag_results)
    ])

    return {
        'use_rag': True,
        'doc_summaries': doc_summaries,
        'detailed_info': detailed_info,
        'rag_context': rag_context,
        'doc_count': len(rag_results)
    }


def generate_rag_sse_events(rag_results: List[Dict[str, Any]]):
    """
    RAG 검색 결과를 Server-Sent Events 형식으로 변환

    Args:
        rag_results: RAG 검색 결과 리스트

    Yields:
        str: SSE 형식의 이벤트 문자열
    """
    formatted = format_rag_results(rag_results)

    if formatted['use_rag']:
        # RAG 결과 있음
        doc_count = formatted['doc_count']
        yield f"data: {json.dumps({'type': 'rag_info', 'content': f'✅ 관련 문서 {doc_count}개를 찾았습니다:', 'doc_count': doc_count}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'doc_details', 'content': formatted['detailed_info'], 'documents': formatted['doc_summaries']}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'info', 'content': '📖 위 문서들을 바탕으로 답변을 생성합니다.'}, ensure_ascii=False)}\n\n"
    else:
        # RAG 결과 없음
        yield f"data: {json.dumps({'type': 'rag_info', 'content': '❌ 관련 문서를 찾을 수 없습니다. 사전학습된 지식으로 답변을 생성합니다.', 'doc_count': 0}, ensure_ascii=False)}\n\n"


def simplify_query(query: str, max_length: int = 30) -> str:
    """
    검색 쿼리를 간결하게 만듦 (Phase 3 Step 2)

    Args:
        query: 원본 쿼리
        max_length: 최대 길이 (기본값: 30자)

    Returns:
        str: 단순화된 쿼리

    Note:
        - 조사 제거
        - 불필요한 단어 제거
        - 핵심 키워드만 추출
        - 길이 제한

    Example:
        >>> simplify_query("오토자이로와 헬리콥터의 로터 구동 방식 차이 및 오토자이로의 헬리콥터 발전에 대한 기여")
        "오토자이로 헬리콥터 로터 구동 차이"
    """
    import re

    # 이미 짧으면 그대로 반환
    if len(query) <= max_length:
        return query

    # 1단계: 불필요한 단어 패턴 제거
    unnecessary_patterns = [
        r'에\s+대한',
        r'에\s+대해',
        r'에\s+관한',
        r'에\s+관해',
        r'를\s+통해',
        r'를\s+통한',
        r'에\s+어떻게',
        r'는\s+무엇',
        r'이\s+무엇',
        r'의\s+기여',
        r'의\s+역할',
        r'을\s+설명',
        r'를\s+설명',
        r'해\s+주',
        r'주십시오',
        r'주세요',
        r'말해줘',
        r'알려줘',
    ]

    simplified = query
    for pattern in unnecessary_patterns:
        simplified = re.sub(pattern, ' ', simplified, flags=re.IGNORECASE)

    # 2단계: 연속된 공백 제거
    simplified = re.sub(r'\s+', ' ', simplified).strip()

    # 3단계: 여전히 길면 첫 N자만 추출 (단어 경계에서)
    if len(simplified) > max_length:
        # 공백 기준으로 단어 분리
        words = simplified.split()
        result = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_length:
                result.append(word)
                current_length += len(word) + 1
            else:
                break

        simplified = ' '.join(result)

    return simplified


def get_all_documents_from_db() -> List[Dict[str, Any]]:
    """
    벡터 DB에서 모든 문서를 가져옵니다 (Keyword 검색용)

    Returns:
        List[Dict[str, Any]]: 모든 문서 리스트 (각 문서는 'content', 'metadata' 포함)

    Note:
        - Parent-Child 모드: Parent Store에서 모든 부모 문서 반환
        - Normal 모드: Collection에서 모든 문서 반환
    """
    try:
        # Parent-Child 모드 확인
        if config.USE_PARENT_DOCUMENT_RETRIEVAL and models.parent_store:
            logger.info(f"Parent Store에서 {len(models.parent_store)}개 문서 로드")
            documents = []

            for parent_id, parent_data in models.parent_store.items():
                doc = {
                    'content': parent_data['content'],
                    'metadata': parent_data.get('metadata', {}),
                    'parent_id': parent_id
                }
                documents.append(doc)

            return documents

        # Normal 모드: Collection에서 가져오기
        if not models.collection:
            logger.error("Collection이 초기화되지 않았습니다")
            return []

        # Chroma collection.get()으로 모든 문서 가져오기
        result = models.collection.get(
            include=['documents', 'metadatas']
        )

        documents = []
        if result and 'documents' in result:
            doc_contents = result['documents']
            doc_metadatas = result.get('metadatas', [{}] * len(doc_contents))

            for i, content in enumerate(doc_contents):
                doc = {
                    'content': content,
                    'metadata': doc_metadatas[i] if i < len(doc_metadatas) else {}
                }
                documents.append(doc)

            logger.info(f"Collection에서 {len(documents)}개 문서 로드")
            return documents

        logger.warning("DB에서 문서를 가져올 수 없습니다")
        return []

    except Exception as e:
        logger.error(f"DB 문서 로드 오류: {e}")
        return []