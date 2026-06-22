"""
RAGSearch 도구

이 모듈은 벡터 DB에서 문서를 검색하는 도구를 정의합니다.
"""

from app.core.tools.base import Tool
from app.services.rag_service import rag_search_with_rerank, simplify_query
from app.services.multi_query import multi_query_retrieval, should_use_multi_query
from app.services.hybrid_retrieval import hybrid_retrieval
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class RAGSearchTool(Tool):
    """
    RAG 검색 도구

    벡터 DB에서 관련 문서를 검색하고 재순위화하여 반환합니다.

    Example:
        >>> tool = RAGSearchTool()
        >>> result = tool.execute(query="양력이란 무엇인가요?", top_k=10, reranker_top_k=3)
        >>> print(result)
        [TOOL_RESULT] {"documents": [...], "count": 3}
    """

    def execute(self, **kwargs) -> str:
        """
        RAG 검색을 수행합니다.

        Phase 3 Step 3: Multi-Query Retrieval 자동 적용
        Phase 3 Step 4: Hybrid Retrieval (Semantic + Keyword Fallback)

        Args:
            **kwargs: 도구 파라미터
                - query (str): 검색 쿼리
                - top_k (int, optional): 벡터 검색에서 가져올 문서 수 (기본값: config)
                - reranker_top_k (int, optional): 재순위화 후 선택할 문서 수 (기본값: config)
                - use_multi_query (bool, optional): Multi-Query 강제 사용 여부 (기본값: 자동 판단)
                - use_hybrid_retrieval (bool, optional): Hybrid Retrieval 사용 여부 (기본값: False)

        Returns:
            str: [TOOL_RESULT] 프리픽스가 붙은 검색 결과 (JSON 형식)

        Raises:
            ValueError: query 파라미터가 없는 경우
        """
        query = kwargs.get('query')
        if query is None:
            raise ValueError("query 파라미터가 필요합니다.")

        top_k = kwargs.get('top_k')
        reranker_top_k = kwargs.get('reranker_top_k')
        use_multi_query = kwargs.get('use_multi_query')  # 명시적 지정
        use_hybrid_retrieval = kwargs.get('use_hybrid_retrieval', False)  # Step 4: Hybrid Retrieval

        # Step 4: Hybrid Retrieval 모드
        if use_hybrid_retrieval:
            logger.info(f"Hybrid Retrieval 사용 (Semantic + Keyword Fallback): '{query}'")
            results = hybrid_retrieval(
                question=query,
                min_results_threshold=3,
                use_multi_query=use_multi_query,
                final_top_k=reranker_top_k
            )
            logger.info(f"Hybrid Retrieval 결과: {len(results)}개 문서")

        # Phase 3 Step 3: Multi-Query 사용 여부 결정
        elif use_multi_query is None:
            # 자동 판단
            use_multi_query = should_use_multi_query(query)

            if use_multi_query:
                # Multi-Query Retrieval 사용
                logger.info(f"복합 질문 감지, Multi-Query Retrieval 사용: '{query}'")
                results = multi_query_retrieval(
                    question=query,
                    max_sub_queries=3,
                    top_k_per_query=5,
                    final_top_k=reranker_top_k
                )
            else:
                # 단일 쿼리 사용 (Phase 3 Step 2: 쿼리 단순화 적용)
                logger.info(f"단순 질문, 단일 쿼리 사용: '{query}'")
                original_query = query
                query = simplify_query(query, max_length=30)

                if query != original_query:
                    logger.info(f"Query simplified: '{original_query}' -> '{query}'")

                # RAG 검색 실행
                results = rag_search_with_rerank(
                    query=query,
                    top_k=top_k,
                    reranker_top_k=reranker_top_k
                )

        elif use_multi_query:
            # 명시적으로 Multi-Query 지정
            logger.info(f"Multi-Query Retrieval 강제 사용: '{query}'")
            results = multi_query_retrieval(
                question=query,
                max_sub_queries=3,
                top_k_per_query=5,
                final_top_k=reranker_top_k
            )
        else:
            # 명시적으로 단일 쿼리 지정
            logger.info(f"단일 쿼리 강제 사용: '{query}'")
            original_query = query
            query = simplify_query(query, max_length=30)

            if query != original_query:
                logger.info(f"Query simplified: '{original_query}' -> '{query}'")

            # RAG 검색 실행
            results = rag_search_with_rerank(
                query=query,
                top_k=top_k,
                reranker_top_k=reranker_top_k
            )

        # 결과를 JSON 형식으로 포맷팅
        formatted_results = {
            "documents": results,
            "count": len(results)
        }

        # JSON 문자열로 변환
        result_json = json.dumps(formatted_results, ensure_ascii=False, indent=2)

        return f"[TOOL_RESULT] {result_json}"

    def get_schema(self) -> Dict[str, Any]:
        """
        도구의 스키마를 반환합니다.

        Returns:
            Dict[str, Any]: 도구 스키마
        """
        return {
            "name": "rag_search",
            "description": "벡터 데이터베이스에서 쿼리와 관련된 문서를 검색합니다. 검색된 문서는 재순위화되어 가장 관련성 높은 문서만 반환됩니다.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "검색할 쿼리 문자열"
                },
                "top_k": {
                    "type": "integer",
                    "description": "벡터 검색에서 가져올 문서 수 (선택사항)"
                },
                "reranker_top_k": {
                    "type": "integer",
                    "description": "재순위화 후 최종 선택할 문서 수 (선택사항)"
                }
            },
            "required": ["query"]
        }
