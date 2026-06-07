"""
LangChain @tool 데코레이터로 래핑된 에이전트 도구.

기존 app/core/tools/rag_search.py 와 final_answer.py 의 동작을
LangChain Tool 인터페이스로 노출한다.

LangChain @tool 의 docstring + 타입 힌트가 자동으로 JSON Schema 로 변환되어
ChatHuggingFace.bind_tools() 와 Qwen apply_chat_template(tools=) 에 전달된다.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def rag_search(query: str) -> str:
    """헬리콥터 매뉴얼에서 사용자 질문과 관련된 문서를 검색한다.

    질문에 답변하기 전에 이 도구를 먼저 호출해야 한다.
    Multi-Query / Hybrid Retrieval 은 내부 로직이 자동 판단한다.

    Args:
        query: 검색용 핵심 명사 키워드. 2~5단어, 30자 이내. 조사·동사·문서형식 단어 제외.

    Returns:
        검색된 문서 본문 (문서별 `[문서 N · p.PAGE]` 헤더 + 내용). LLM 컨텍스트 예산
        (config.MAX_CONTEXT_LENGTH) 초과분은 잘림 — metadata JSON bloat 없이 본문만.

    Note:
        top_k / reranker_top_k 는 LLM 의 임의 변조를 막기 위해 도구 인자에서 제외 —
        항상 config.VECTOR_SEARCH_TOP_K / config.RERANKER_TOP_K 사용. LLM 이 결정할
        수 있는 건 query 문자열뿐. (이전엔 LLM 이 reranker_top_k 에 3 을 박아 검색
        결과가 절반 이하로 쪼그라들곤 했음.)
    """
    from app.services.rag_service import rag_search_with_rerank, simplify_query
    from app.services.multi_query import multi_query_retrieval, should_use_multi_query
    from app.config import config

    if not query:
        raise ValueError("query 파라미터가 비어있습니다.")

    # Multi-Query 자동 판단 (기존 rag_search.py 와 동일 로직, 단순화)
    if should_use_multi_query(query):
        logger.info(f"[agent_v2.rag_search] Multi-Query 모드: '{query}'")
        results = multi_query_retrieval(
            question=query,
            max_sub_queries=3,
            top_k_per_query=5,
            final_top_k=None,  # config 기본값 (= RERANKER_TOP_K)
        )
    else:
        simplified = simplify_query(query, max_length=30)
        if simplified != query:
            logger.info(f"[agent_v2.rag_search] Query 단순화: '{query}' -> '{simplified}'")
        results = rag_search_with_rerank(
            query=simplified,
            top_k=None,           # config 기본값 강제 (= VECTOR_SEARCH_TOP_K)
            reranker_top_k=None,  # config 기본값 강제 (= RERANKER_TOP_K)
        )

    if not results:
        return "검색 결과가 없습니다."

    # LLM 에게는 본문만 — metadata dict / JSON 들여쓰기 bloat 가 컨텍스트를 2배로 부풀려
    # context window 초과를 유발하므로 (page 헤더만 남기고) 본문만 전달, 예산 초과분은 절단.
    budget = getattr(config, "MAX_CONTEXT_LENGTH", 8000)
    parts, total = [], 0
    for i, d in enumerate(results, 1):
        content = (d.get("content") or "").strip()
        if not content:
            continue
        if total + len(content) > budget:
            content = content[: max(0, budget - total)].rstrip()
            if content:
                parts.append(f"[문서 {i} · p.{d.get('page', '?')}]\n{content}")
            logger.info(f"[agent_v2.rag_search] 컨텍스트 예산({budget}자) 도달 — {i}번째에서 절단")
            break
        parts.append(f"[문서 {i} · p.{d.get('page', '?')}]\n{content}")
        total += len(content)
    return "\n\n".join(parts) if parts else "검색 결과가 없습니다."


# bind_tools 에 넘길 도구 리스트. 답변은 free-text streaming 으로 처리하므로 final_answer 같은
# wrapper 도구는 두지 않는다 — LLM 이 호출할 도구는 rag_search 하나뿐.
ALL_TOOLS = [rag_search]
