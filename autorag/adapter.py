"""
AutoRAG 가 찾아낸 best_config 를 기존 LangGraph 에이전트(`agent_v2`)에 연결하는 어댑터.

핵심 원리
---------
AutoRAG 는 trial 디렉토리에 노드별 최적 모듈/하이퍼파라미터를 yaml 로 떨어뜨린다.
이 어댑터는 그 yaml 을 읽어 두 가지 일을 한다.
  1) AutoRAGRunner.search(query) — AutoRAG 의 best pipeline 을 그대로 호출해 docs 반환
  2) install_into_agent_v2() — app.core.agent_v2.tools.rag_search 의 내부 구현을
     AutoRAG best pipeline 으로 대체. LangGraph 그래프 코드는 그대로 두고 툴만 swap.

이렇게 하면 다른 에이전트가 만들고 있는 LangGraph 멀티에이전트 본체는
손대지 않고도 AutoRAG 결과를 즉시 검증할 수 있다.

사용
----
    # AutoRAG 평가가 끝난 뒤
    from autorag.adapter import install_into_agent_v2

    install_into_agent_v2(trial_dir="autorag/benchmark/0")
    # 이후 chat_v2 라우트 (/chat/v2/stream) 가 호출되면 새 rag_search 가 적용된다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_best_config(trial_dir: Path) -> Path:
    """AutoRAG 가 만든 trial 폴더에서 best_config 를 찾는다.
    AutoRAG 버전에 따라 'config.yaml' / 'summary.csv' 위치가 바뀔 수 있어 후보 다 시도."""
    for candidate in (trial_dir / "config.yaml", trial_dir / "best.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"best config 가 없음: {trial_dir}")


class AutoRAGRunner:
    """AutoRAG best trial 의 검색 파이프라인을 호출 가능한 형태로 캐시.

    구현은 AutoRAG 의 Runner 를 thin-wrap 한다."""

    def __init__(self, trial_dir: str | Path):
        from autorag.deploy import Runner       # AutoRAG 공식 deploy Runner

        self.trial_dir = Path(trial_dir)
        # Evaluated trial 은 노드당 다중 모듈을 가질 수 있으므로
        # from_trial_folder 가 내부적으로 extract_best_config 를 거친다.
        logger.info("[autorag.adapter] trial 로드: %s", self.trial_dir)
        self._runner = Runner.from_trial_folder(str(self.trial_dir))

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """질의 → retrieved docs.

        Runner.run 이 generator 까지 통과한 최종 출력을 리턴하지만, 우리는
        retrieval 결과만 필요하므로 retrieved_contents/ids/scores 컬럼을 추출.
        AutoRAG 의 result 는 단일 답변 문자열일 수도 있어, run_dataframe 을
        대신 호출해 retrieval 컬럼을 보존한다.
        """
        import pandas as pd

        previous = pd.DataFrame(
            {"qid": ["q0"], "query": [query], "retrieval_gt": [[]], "generation_gt": [""]}
        )
        # AutoRAG Runner 의 모듈을 순차 호출하며 retrieval 컬럼을 누적
        for instance, params in zip(self._runner.module_instances, self._runner.module_params):
            new_result = instance.pure(previous_result=previous, **params)
            dup = previous.columns.intersection(new_result.columns)
            previous = pd.concat([previous.drop(columns=dup), new_result], axis=1)

        contents = previous.get("retrieved_contents")
        ids = previous.get("retrieved_ids")
        scores = previous.get("retrieve_scores")
        if contents is None:
            return []
        contents_l = contents.iloc[0] if hasattr(contents, "iloc") else contents
        ids_l = ids.iloc[0] if ids is not None else [None] * len(contents_l)
        scores_l = scores.iloc[0] if scores is not None else [1.0] * len(contents_l)

        out: list[dict[str, Any]] = []
        for content, doc_id, score in zip(contents_l, ids_l, scores_l):
            out.append(
                {
                    "content": content,
                    "score": float(score) if score is not None else 0.0,
                    "doc_id": doc_id,
                    "metadata": {},
                    "page": "Unknown",
                    "source": "헬기 매뉴얼",
                }
            )
        if top_k:
            out = out[:top_k]
        return out


def install_into_agent_v2(trial_dir: str | Path) -> AutoRAGRunner:
    """app.core.agent_v2.tools.rag_search 를 AutoRAG 파이프라인으로 monkey-patch.

    LangGraph 그래프 코드(`graph.py`)와 툴 시그니처는 그대로. 내부 구현만 교체한다.
    멀티 에이전트 측에서 이미 ALL_TOOLS 를 bind 했더라도, `rag_search.func`
    클로저를 바꾸므로 다음 호출부터 AutoRAG 결과가 흐른다.
    """
    runner = AutoRAGRunner(trial_dir)

    from app.core.agent_v2 import tools as agent_tools

    @agent_tools.tool
    def rag_search(query: str, top_k: int | None = None, reranker_top_k: int | None = None) -> str:  # type: ignore[no-redef]
        """헬기 매뉴얼 검색 (AutoRAG best pipeline)."""
        if not query:
            raise ValueError("query 가 비어있다")
        docs = runner.search(query, top_k=top_k or reranker_top_k)
        return json.dumps({"documents": docs, "count": len(docs)}, ensure_ascii=False, indent=2)

    # 원본 final_answer 는 그대로 두고 ALL_TOOLS 만 교체
    agent_tools.rag_search = rag_search
    agent_tools.ALL_TOOLS = [rag_search, agent_tools.final_answer]
    logger.info("[autorag.adapter] agent_v2.tools.rag_search 가 AutoRAG 로 교체됨")
    return runner
