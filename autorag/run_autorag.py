"""
AutoRAG Evaluator 실행 진입점.

전제
----
- vLLM 이 OpenAI 호환 API 로 떠 있어야 한다 (config.yaml 헤더 참고).
- ./data/corpus.parquet, ./data/qa.parquet 가 있어야 한다 (build_dataset.py).

사용
----
    python run_autorag.py            # 평가 + 최적 trial 폴더 생성
    python run_autorag.py --restart  # 직전 trial 이어서 진행

결과
----
- ./benchmark/<trial>/  : 노드별 best 모듈 + best_config.yaml
  → adapter.py 가 best_config.yaml 을 읽어 LangGraph rag_search 에 적용
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from autorag.evaluator import Evaluator        # AutoRAG 가 설치돼 있어야 한다

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
QA = ROOT / "data" / "qa.parquet"
CORPUS = ROOT / "data" / "corpus.parquet"
TRIALS = ROOT / "benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--config", default=str(CONFIG), help="AutoRAG yaml config path")
    parser.add_argument("--project-dir", default=str(TRIALS), help="trial 결과 디렉토리")
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

    Path(args.project_dir).mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(
        qa_data_path=str(QA),
        corpus_data_path=str(CORPUS),
        project_dir=args.project_dir,
    )
    if args.restart:
        evaluator.restart_trial()
    else:
        evaluator.start_trial(args.config)


if __name__ == "__main__":
    main()
