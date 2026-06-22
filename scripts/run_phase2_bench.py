#!/usr/bin/env python3
"""
Phase 2 Orchestrator 벤치마크 스크립트

테스트 질문 JSON을 기반으로 Phase 2 Orchestrator를 실행하고
LLM 기반 도구 선택 결과를 평가합니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import re

# 프로젝트 루트 경로를 PYTHONPATH에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.orchestrator import Orchestrator  # noqa: E402
from app.core.tool_executor import ToolExecutor  # noqa: E402
from app.core.tools import FinalAnswerTool, RAGSearchTool  # noqa: E402
from app.core import models  # noqa: E402


DEFAULT_CASES_PATH = ROOT_DIR / "tests" / "rag_cases.json"
RESULTS_DIR = ROOT_DIR / "results"
BENCH_PREFIX = "phase2_bench_"
BENCH_PADDING = 4


def setup_logging(verbose: bool) -> None:
    """콘솔 로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def load_cases(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """테스트 질문 케이스 로드"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
    elif isinstance(data, list):
        cases = data
    else:
        raise ValueError("JSON 구조가 올바르지 않습니다. 'cases' 배열 또는 리스트가 필요합니다.")

    if limit is not None:
        cases = cases[:limit]

    return cases


def sanitize_tokens(text: str) -> List[str]:
    """간단한 토큰화 도움 함수"""
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    return [token for token in tokens if len(token) > 1]


def compute_metrics(answer: str, ideal: str) -> Dict[str, Any]:
    """
    평가 메트릭 계산

    Args:
        answer: 생성된 답변
        ideal: 이상적 답변 (참조)

    Returns:
        메트릭 딕셔너리 (keyword_overlap, precision, recall, f1_score 등)
    """
    if not ideal:
        return {
            "answer_length": len(answer),
            "ideal_length": 0,
            "keyword_overlap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    answer_tokens = set(sanitize_tokens(answer))
    ideal_tokens = set(sanitize_tokens(ideal))

    if not ideal_tokens:
        overlap = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        common = answer_tokens & ideal_tokens
        overlap = len(common) / len(ideal_tokens)

        # Precision: 답변에서 언급한 것 중 이상적 답변에 있는 비율
        precision = len(common) / len(answer_tokens) if answer_tokens else 0.0

        # Recall: 이상적 답변의 내용을 얼마나 커버하는지
        recall = len(common) / len(ideal_tokens)

        # F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

    return {
        "answer_length": len(answer),
        "ideal_length": len(ideal),
        "keyword_overlap": round(overlap, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def get_actual_model_info() -> Dict[str, Any]:
    """실제 로드된 모델 정보 확인 (사용 직전 검증용)"""
    def _name(obj):
        try:
            if obj is None:
                return 'Not loaded'
            if hasattr(obj, 'config') and hasattr(obj.config, 'name_or_path'):
                return obj.config.name_or_path
            return str(type(obj))
        except Exception:
            return 'Error checking'

    from app.config import config
    model_info = {
        'config_llm_model': config.LLM_MODEL,
        'config_embedding_model': config.EMBEDDING_MODEL,
        'config_reranker_model': config.RERANKER_MODEL,
        'actual_llm_model': _name(models.model),
        'actual_embedding_model': _name(models.embedding_model),
        'actual_reranker_model': _name(models.reranker_model),
    }
    return model_info


def log_model_info(model_info: Dict[str, Any], prefix: str = "") -> None:
    """모델 정보 로그/프린트 (노트북 가시성 확보)"""
    separator = "=" * 60
    msg = [
        separator,
        f"{prefix}Config vs Actual Models (JUST-IN-TIME):",
        "-" * 60,
        f"Config LLM: {model_info['config_llm_model']}",
        f"Actual LLM: {model_info['actual_llm_model']}",
        "-" * 60,
        f"Config Embedding: {model_info['config_embedding_model']}",
        f"Actual Embedding: {model_info['actual_embedding_model']}",
        "-" * 60,
        f"Config Reranker: {model_info['config_reranker_model']}",
        f"Actual Reranker: {model_info['actual_reranker_model']}",
        separator
    ]

    # print만 사용 (logging.info와 중복 출력 방지, 로깅 시스템 간섭 방지)
    for line in msg:
        print(line)


def initialize_models() -> Dict[str, Any]:
    """필요한 모델 로드 및 실제 로드된 모델 정보 반환"""
    logging.info("모델 로딩 시작...")
    models.load_all_models()
    logging.info("모델 로딩 완료")

    # 실제 로드된 모델 정보 확인 (사용 직전)
    model_info = get_actual_model_info()
    log_model_info(model_info)

    return model_info


def initialize_orchestrator() -> Orchestrator:
    """Orchestrator 초기화 및 도구 등록"""
    executor = ToolExecutor()
    executor.register_tool(FinalAnswerTool())
    executor.register_tool(RAGSearchTool())

    orchestrator = Orchestrator(executor)
    logging.info(f"Orchestrator 초기화 완료: {orchestrator}")
    return orchestrator


def run_single_case(
    orchestrator: Orchestrator,
    case: Dict[str, Any],
    use_llm: bool,
    max_turns: int
) -> Dict[str, Any]:
    """
    단일 테스트 케이스 실행

    Args:
        orchestrator: Orchestrator 인스턴스
        case: 테스트 케이스 딕셔너리
        use_llm: LLM 사용 여부 (True: Phase 2, False: Phase 1)
        max_turns: 최대 턴 수

    Returns:
        결과 딕셔너리
    """
    case_id = case.get("id") or f"case_{int(time.time()*1000)}"
    question = case.get("question", "")
    ideal_answer = case.get("ideal_answer", "")
    meta = case.get("meta", {})

    if not question:
        raise ValueError(f"{case_id}: 'question' 필드가 비어 있습니다.")

    # Orchestrator 리셋 (깨끗한 상태)
    orchestrator.reset()

    timings: Dict[str, float] = {}

    # Orchestrator 실행
    start = time.perf_counter()
    try:
        answer = orchestrator.run(question, use_llm=use_llm, max_turns=max_turns)
        success = True
        error = None
    except Exception as e:
        answer = ""
        success = False
        error = str(e)
        logging.warning(f"[{case_id}] 실행 중 오류: {error}")

    timings["total_ms"] = round((time.perf_counter() - start) * 1000, 2)

    # 메트릭 계산
    metrics = compute_metrics(answer, ideal_answer)

    # 대화 정보 수집
    turn_count = orchestrator.get_turn_count()
    message_count = len(orchestrator.get_conversation().messages)

    # 대화 히스토리 추출 (중간 결과 저장)
    conversation = orchestrator.get_conversation()
    conversation_history = []
    tool_calls_summary = []

    for msg in conversation.messages:
        msg_detail = {
            "role": msg.role,
            "content": msg.content,
        }

        # 도구 호출 정보 추가
        if msg.tool_calls:
            msg_detail["tool_calls"] = [
                {
                    "name": tc.name,
                    "arguments": tc.arguments
                }
                for tc in msg.tool_calls
            ]

            # 도구 호출 요약 (분석용)
            for tc in msg.tool_calls:
                tool_calls_summary.append({
                    "tool": tc.name,
                    "arguments": tc.arguments
                })

        conversation_history.append(msg_detail)

    case_result: Dict[str, Any] = {
        "id": case_id,
        "question": question,
        "ideal_answer": ideal_answer,
        "meta": meta,
        "phase": "Phase 2 (LLM)" if use_llm else "Phase 1 (Hardcoded)",
        "use_llm": use_llm,
        "max_turns": max_turns,
        "answer": answer,
        "success": success,
        "error": error,
        "turn_count": turn_count,
        "message_count": message_count,
        "conversation_history": conversation_history,  # 전체 대화 히스토리
        "tool_calls_summary": tool_calls_summary,      # 도구 호출 요약
        "metrics": metrics,
        "timings": timings,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    return case_result


def _ensure_results_dir() -> Path:
    """결과 디렉토리 생성"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _list_bench_files(results_dir: Path) -> List[Path]:
    """기존 벤치마크 파일 목록"""
    return sorted(results_dir.glob(f"{BENCH_PREFIX}*.json"))


def get_next_bench_path(results_dir: Path | None = None) -> Path:
    """다음 벤치마크 파일 경로 결정"""
    directory = results_dir or _ensure_results_dir()
    existing = _list_bench_files(directory)

    if not existing:
        next_index = 0
    else:
        max_index = 0
        for file_path in existing:
            stem = file_path.stem
            try:
                num_str = stem.replace(BENCH_PREFIX, "")
                num = int(num_str)
                max_index = max(max_index, num)
            except (ValueError, AttributeError):
                continue
        next_index = max_index + 1

    filename = f"{BENCH_PREFIX}{next_index:0{BENCH_PADDING}d}.json"
    return directory / filename


def save_result(path: Path, payload: Dict[str, Any]) -> None:
    """결과를 JSON 파일로 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_bench(
    cases_path: Path = DEFAULT_CASES_PATH,
    output_path: Path | None = None,
    limit: int | None = None,
    use_llm: bool = True,
    max_turns: int = 10,
    verbose: bool = False,
) -> Path:
    """
    벤치마크를 실행하고 결과 파일 경로를 반환

    Args:
        cases_path: 테스트 케이스 JSON 경로
        output_path: 결과 저장 경로 (None이면 자동 생성)
        limit: 실행할 케이스 수 제한
        use_llm: LLM 사용 여부 (True: Phase 2, False: Phase 1)
        max_turns: 최대 턴 수
        verbose: 상세 로그 출력

    Returns:
        결과 파일 경로
    """
    setup_logging(verbose)

    # 🔥 CRITICAL: Config 파일이 변경된 경우를 대비해 모듈 reload
    # config를 참조하는 모든 서비스 모듈도 함께 reload 해야 함
    import importlib
    import app.config
    import app.services.rag_service
    import app.services.multi_query
    import app.services.hybrid_retrieval

    importlib.reload(app.config)
    importlib.reload(app.services.rag_service)      # config.USE_RERANKER 참조
    importlib.reload(app.services.multi_query)       # config 설정 참조
    importlib.reload(app.services.hybrid_retrieval)  # config 설정 참조
    logging.info("Config 및 서비스 모듈 reloaded")

    logging.info(f"테스트 케이스 로드: {cases_path}")
    cases = load_cases(cases_path, limit)
    if not cases:
        raise ValueError("실행할 테스트 케이스가 없습니다.")

    logging.info(f"총 {len(cases)}개 케이스 실행 예정")

    # 모델 및 Orchestrator 초기화
    model_info = initialize_models()  # 실제 로드된 모델 정보 받기
    orchestrator = initialize_orchestrator()
    last_logged_model_info: Dict[str, Any] | None = None

    start_ts = datetime.utcnow()
    run_id = start_ts.strftime("phase2-bench-%Y%m%dT%H%M%SZ")

    bench_result: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": start_ts.isoformat() + "Z",
        "cases_file": str(cases_path),
        "use_llm": use_llm,
        "max_turns": max_turns,
        "total_cases": len(cases),
        "model_info": model_info,  # 실제 로드된 모델 정보 추가
        "cases": [],
    }

    # 각 케이스 실행
    success_count = 0
    for idx, case in enumerate(cases, start=1):
        case_id = case.get("id", f"case_{idx}")
        logging.info(f"[{idx}/{len(cases)}] {case_id} 실행 시작...")

        # Q0에서는 무조건, 이후에는 변경 시에만 모델 정보 출력
        case_model_info = get_actual_model_info()
        if last_logged_model_info is None or case_model_info != last_logged_model_info:
            log_model_info(case_model_info, prefix=f"[{case_id}] ")
            last_logged_model_info = case_model_info

        try:
            result = run_single_case(orchestrator, case, use_llm, max_turns)
            bench_result["cases"].append(result)

            if result["success"]:
                success_count += 1

            logging.info(
                f"[{case_id}] 완료 (turns={result['turn_count']}, "
                f"time={result['timings']['total_ms']}ms, "
                f"f1={result['metrics']['f1_score']:.3f})"
            )
        except Exception as exc:
            logging.exception(f"[{case_id}] 테스트 중 오류 발생: {exc}")
            bench_result["cases"].append({
                "id": case_id,
                "question": case.get("question", ""),
                "ideal_answer": case.get("ideal_answer", ""),
                "meta": case.get("meta", {}),
                "error": str(exc),
                "success": False,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

    finish_ts = datetime.utcnow()
    bench_result["finished_at"] = finish_ts.isoformat() + "Z"
    bench_result["duration_ms"] = round(
        (finish_ts - start_ts).total_seconds() * 1000, 2
    )
    bench_result["success_count"] = success_count
    bench_result["success_rate"] = round(success_count / len(cases) * 100, 2) if cases else 0.0

    # 🔥 CRITICAL: 최종 model_info 업데이트 (마지막 케이스 실행 후 실제 모델 정보)
    bench_result["model_info"] = get_actual_model_info()

    # 🔥 CRITICAL: config_info 추가 (Phase 6 validation 필요)
    from app.config import config as app_config
    config_dir = app_config.BASE_DIR / "config"

    with open(config_dir / "rag.json", 'r', encoding='utf-8') as f:
        rag_config = json.load(f)
    with open(config_dir / "prompts.json", 'r', encoding='utf-8') as f:
        prompts_config = json.load(f)

    bench_result["config_info"] = {
        "rag": rag_config,
        "prompts": prompts_config,
    }

    # 평균 메트릭 계산
    successful_cases = [c for c in bench_result["cases"] if c.get("success")]
    if successful_cases:
        avg_f1 = sum(c["metrics"]["f1_score"] for c in successful_cases) / len(successful_cases)
        avg_precision = sum(c["metrics"]["precision"] for c in successful_cases) / len(successful_cases)
        avg_recall = sum(c["metrics"]["recall"] for c in successful_cases) / len(successful_cases)
        avg_turns = sum(c["turn_count"] for c in successful_cases) / len(successful_cases)

        bench_result["average_metrics"] = {
            "f1_score": round(avg_f1, 4),
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "avg_turns": round(avg_turns, 2),
        }

    # 결과 저장
    results_dir = _ensure_results_dir()
    target_path = output_path or get_next_bench_path(results_dir)
    save_result(target_path, bench_result)

    logging.info(f"벤치마크 결과 저장: {target_path}")
    logging.info(f"성공률: {bench_result['success_rate']:.1f}% ({success_count}/{len(cases)})")

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Orchestrator 벤치마크 실행기")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="테스트 질문 JSON 경로")
    parser.add_argument("--output", type=Path, default=None, help="결과 저장 경로 (미지정 시 자동 번호 할당)")
    parser.add_argument("--limit", type=int, default=None, help="실행할 케이스 수 제한")
    parser.add_argument("--no-llm", action="store_true", help="Phase 1 하드코딩 모드 (use_llm=False)")
    parser.add_argument("--max-turns", type=int, default=10, help="최대 턴 수 (기본값: 10)")
    parser.add_argument("--verbose", action="store_true", help="디버그 로그 출력")

    args = parser.parse_args()

    try:
        path = run_bench(
            cases_path=args.cases,
            output_path=args.output,
            limit=args.limit,
            use_llm=not args.no_llm,  # --no-llm 플래그가 있으면 False
            max_turns=args.max_turns,
            verbose=args.verbose,
        )
        print(str(path))
    except Exception:
        logging.exception("벤치마크 실행 중 오류")
        sys.exit(1)


if __name__ == "__main__":
    main()
