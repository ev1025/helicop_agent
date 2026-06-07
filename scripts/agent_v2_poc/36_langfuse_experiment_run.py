"""
Langfuse Experiment runner — `agent_v2_eval_7q` 데이터셋 자동 평가.

각 아이템마다 /chat/v2/stream SSE 호출 → 답변/route/시간 수집 →
keyword_recall / route_correct / latency_sec / answer_length 점수 산출 →
Langfuse Dataset Run 으로 묶어 push (UI 에서 run 별 비교 가능).

실행:
    .venv/Scripts/python.exe scripts/agent_v2_poc/36_langfuse_experiment_run.py <run_name>
    예) ... gemma-arctic-ko-v1
    예) ... qwen-arctic-ko-v1

서버가 떠 있어야 함 (CHAT_V2_BACKEND 환경변수로 백엔드 선택).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


DATASET_NAME = "agent_v2_eval_7q"
SSE_URL = "http://127.0.0.1:8000/chat/v2/stream"


def call_sse(query: str) -> dict:
    """SSE 호출하고 메타 + 답변 누적해서 반환."""
    url = SSE_URL + "?" + urllib.parse.urlencode({"message": query, "mode": "multi"})
    t0 = time.perf_counter()
    answer = ""
    route = None
    rag_chars = 0
    with urllib.request.urlopen(url, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except Exception:
                continue
            t = ev.get("type")
            c = ev.get("content", "")
            if t == "text":
                answer += c
            elif t == "rag_info":
                if "Supervisor →" in c:
                    m = re.search(r"Supervisor → (\w+)", c)
                    if m:
                        route = m.group(1)
                elif "RAG 결과" in c:
                    m = re.search(r"(\d+)자", c)
                    if m:
                        rag_chars = int(m.group(1))
    return {
        "answer": answer,
        "route": route,
        "rag_chars": rag_chars,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "answer_length": len(answer),
    }


def keyword_recall(answer: str, expected_keywords: list) -> float:
    if not expected_keywords:
        return 0.0
    return sum(1 for k in expected_keywords if k in answer) / len(expected_keywords)


# ── task: 아이템 1개 처리 (서버 호출) ───────────────────────────
def task(*, item, **kwargs):
    query = item.input["query"]
    print(f"  · {query[:50]}", flush=True)
    return call_sse(query)


from langfuse.experiment import Evaluation


# ── 평가자들 (각자 Evaluation 1개 반환) ──────────────────────────
def eval_keyword_recall(*, input, output, expected_output, metadata, **kwargs):
    kw = (expected_output or {}).get("expected_keywords", [])
    return Evaluation(
        name="keyword_recall",
        value=keyword_recall(output.get("answer", ""), kw),
        comment=f"expected_keywords={kw}",
    )


def eval_route_correct(*, input, output, expected_output, metadata, **kwargs):
    expected = (expected_output or {}).get("expected_route")
    actual = output.get("route")
    return Evaluation(
        name="route_correct",
        value=1.0 if expected and actual == expected else 0.0,
        comment=f"expected={expected!r} actual={actual!r}",
    )


def eval_latency(*, input, output, expected_output, metadata, **kwargs):
    return Evaluation(name="latency_sec", value=output.get("elapsed_sec", 0))


def eval_length(*, input, output, expected_output, metadata, **kwargs):
    return Evaluation(name="answer_length", value=output.get("answer_length", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", help="실험 이름 (예: gemma-arctic-ko-v1)")
    ap.add_argument("--description", default="", help="실험 설명")
    args = ap.parse_args()

    from app.core.agent_v2.langfuse_handler import is_enabled, _init
    _init()
    if not is_enabled():
        print("❌ Langfuse 비활성화. .env 확인.")
        sys.exit(1)

    # 서버 살아있는지 확인
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/chat", timeout=5) as r:
            if r.status != 200:
                raise Exception(f"status {r.status}")
    except Exception as e:
        print(f"❌ 서버 응답 없음 (uvicorn 시작 필요): {e}")
        sys.exit(1)

    from langfuse import get_client
    client = get_client()
    dataset = client.get_dataset(name=DATASET_NAME)
    print(f"=== Run: {args.run_name}  /  데이터셋: {DATASET_NAME} ({len(list(dataset.items))} 아이템) ===\n")

    result = dataset.run_experiment(
        name=args.run_name,
        run_name=args.run_name,
        description=args.description or f"Auto run {args.run_name}",
        task=task,
        evaluators=[eval_keyword_recall, eval_route_correct, eval_latency, eval_length],
        max_concurrency=1,  # LLM 한 대 — 동시 호출 금지
    )

    client.flush()
    print()
    print(result.format())
    print()
    if hasattr(result, "dataset_run_url") and result.dataset_run_url:
        print(f"  Langfuse UI: {result.dataset_run_url}")


if __name__ == "__main__":
    main()
