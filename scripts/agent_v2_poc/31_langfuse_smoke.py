"""
#10 Langfuse 통합 smoke 테스트.

검증:
  1. env vars 없으면 no-op (예외 없이 전체 흐름 정상 실행)
  2. env vars 설정 시 요청 1개 = trace 1개 로 묶여 전송 (Supervisor + Researcher 가 한 trace)

env vars (.env 또는 셸):
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=https://cloud.langfuse.com
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from app.core import models
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model_gguf
    from app.core.agent_v2.agents import supervisor as sv_mod
    from app.core.agent_v2.agents.researcher import run_streaming
    from app.core.agent_v2.langfuse_handler import request_span, is_enabled, flush

    print("[1/3] 임베딩 + Chroma + Qwen GGUF 로드")
    models.load_embedding_model()
    models.load_vector_db()
    chat = build_qwen_chat_model_gguf(max_new_tokens=256)

    print(f"[2/3] Langfuse 활성화: {is_enabled()}")
    if not is_enabled():
        print("       → env vars 미설정이므로 no-op. 트레이스 전송 안 함.")

    print()
    print("[3/3] 요청 1개 = trace 1개 (Supervisor → Researcher 한 묶음)")
    print("-" * 60)
    query = "베르누이 원리로 양력이 어떻게 발생하나요?"
    print(f"Q: {query}")
    print()

    t0 = time.perf_counter()
    n_tokens = 0
    full = ""
    # 동기 스크립트라 OTEL contextvars 가 그냥 전파됨 (async 불필요).
    with request_span("smoke-chat-v2-multi", input=query) as span:
        decision = sv_mod.route(chat, query)
        print(f"  Supervisor → {decision.route} ({time.perf_counter()-t0:.2f}s)")
        print(f"  답변: ", end="", flush=True)
        for ev_type, content in run_streaming(chat, query):
            if ev_type == "text":
                print(content, end="", flush=True)
                full += content
                n_tokens += 1
        if span is not None:
            try:
                span.update(output=full)
                span.set_trace_io(output=full)
            except Exception:
                pass
    flush()

    elapsed = time.perf_counter() - t0
    print()
    print()
    print(f"  총 시간 {elapsed:.2f}s, 토큰 {n_tokens}개")
    print()
    if is_enabled():
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        print(f"  → {host} → Tracing 에서 'smoke-chat-v2-multi' 1개 trace 안에")
        print(f"     RunnableSequence / QwenLlamaCppChatModel 들이 nested 돼 있어야 함.")
    else:
        print(f"  → no-op 모드. 모든 LLM 호출 정상 동작 + langfuse 영향 0.")


if __name__ == "__main__":
    main()
