"""
#9 streaming smoke test.

QwenLlamaCppChatModel._stream + researcher.run_streaming 동작 확인.
TTFT (time to first token) 와 총 시간 측정.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from app.core import models
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model_gguf
    from app.core.agent_v2.agents.researcher import run_streaming

    print("[1/2] 임베딩 + Chroma + Qwen GGUF 로드")
    models.load_embedding_model()
    models.load_vector_db()
    chat = build_qwen_chat_model_gguf(max_new_tokens=512)

    print()
    print("[2/2] 스트리밍 측정")
    print("-" * 60)
    query = "베르누이 원리로 양력이 어떻게 발생하나요?"
    print(f"Q: {query}")
    print()

    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    full_answer = ""

    for ev_type, content in run_streaming(chat, query):
        now = time.perf_counter()
        if ev_type == "rag_info":
            print(f"[{now - t0:5.2f}s] {content}")
        elif ev_type == "text":
            if ttft is None:
                ttft = now - t0
                print(f"[{ttft:5.2f}s] ⭐ TTFT (첫 토큰)")
                print()
                print("답변: ", end="", flush=True)
            print(content, end="", flush=True)
            full_answer += content
            n_tokens += 1

    elapsed = time.perf_counter() - t0
    print()
    print()
    print("-" * 60)
    print(f"TTFT      : {ttft:.2f}s" if ttft else "TTFT      : (no tokens)")
    print(f"총 시간   : {elapsed:.2f}s")
    print(f"토큰 수   : {n_tokens}")
    print(f"답변 길이 : {len(full_answer)}자")


if __name__ == "__main__":
    main()
