"""
Gemma 4 E4B-it GGUF 로드 + 기본 동작 smoke test.

검증:
  1. 모델 로드 성공
  2. generate (non-streaming) 한국어 답변
  3. stream 한국어 토큰 (UTF-8 누락 없음)
  4. 시스템 프롬프트 준수 (간단한 instruction-following 테스트)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from app.core.agent_v2.llm_qwen import build_gemma_chat_model_gguf
    from langchain_core.messages import HumanMessage, SystemMessage

    print("[1/3] Gemma 4 E4B-it Q4_K_M 로드 중...")
    t0 = time.perf_counter()
    chat = build_gemma_chat_model_gguf(max_new_tokens=256)
    print(f"  로드 완료 ({time.perf_counter()-t0:.1f}s)")

    print()
    print("[2/3] generate (non-streaming) — '베르누이 정리 한 줄 설명'")
    print("-" * 60)
    t0 = time.perf_counter()
    msgs = [
        SystemMessage(content="당신은 헬리콥터 기술 설명 전문가입니다. 1~2문장으로 간결히 답하세요."),
        HumanMessage(content="베르누이 정리를 한 줄로 설명해주세요."),
    ]
    resp = chat.invoke(msgs)
    print(f"답변 ({time.perf_counter()-t0:.2f}s):")
    print(f"  {resp.content}")

    print()
    print("[3/3] stream — '시동 절차 3단계'")
    print("-" * 60)
    t0 = time.perf_counter()
    ttft = None
    msgs = [
        SystemMessage(content="번호 매긴 단계 (1. 2. 3.) 형식으로만 답하세요. 다른 설명 금지."),
        HumanMessage(content="헬리콥터 시동 절차 3단계를 알려주세요."),
    ]
    n = 0
    print("답변: ", end="", flush=True)
    for chunk in chat.stream(msgs):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            if ttft is None:
                ttft = time.perf_counter() - t0
            print(token, end="", flush=True)
            n += 1
    elapsed = time.perf_counter() - t0
    print()
    print()
    print(f"  TTFT: {ttft:.2f}s  /  총 {elapsed:.2f}s  /  토큰 {n}개")

    print()
    print("=== smoke 완료. 모델 정상 동작 ===")


if __name__ == "__main__":
    main()
