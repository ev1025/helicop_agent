"""
#011 - chat_v2 라우트가 GGUF 백엔드로 동작하는지 직접 검증.

uvicorn 없이 chat_v2_stream 의 event_stream() 을 직접 소비.

확인:
  - 기본 backend = gguf (#011 변경 적용 확인)
  - SSE 이벤트 형식 (info, text, done)
  - 응답 시간이 GGUF 수준 (5~25초)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


async def consume(route_call):
    """StreamingResponse 의 body_iterator 를 한 청크씩 소비."""
    response = route_call
    chunks = []
    iterator = response.body_iterator
    async for chunk in iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        chunks.append(chunk)
    return chunks


async def main():
    # 임베딩 + Chroma 먼저 로드 (multi_graph 의 rag_search 가 사용)
    from app.core import models
    print("=" * 70)
    print("[1/3] RAG 인프라 로드")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()

    # chat_v2 임포트 + GGUF 모델 로드
    from app.api.routes import chat_v2
    print()
    print("=" * 70)
    print("[2/3] chat_v2 ChatModel 초기화 (backend=gguf)")
    print("=" * 70)
    t0 = time.perf_counter()
    chat = chat_v2.get_chat_model()
    print(f"  로드: {time.perf_counter()-t0:.1f}초, 타입: {type(chat).__name__}")

    queries = [
        ("multi", "베르누이 원리로 양력이 어떻게 발생하나요?"),
        ("multi", "안녕하세요"),
        ("single", "Vortex Ring State 란 무엇인가요?"),
    ]

    print()
    print("=" * 70)
    print("[3/3] /chat/v2/stream 시뮬레이션")
    print("=" * 70)
    for mode, q in queries:
        print()
        print(f"--- mode={mode}, query={q!r}")
        t0 = time.perf_counter()
        # 라우트 함수 직접 호출 (FastAPI 의존성 무시)
        response = await chat_v2.chat_v2_stream(message=q, mode=mode)
        chunks = await consume(response)
        elapsed = time.perf_counter() - t0
        print(f"  소요: {elapsed:.2f}초")
        print(f"  SSE chunks: {len(chunks)}개")
        # 이벤트 타입별로 한 줄씩 미리보기
        for c in chunks:
            for line in c.splitlines():
                if line.startswith("data: "):
                    try:
                        evt = json.loads(line[6:])
                        t = evt.get("type", "?")
                        content = (evt.get("content") or "")[:120]
                        print(f"    [{t:8s}] {content}")
                    except Exception:
                        print(f"    {line[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
