"""
Gemma 4 vs Qwen 2.5 — 같은 3질의 비교.

지금 떠있는 서버 (CHAT_V2_BACKEND 으로 결정) 에 SSE 호출 3번 보내고,
Langfuse 에서 query/route/답변 메타 추출.

Usage:
  # Gemma 서버 띄운 상태:
  python scripts/agent_v2_poc/34_gemma_vs_qwen_3q.py gemma
  # 그 다음 Qwen 서버 띄운 상태:
  python scripts/agent_v2_poc/34_gemma_vs_qwen_3q.py qwen
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

QUERIES = [
    "헬리콥터 시동 절차를 단계별로 설명해주세요",
    "Vortex Ring State 란 무엇인가요",
    "양력은 어떻게 발생하나요",
]


def call_sse(query: str) -> dict:
    """SSE 호출하고 메타 + 답변 텍스트 누적해서 반환."""
    url = "http://127.0.0.1:8000/chat/v2/stream?" + urllib.parse.urlencode(
        {"message": query, "mode": "multi"}
    )
    t0 = time.perf_counter()
    answer = ""
    rag_info = []
    text_tokens = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
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
                text_tokens += 1
            elif t == "rag_info":
                rag_info.append(c)
    return {
        "elapsed": time.perf_counter() - t0,
        "rag_info": rag_info,
        "n_text_tokens": text_tokens,
        "answer": answer,
    }


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "?"
    print(f"=== Backend: {backend} ===\n")
    for q in QUERIES:
        print(f"Q: {q}")
        try:
            r = call_sse(q)
        except Exception as e:
            print(f"  💥 {e}\n")
            continue
        print(f"  rag_info: {r['rag_info']}")
        print(f"  elapsed: {r['elapsed']:.1f}s, tokens: {r['n_text_tokens']}")
        print(f"  답변: {r['answer'][:400]}")
        print()

    # Langfuse 에서 최근 3개 trace 의 query 추출
    print("--- Langfuse 에서 추출된 query 들 (가장 최근 3개) ---")
    try:
        from langfuse import Langfuse
        c = Langfuse(
            public_key="pk-lf-2276ef94-c1d7-4a1f-ad22-55548ab7be69",
            secret_key="sk-lf-b2363c0c-ba70-49bc-9e35-506762d8178a",
            host="https://cloud.langfuse.com",
        )
        time.sleep(3)
        for t in c.api.trace.list(limit=3).data:
            full = c.api.trace.get(t.id)
            for o in full.observations or []:
                if o.name == "rag_search" and o.type == "TOOL":
                    print(f"  [{t.input!r}] → rag_search.query={str(o.input)[:80]}")
                    break
    except Exception as e:
        print(f"  langfuse 조회 실패: {e}")


if __name__ == "__main__":
    main()
