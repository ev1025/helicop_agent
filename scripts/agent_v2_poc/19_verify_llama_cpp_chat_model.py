"""
QwenLlamaCppChatModel + LangGraph multi_graph 통합 검증.

확인:
  1) bind_tools(tools, tool_choice='required') 로 도구 호출
  2) bind_tools(..., tool_choice='rag_search') 로 특정 도구 강제
  3) multi_graph (Supervisor → Worker → Critic) 한 사이클 정상 종료
  4) 답변 시간 (5~10초/쿼리 기대)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


def main():
    from app.core import models
    from app.core.agent_v2.qwen_llama_cpp_chat_model import QwenLlamaCppChatModel
    from app.core.agent_v2.tools import rag_search, final_answer
    from langchain_core.messages import HumanMessage, SystemMessage

    print("=" * 70)
    print("[1/4] 임베딩 + Chroma + GGUF 모델 로드")
    print("=" * 70)
    models.load_embedding_model()
    models.load_vector_db()

    t0 = time.perf_counter()
    chat = QwenLlamaCppChatModel(max_new_tokens=512)
    print(f"  GGUF 로드: {time.perf_counter()-t0:.1f}초")

    # ── 검증 1: tool_choice='required'
    print()
    print("=" * 70)
    print("[2/4] bind_tools(tool_choice='required') 검증")
    print("=" * 70)
    llm = chat.bind_tools([rag_search, final_answer], tool_choice="required")
    t0 = time.perf_counter()
    ai = llm.invoke([
        SystemMessage(content="헬리콥터 AI 튜터"),
        HumanMessage(content="베르누이 원리 설명"),
    ])
    elapsed = time.perf_counter() - t0
    print(f"  소요: {elapsed:.2f}초")
    print(f"  content: {(ai.content or '')[:100]!r}")
    print(f"  tool_calls: {len(ai.tool_calls or [])}개")
    for tc in ai.tool_calls or []:
        print(f"    name={tc['name']!r}  args_keys={list(tc['args'].keys())}")

    # ── 검증 2: tool_choice='rag_search'
    print()
    print("=" * 70)
    print("[3/4] bind_tools(tool_choice='rag_search') 강제 검증")
    print("=" * 70)
    llm2 = chat.bind_tools([rag_search, final_answer], tool_choice="rag_search")
    t0 = time.perf_counter()
    ai2 = llm2.invoke([
        SystemMessage(content="헬리콥터 AI 튜터"),
        HumanMessage(content="Vortex Ring State 란?"),
    ])
    elapsed = time.perf_counter() - t0
    print(f"  소요: {elapsed:.2f}초")
    print(f"  tool_calls: {len(ai2.tool_calls or [])}개")
    for tc in ai2.tool_calls or []:
        print(f"    name={tc['name']!r}  args={tc['args']}")

    # ── 검증 3: multi_graph 통합
    print()
    print("=" * 70)
    print("[4/4] multi_graph (Supervisor + Workers + Critic) 통합 실행")
    print("=" * 70)
    from app.core.agent_v2.multi_graph import run_multi_agent

    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "안녕하세요",
    ]
    for q in queries:
        print(f"\n  질의: {q}")
        t0 = time.perf_counter()
        try:
            r = run_multi_agent(chat, q)
            elapsed = time.perf_counter() - t0
            print(f"    소요: {elapsed:.2f}초, route={r['route']}, score={r['score']}")
            print(f"    답변 ({len(r['answer'])}자): {r['answer'][:200]}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    [ERROR] {e}")


if __name__ == "__main__":
    main()
