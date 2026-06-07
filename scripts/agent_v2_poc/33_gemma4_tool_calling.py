"""
Gemma 4 E4B-it tool calling 검증.

3가지 케이스:
  1. bind_tools(..., tool_choice="rag_search").invoke(messages) — rag_search 도구 호출 강제
  2. bind_tools(..., tool_choice="auto").invoke(messages) — LLM 자율 선택
  3. with_structured_output(RouteDecision).invoke(messages) — Supervisor 라우팅 패턴

각각 성공/실패 + 출력 확인.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from app.core.agent_v2.llm_qwen import build_gemma_chat_model_gguf
    from app.core.agent_v2.tools import rag_search, final_answer
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    print("[1/4] Gemma 4 E4B-it 로드")
    chat = build_gemma_chat_model_gguf(max_new_tokens=256)
    print("  완료")
    print()

    # ─────────────────────────────────────
    print("[2/4] bind_tools + tool_choice='rag_search' (강제)")
    print("-" * 60)
    try:
        llm = chat.bind_tools([rag_search, final_answer], tool_choice="rag_search")
        msgs = [
            SystemMessage(content="당신은 검색 쿼리 추출기입니다. rag_search 도구로 검색하세요."),
            HumanMessage(content="베르누이 정리에 대해 알려주세요"),
        ]
        ai = llm.invoke(msgs)
        print(f"  content: {ai.content!r}")
        print(f"  tool_calls: {ai.tool_calls}")
        if ai.tool_calls:
            tc = ai.tool_calls[0]
            print(f"  → name: {tc.get('name')}, args: {tc.get('args')}")
            print(f"  ✅ tool_choice 강제 동작")
        else:
            print(f"  ❌ tool_call 없음")
    except Exception as e:
        print(f"  💥 예외: {e}")
    print()

    # ─────────────────────────────────────
    print("[3/4] bind_tools + tool_choice='auto' (LLM 자율)")
    print("-" * 60)
    try:
        llm = chat.bind_tools([rag_search, final_answer], tool_choice="auto")
        msgs = [
            SystemMessage(content="헬리콥터 질문에 답하기 전에 rag_search 도구를 호출하세요."),
            HumanMessage(content="베르누이 정리에 대해 알려주세요"),
        ]
        ai = llm.invoke(msgs)
        print(f"  content: {ai.content!r}")
        print(f"  tool_calls: {ai.tool_calls}")
        if ai.tool_calls:
            tc = ai.tool_calls[0]
            print(f"  → name: {tc.get('name')}, args: {tc.get('args')}")
            print(f"  ✅ auto tool_call 동작")
        else:
            print(f"  ⚠️  도구 호출 안 함 (LLM 이 직답으로 처리)")
    except Exception as e:
        print(f"  💥 예외: {e}")
    print()

    # ─────────────────────────────────────
    print("[4/4] with_structured_output(RouteDecision) — Supervisor 패턴")
    print("-" * 60)

    class RouteDecision(BaseModel):
        route: Literal["researcher", "procedure", "smalltalk"] = Field(
            description="질문 분류: researcher=사실, procedure=절차, smalltalk=잡담"
        )
        reason: str = Field(description="이유 (한국어 1~2문장)")

    try:
        structured = chat.with_structured_output(RouteDecision)
        msgs = [
            SystemMessage(content="사용자 질문을 분류하세요."),
            HumanMessage(content="베르누이 정리에 대해 알려주세요"),
        ]
        decision = structured.invoke(msgs)
        print(f"  route: {decision.route}")
        print(f"  reason: {decision.reason}")
        print(f"  ✅ structured output 동작")
    except Exception as e:
        print(f"  💥 예외: {e}")
    print()

    print("=== 종합 ===")
    print("[2] 통과 → bind_tools 강제 OK → Researcher/Procedure 의 첫 LLM 호출 가능")
    print("[3] 통과 → 단일 ReAct (graph.py) 가능")
    print("[4] 통과 → Supervisor 라우팅 가능 → 풀 multi-agent 흐름 가능")


if __name__ == "__main__":
    main()
