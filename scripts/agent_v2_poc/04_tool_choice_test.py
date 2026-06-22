"""
QwenChatModel.bind_tools(tool_choice=...) 강제 옵션 검증.

3가지 모드 테스트:
  - tool_choice="auto"     : 모델이 자율 결정 (기본, 03 와 동일)
  - tool_choice="required" : 반드시 도구 중 하나 호출
  - tool_choice="rag_search" (이름 단축형): 반드시 rag_search 호출

기대 결과:
  - "required" / "rag_search" 모드에서는 도구 호출이 100% 발생해야 함
  - 응답 형식이 prefix 강제로 안정화 (BPE 노이즈 거의 없어야 함)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@tool
def rag_search(query: str, top_k: int = 5) -> str:
    """헬리콥터 표준 교재(PDF)에서 관련 문서를 검색한다.

    Args:
        query: 검색할 한국어 키워드.
        top_k: 사용할 문서 수.
    """
    return json.dumps({"documents": [{"content": "mock", "page": 1}], "count": 1}, ensure_ascii=False)


@tool
def final_answer(answer: str) -> str:
    """사용자에게 최종 답변을 전달한다.

    Args:
        answer: 사용자에게 보여줄 답변.
    """
    return answer


ALL_TOOLS = [rag_search, final_answer]

SYSTEM_PROMPT = "당신은 헬리콥터 AI 튜터입니다. 한국어로 간결하게 답변하세요."


def main():
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model

    print("[1/2] Qwen 로드")
    chat = build_qwen_chat_model(model_name="Qwen/Qwen2.5-7B-Instruct")

    print("[2/2] tool_choice 시나리오별 테스트")
    query = "베르누이 원리로 양력이 어떻게 발생하나요?"

    scenarios = [
        ("auto (기본)", {"tool_choice": None}),
        ("required (도구 중 하나 강제)", {"tool_choice": "required"}),
        ("rag_search 강제 (특정 도구 지정)", {"tool_choice": "rag_search"}),
        ("final_answer 강제 (특정 도구 지정)", {"tool_choice": "final_answer"}),
    ]

    for name, bind_kwargs in scenarios:
        print()
        print("=" * 70)
        print(f"[{name}]  bind_kwargs={bind_kwargs}")
        print("=" * 70)

        llm_with_tools = chat.bind_tools(ALL_TOOLS, **bind_kwargs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        try:
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"  content (200자): {str(ai_msg.content)[:200]!r}")
        if ai_msg.tool_calls:
            print(f"  tool_calls 수: {len(ai_msg.tool_calls)}")
            for tc in ai_msg.tool_calls:
                args_preview = {k: (str(v)[:60] + '...' if len(str(v)) > 60 else v) for k, v in tc.get("args", {}).items()}
                print(f"    - name={tc['name']!r}, args={args_preview}")
        else:
            print(f"  tool_calls 없음 (LLM 직답)")


if __name__ == "__main__":
    main()
