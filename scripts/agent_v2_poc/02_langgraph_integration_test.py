"""
LangGraph + ChatHuggingFace + Qwen native tool calling 통합 검증.

목적:
  - app/core/agent_v2/ 의 graph.py 와 runner.py 가 실제로 동작하는지
  - ChatHuggingFace.bind_tools() 가 Qwen의 <tool_call> 응답을
    AIMessage.tool_calls 로 자동 파싱하는지
  - LangGraph 의 ToolNode 가 LangChain @tool 함수를 정상 실행하는지
  - 라우팅 (final_answer 시 종료) 이 의도대로 동작하는지

실제 RAG 검색 / Chroma DB / 임베딩 모델 로드를 피하기 위해
rag_search 도구를 mock 으로 패치한다.

실행: .venv/Scripts/python.exe scripts/agent_v2_poc/02_langgraph_integration_test.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 프로젝트 루트를 path 에 추가 (스크립트 단독 실행 대비)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

# ────────────────────────────────────────────────────────────
# rag_search 를 mock 으로 패치 (실제 Chroma/임베딩 로드 회피)
# ────────────────────────────────────────────────────────────
import json
from langchain_core.tools import tool


@tool
def rag_search(query: str, top_k: int = 5, reranker_top_k: int = 5) -> str:
    """헬리콥터 표준 교재(PDF)에서 사용자 질문과 관련된 문서를 검색한다.

    질문에 답변하기 전에 이 도구를 먼저 호출해야 한다.

    Args:
        query: 검색할 한국어 키워드. 30자 이내 핵심 키워드만.
        top_k: 벡터 검색 후보 수.
        reranker_top_k: 재순위 후 사용할 문서 수.

    Returns:
        JSON 문자열. {"documents": [...], "count": N} 형식.
    """
    # 실제 검색 대신 고정 응답
    fake_docs = [
        {
            "content": (
                "헬리콥터 메인 로터 블레이드는 에어포일 형상으로 설계되어 있어, "
                "회전 시 베르누이의 정리에 따라 블레이드 상부의 공기 속도가 빨라지고 압력이 낮아진다. "
                "이 압력 차이가 위쪽으로 향하는 양력을 만든다. "
                "양력의 크기는 블레이드 회전 속도, 피치 각도, 공기 밀도에 따라 결정된다."
            ),
            "page": 17,
            "source": "조종사표준교재.pdf",
            "score": 0.87,
        },
        {
            "content": (
                "피치 각도(pitch angle)가 증가하면 블레이드와 공기의 받음각이 커져 "
                "양력이 비례하여 증가한다. 다만 임계각을 넘으면 실속(stall) 이 발생해 "
                "양력이 급격히 감소한다."
            ),
            "page": 19,
            "source": "조종사표준교재.pdf",
            "score": 0.71,
        },
    ]
    payload = {"documents": fake_docs, "count": len(fake_docs)}
    print(f"\n[mock rag_search 호출됨] query={query!r}, 반환 문서 {len(fake_docs)}개")
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def final_answer(answer: str) -> str:
    """사용자에게 최종 답변을 전달하고 대화 루프를 종료한다.

    참고 문서 검색을 충분히 마친 뒤 호출한다.

    Args:
        answer: 사용자에게 보여줄 최종 답변 텍스트 (한국어).

    Returns:
        입력으로 받은 answer 그대로.
    """
    print(f"\n[final_answer 호출됨] answer 길이 = {len(answer)}자")
    return answer


ALL_TOOLS = [rag_search, final_answer]

# ────────────────────────────────────────────────────────────
# 그래프 조립 (app/core/agent_v2/graph.py 의 로직을 이식)
# ────────────────────────────────────────────────────────────
from typing import Annotated, Literal, Sequence, TypedDict
import operator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _route_after_llm(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END
    return "tools"


def _route_after_tool(state: AgentState) -> Literal["llm", "__end__"]:
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        if msg.name == "final_answer":
            return END
    return "llm"


def build_graph(llm_with_tools):
    def call_llm(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    g = StateGraph(AgentState)
    g.add_node("llm", call_llm)
    g.add_node("tools", tool_node)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", _route_after_llm)
    g.add_conditional_edges("tools", _route_after_tool)
    return g.compile()


# ────────────────────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "당신은 헬리콥터 표준 교재를 바탕으로 답변하는 항공 AI 튜터입니다.\n"
    "사용자 질문에 답하기 전 반드시 rag_search 도구를 호출해 관련 문서를 찾고, "
    "검색 결과를 바탕으로 final_answer 도구로 최종 답변을 전달하세요.\n"
    "- 추측 금지. 문서에 없는 내용은 '문서에 충분한 정보가 없습니다'라고 답하세요.\n"
    "- 답변은 간결하고 한국어로 작성하세요."
)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[1/4] {MODEL} 로드 중...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
    print(f"      VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("[2/4] HuggingFacePipeline + ChatHuggingFace 래핑")
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                    max_new_tokens=512, temperature=0.3, top_p=0.85,
                    do_sample=True, return_full_text=False)
    hf = HuggingFacePipeline(pipeline=pipe)
    chat = ChatHuggingFace(llm=hf, tokenizer=tok, model_id=MODEL)

    print("[3/4] bind_tools + LangGraph 조립")
    llm_with_tools = chat.bind_tools(ALL_TOOLS)
    app = build_graph(llm_with_tools)

    print("[4/4] 질의 실행")
    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "피치 각도가 양력에 어떤 영향을 주나요?",
    ]

    for i, q in enumerate(queries, 1):
        print()
        print("=" * 70)
        print(f"[테스트 {i}] {q}")
        print("=" * 70)

        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=q),
            ]
        }
        try:
            final_state = app.invoke(initial_state, config={"recursion_limit": 8})
        except Exception as e:
            print(f"[ERROR] 그래프 실행 실패: {e}")
            continue

        print(f"\n  메시지 누적: {len(final_state['messages'])}개")
        for j, msg in enumerate(final_state["messages"]):
            kind = type(msg).__name__
            preview = str(msg.content)[:120].replace("\n", " ")
            tool_info = ""
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_info = f" [tool_calls={[tc['name'] for tc in msg.tool_calls]}]"
            print(f"  [{j}] {kind}{tool_info}: {preview}...")

        # 최종 답변 추출
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "final_answer":
                print(f"\n  최종 답변:\n  {msg.content}")
                break
        else:
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n  LLM 직답:\n  {msg.content}")
                    break


if __name__ == "__main__":
    main()
