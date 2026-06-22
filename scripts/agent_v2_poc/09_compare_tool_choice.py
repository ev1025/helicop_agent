"""
#002 - tool_choice 강제 옵션 비교 (auto vs required vs 특정 도구).

비교 대상:
  A. tool_choice = None (auto)            : 기존 동작, LLM 자율 결정
  B. tool_choice = "required"             : 도구 중 하나는 반드시 호출
  C. tool_choice = "rag_search"           : 특정 도구 강제 (assistant prefix 박기)

측정 metric:
  - tool_call_success_rate : tool_calls 가 비어있지 않은 비율
  - required_args_present  : 호출된 도구에 필수 인자가 채워졌는지 비율
  - tool_name_correct      : 호출된 도구가 의도한 도구(rag_search)인지 비율
  - bpe_noise_chars        : AIMessage.content 의 노이즈 글자 수 (ronics 등)
  - elapsed_sec            : 단일 LLM 호출 시간

LLM 답변 품질은 별도 (벤치 #006 에서 일괄). 여기는 tool 호출 단계만 측정.
agent loop 안 돌림 → 빠름 (질의당 ~1~2분).

실행:
    .venv/Scripts/python.exe scripts/agent_v2_poc/09_compare_tool_choice.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool


@tool
def rag_search(query: str, top_k: int = 5) -> str:
    """헬리콥터 표준 교재(PDF)에서 사용자 질문과 관련된 문서를 검색한다.

    Args:
        query: 검색할 한국어 키워드.
        top_k: 사용할 문서 수.
    """
    return ""


@tool
def final_answer(answer: str) -> str:
    """사용자에게 최종 답변을 전달한다.

    Args:
        answer: 사용자에게 보여줄 답변.
    """
    return answer


ALL_TOOLS = [rag_search, final_answer]
SYSTEM_PROMPT = "당신은 헬리콥터 AI 튜터입니다. 한국어로 간결하게 답변하세요."

# BPE 잔여 노이즈 패턴 (3~12자 알파벳 only 단독 토큰)
_NOISE_RE = re.compile(r"\b[a-zA-Z_]{3,12}\b")


def measure(llm_with_tools, query: str, expect_tool: str | None = None) -> dict:
    """단일 LLM 호출 → 도구 호출 메타데이터 측정."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    t0 = time.perf_counter()
    try:
        ai: AIMessage = llm_with_tools.invoke(messages)
        ok = True
        err = None
    except Exception as e:
        ai = AIMessage(content=str(e), tool_calls=[])
        ok = False
        err = str(e)
    elapsed = time.perf_counter() - t0

    tool_calls = ai.tool_calls or []
    has_tool_call = len(tool_calls) > 0

    # required args 검사 (rag_search → query 필수, final_answer → answer 필수)
    required_args_present = False
    tool_name_correct = False
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        if name == "rag_search":
            required_args_present = bool(args.get("query"))
            if expect_tool == "rag_search" or expect_tool is None:
                tool_name_correct = True
        elif name == "final_answer":
            required_args_present = bool(args.get("answer"))
            if expect_tool == "final_answer":
                tool_name_correct = True
        if expect_tool is None and required_args_present:
            tool_name_correct = True
        break  # 첫 호출만 본다

    # BPE 노이즈 추정: content 에서 의미 없는 영문 토큰 갯수
    content = str(ai.content) if ai.content else ""
    noise_tokens = _NOISE_RE.findall(content)
    bpe_noise_chars = sum(len(t) for t in noise_tokens)

    return {
        "ok": ok,
        "err": err,
        "elapsed_sec": round(elapsed, 2),
        "has_tool_call": has_tool_call,
        "required_args_present": required_args_present,
        "tool_name_correct": tool_name_correct,
        "tool_called": tool_calls[0].get("name") if tool_calls else None,
        "tool_args": tool_calls[0].get("args") if tool_calls else None,
        "content_len": len(content),
        "bpe_noise_chars": bpe_noise_chars,
        "content_preview": content[:80],
    }


def main():
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model

    print("=" * 70)
    print("[1/2] Qwen 로드")
    print("=" * 70)
    chat = build_qwen_chat_model(model_name="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=256)

    # 3가지 모드 준비
    llm_auto = chat.bind_tools(ALL_TOOLS)                              # tool_choice=None
    llm_required = chat.bind_tools(ALL_TOOLS, tool_choice="required")  # 도구 중 하나
    llm_force_rag = chat.bind_tools(ALL_TOOLS, tool_choice="rag_search")

    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "헬리콥터 메인 로터의 피치 각도가 양력에 미치는 영향은?",
        "Vortex Ring State 란 무엇인가요?",
        "동적 롤오버에 대해 설명해줘",
        "헬리콥터의 양력 4요소가 뭐야",
    ]

    modes = [
        ("auto",          llm_auto,      None),
        ("required",      llm_required,  None),
        ("force_rag_search", llm_force_rag, "rag_search"),
    ]

    print()
    print("=" * 70)
    print(f"[2/2] {len(queries)} 질의 × {len(modes)} 모드 측정 (각 호출 30~120초 예상)")
    print("=" * 70)

    rows = []
    for q in queries:
        print(f"\n질의: {q}")
        for mode_name, llm_inst, expect in modes:
            t0 = time.perf_counter()
            r = measure(llm_inst, q, expect_tool=expect)
            t = time.perf_counter() - t0
            print(
                f"  [{mode_name:14s}] tool={r['tool_called']!r:18s} "
                f"args_ok={r['required_args_present']:1d} "
                f"name_ok={r['tool_name_correct']:1d} "
                f"noise={r['bpe_noise_chars']:3d}자 "
                f"elapsed={t:5.1f}s"
            )
            rows.append({"query": q, "mode": mode_name, **r})

    # 종합
    def agg(mode):
        sub = [r for r in rows if r["mode"] == mode]
        n = len(sub)
        return {
            "n": n,
            "tool_call_rate": round(sum(r["has_tool_call"] for r in sub) / n, 3),
            "args_present_rate": round(sum(r["required_args_present"] for r in sub) / n, 3),
            "name_correct_rate": round(sum(r["tool_name_correct"] for r in sub) / n, 3),
            "avg_bpe_noise": round(sum(r["bpe_noise_chars"] for r in sub) / n, 1),
            "avg_elapsed_sec": round(sum(r["elapsed_sec"] for r in sub) / n, 2),
        }

    summary = {m[0]: agg(m[0]) for m in modes}

    print()
    print("=" * 70)
    print("종합 (질의당 평균)")
    print("=" * 70)
    print(f"{'mode':16s} | {'tool_rate':>9s} | {'args_ok':>7s} | {'name_ok':>7s} | {'noise':>5s} | {'sec':>5s}")
    print("-" * 70)
    for m, s in summary.items():
        print(
            f"{m:16s} | {s['tool_call_rate']:>9.3f} | {s['args_present_rate']:>7.3f} | "
            f"{s['name_correct_rate']:>7.3f} | {s['avg_bpe_noise']:>5.1f} | {s['avg_elapsed_sec']:>5.2f}"
        )

    out = ROOT / "results" / "agent_v2_changes" / "02_tool_choice_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
