"""
llama-cpp + Qwen GGUF 의 tool calling 동작 확인 (작은 검증).

확인 사항:
  1) create_chat_completion 에 tools 전달 시 모델이 tool_calls 반환하는가
  2) tool_choice 강제 동작하는가
  3) 응답 형식이 OpenAI 호환인가 (response.choices[0].message.tool_calls)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if hasattr(os, 'add_dll_directory') and os.path.isdir(torch_lib):
    os.add_dll_directory(torch_lib)

from llama_cpp import Llama


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "헬리콥터 표준 교재에서 관련 문서를 검색한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색 키워드 (한국어)"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "사용자에게 최종 답변을 전달한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "최종 답변"},
                },
                "required": ["answer"],
            },
        },
    },
]


def main():
    candidates = list(Path.home().glob(".cache/huggingface/hub/**/*Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    if not candidates:
        print("GGUF 파일 없음")
        return
    gguf_path = str(candidates[0])
    print(f"파일: {gguf_path}")

    print("\n[1/3] 모델 로드")
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
        chat_format="chatml-function-calling",  # Qwen 호환
    )

    queries = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "안녕하세요",
    ]

    print("\n[2/3] tool_choice='auto' (기본)")
    for q in queries:
        print(f"\n  질의: {q}")
        try:
            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": q}],
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=256,
            )
            msg = resp["choices"][0]["message"]
            print(f"    content: {(msg.get('content') or '')[:120]!r}")
            print(f"    tool_calls: {msg.get('tool_calls')}")
        except Exception as e:
            print(f"    [ERROR] {e}")

    print("\n[3/3] tool_choice = rag_search 강제")
    q = queries[0]
    print(f"  질의: {q}")
    try:
        resp = llm.create_chat_completion(
            messages=[{"role": "user", "content": q}],
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "rag_search"}},
            max_tokens=256,
        )
        msg = resp["choices"][0]["message"]
        print(f"    content: {(msg.get('content') or '')[:120]!r}")
        tcs = msg.get('tool_calls') or []
        print(f"    tool_calls 수: {len(tcs)}")
        for tc in tcs:
            fn = tc.get('function', {})
            print(f"      name={fn.get('name')!r}  args={fn.get('arguments')!r}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"    [ERROR] {e}")


if __name__ == "__main__":
    main()
