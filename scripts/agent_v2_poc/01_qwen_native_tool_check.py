"""
Qwen2.5-7B native function calling 동작 검증 스크립트.

목적:
  1. Qwen2.5-7B-Instruct 모델을 4-bit로 RTX 4060 (8GB)에 로드 가능한지
  2. tokenizer.apply_chat_template(tools=...) 가 올바르게 동작하는지
  3. 모델이 <tool_call> 태그로 도구 호출을 내놓는지
  4. 한국어 질문에서도 동일하게 작동하는지

실행: .venv/Scripts/python.exe scripts/agent_v2_poc/01_qwen_native_tool_check.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "헬리콥터 표준 교재 PDF에서 사용자 질문과 관련된 문서를 검색한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 한국어 키워드. 30자 이내 핵심 키워드만.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "최종적으로 사용할 문서 수.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "충분한 정보가 모였을 때 사용자에게 최종 답변을 전달한다.",
            "parameter": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "사용자에게 보여줄 최종 답변 텍스트.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


def load_model():
    print("=" * 60)
    print(f"모델 로드: {MODEL_NAME} (4-bit 양자화)")
    print("=" * 60)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
    )
    print(f"  로드 완료. VRAM 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return tok, model


def render_prompt(tok, user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    text = tok.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def generate(tok, model, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    response_tokens = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(response_tokens, skip_special_tokens=False)


def main():
    tok, model = load_model()

    test_questions = [
        "베르누이 원리로 양력이 어떻게 발생하나요?",
        "헬리콥터 메인 로터의 피치 각도가 양력에 미치는 영향은?",
        "안녕하세요",  # 도구 불필요한 일반 인사
    ]

    for i, q in enumerate(test_questions, 1):
        print()
        print("=" * 60)
        print(f"[테스트 {i}] 질문: {q}")
        print("=" * 60)

        prompt = render_prompt(tok, q)
        print()
        print("--- 모델에 전달되는 프롬프트 (마지막 800자) ---")
        print(prompt[-800:])
        print()

        response = generate(tok, model, prompt)
        print("--- 모델 응답 (raw, special token 포함) ---")
        print(response)
        print()


if __name__ == "__main__":
    main()
