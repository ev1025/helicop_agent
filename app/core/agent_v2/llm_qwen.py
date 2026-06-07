"""
GGUF (llama.cpp) ChatModel 팩토리.

두 가지 백엔드 함수:
  - build_qwen_chat_model_gguf()    : Qwen2.5-7B GGUF Q4_K_M
  - build_gemma_chat_model_gguf()   : Gemma 4 E4B-it GGUF Q4_K_M (기본)

둘 다 동일한 QwenLlamaCppChatModel 래퍼를 쓰고, chat_format 만 다르다.
"""

from __future__ import annotations

import logging

from app.core.agent_v2.qwen_llama_cpp_chat_model import QwenLlamaCppChatModel

logger = logging.getLogger(__name__)


def build_qwen_chat_model_gguf(
    gguf_filename_pattern: str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    n_ctx: int = 8192,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> QwenLlamaCppChatModel:
    """Qwen2.5-7B Instruct, Q4_K_M, llama.cpp 백엔드.

    먼저 huggingface_hub 로 GGUF 파일을 받아 캐시에 두어야 한다.
    예: bartowski/Qwen2.5-7B-Instruct-GGUF / Qwen2.5-7B-Instruct-Q4_K_M.gguf
    """
    return QwenLlamaCppChatModel(
        gguf_filename_pattern=gguf_filename_pattern,
        n_ctx=n_ctx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def build_gemma_chat_model_gguf(
    gguf_filename_pattern: str = "gemma-4-E4B-it-Q4_K_M.gguf",
    n_ctx: int = 8192,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> QwenLlamaCppChatModel:
    """Gemma 4 E4B-it, Q4_K_M, llama.cpp 백엔드.

    chat_format=None → GGUF 임베디드 jinja 템플릿 자동 사용 (Gemma 4 표준 역할 +
    native function calling). 도구 호출 안정성 우선이라 temperature=0.0.

    먼저 huggingface_hub 로 GGUF 파일을 받아 캐시에 두어야 한다.
    예: unsloth/gemma-4-E4B-it-GGUF / gemma-4-E4B-it-Q4_K_M.gguf
    """
    return QwenLlamaCppChatModel(
        model_id="google/gemma-4-E4B-it",
        gguf_filename_pattern=gguf_filename_pattern,
        n_ctx=n_ctx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        chat_format=None,
    )
