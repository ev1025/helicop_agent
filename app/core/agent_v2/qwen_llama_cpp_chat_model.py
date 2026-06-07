"""
Qwen GGUF + llama.cpp 를 LangChain ChatModel 인터페이스로 노출.

#008 결과: GGUF Q4_K_M + llama.cpp 가 BNB 4bit 대비 60× 빠름 (304s → 5.2s).
이 클래스를 사용하면 LangGraph multi_graph / agent_v2 전체가 동일 속도 향상.

기존 QwenChatModel 과 인터페이스 100% 호환 (graph.py / runner.py 변경 불필요):
  - bind_tools(tools, tool_choice=...) 지원
  - _generate(messages) → ChatResult
  - with_structured_output() 자동 동작 (BaseChatModel 기본 구현이 bind_tools 활용)

Windows 한글 경로 + CUDA dll 로딩 우회 코드 포함.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, List, Optional, Sequence

from typing import Iterator

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)


def _ensure_llama_cpp_dlls():
    """Windows 에서 llama_cpp 로드 전 torch 의 CUDA dll 경로를 등록."""
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if hasattr(os, "add_dll_directory") and os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass


def _find_gguf(filename_glob: str) -> Optional[str]:
    """HuggingFace 캐시에서 GGUF 파일 찾기."""
    candidates = list(
        Path.home().glob(f".cache/huggingface/hub/**/*{filename_glob}*")
    )
    return str(candidates[0]) if candidates else None


class QwenLlamaCppChatModel(BaseChatModel):
    """Qwen GGUF (Q4_K_M) 를 llama.cpp 백엔드로 LangChain ChatModel 노출."""

    model_id: str = Field(default="Qwen/Qwen2.5-7B-Instruct-GGUF")
    gguf_filename_pattern: str = Field(default="Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    n_ctx: int = Field(default=8192)  # RAG 결과는 rag_search 도구가 MAX_CONTEXT_LENGTH 로 캡 → ~6400 토큰, 8192 면 충분.
    n_gpu_layers: int = Field(default=-1)
    max_new_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.0)  # GGUF + tool 강제는 greedy 가 안정적
    # llama.cpp chat 핸들러. 기본은 Qwen 의 chatml-function-calling.
    # Gemma 4 등 자체 템플릿이 GGUF 에 임베딩된 모델은 None 으로 두면 auto-detect.
    chat_format: Optional[str] = Field(default="chatml-function-calling")

    _llm: Any = PrivateAttr(default=None)
    _tools_schema: Optional[List[dict]] = PrivateAttr(default=None)
    _tool_choice: Any = PrivateAttr(default=None)

    def __init__(self, *, _shared_llm=None, **kwargs):
        """모델 로드.

        Args:
            _shared_llm: 이미 로드된 Llama 인스턴스. bind_tools 가 새 인스턴스를
                         만들 때 모델 재로드를 피하기 위해 내부적으로 사용.
        """
        super().__init__(**kwargs)
        if _shared_llm is not None:
            self._llm = _shared_llm
            return

        _ensure_llama_cpp_dlls()
        from llama_cpp import Llama

        gguf_path = _find_gguf(self.gguf_filename_pattern)
        if not gguf_path:
            raise FileNotFoundError(
                f"GGUF 파일 못 찾음 (패턴: {self.gguf_filename_pattern}). "
                f"먼저 huggingface_hub.hf_hub_download 으로 받으세요."
            )
        logger.info(f"[QwenLlamaCpp] 로드: {gguf_path}  (chat_format={self.chat_format})")
        llama_kwargs = dict(
            model_path=gguf_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        # chat_format=None 이면 GGUF 임베디드 jinja 템플릿 자동 사용 (Gemma 4 등).
        if self.chat_format:
            llama_kwargs["chat_format"] = self.chat_format
        self._llm = Llama(**llama_kwargs)

    @property
    def _llm_type(self) -> str:
        return "qwen-llama-cpp"

    @property
    def _identifying_params(self) -> dict:
        """LangChain/Langfuse 가 trace 에 표시할 모델 식별 정보."""
        return {
            "model_name": self.gguf_filename_pattern,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "n_ctx": self.n_ctx,
        }

    # ─────────────────────────────────────────────
    # bind_tools — QwenChatModel 와 동일 시그니처
    # ─────────────────────────────────────────────
    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs,
    ):
        schemas = [convert_to_openai_tool(t) for t in tools]
        # _shared_llm 으로 기존 모델 인스턴스 공유 → 재로드 방지
        bound = self.__class__(
            _shared_llm=self._llm,
            model_id=self.model_id,
            gguf_filename_pattern=self.gguf_filename_pattern,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            chat_format=self.chat_format,
        )
        bound._tools_schema = schemas
        bound._tool_choice = tool_choice
        return bound

    # ─────────────────────────────────────────────
    # _generate
    # ─────────────────────────────────────────────
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        chat_messages = self._messages_to_openai_format(messages)
        tool_choice_param = self._build_tool_choice_param()
        # 호출 시점 override 허용: chat_model.invoke(messages, temperature=0.6) 같은 사용.
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_new_tokens))

        kwargs_for_call = dict(
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if self._tools_schema:
            kwargs_for_call["tools"] = self._tools_schema
            if tool_choice_param is not None:
                kwargs_for_call["tool_choice"] = tool_choice_param

        try:
            resp = self._llm.create_chat_completion(**kwargs_for_call)
        except Exception as e:
            logger.error(f"[QwenLlamaCpp] create_chat_completion 실패: {e}")
            raise

        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        raw_tool_calls = msg.get("tool_calls") or []

        # OpenAI 형식 → LangChain 형식
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except json.JSONDecodeError:
                logger.warning(f"[QwenLlamaCpp] tool_call 인자 JSON 파싱 실패: {args_str[:120]!r}")
                args = {}
            if not isinstance(args, dict):
                args = {}
            if name:
                tool_calls.append(
                    {
                        "name": name,
                        "args": args,
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }
                )

        ai = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ─────────────────────────────────────────────
    # _stream — token-단위 streaming (#9: 사용자 TTFT 향상용)
    # ─────────────────────────────────────────────
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """plain text streaming.

        chatml-function-calling 핸들러 (Qwen) 는 streaming 시 chunk 경계의 불완전
        UTF-8 byte 를 버려서 한글 글자가 누락됨 (#9 후속). → 그 경우엔 raw
        create_completion + 수동 ChatML 프롬프트로 우회 (byte 버퍼링됨).
        그 외 (chat_format=None 인 Gemma 4 등) 는 GGUF 의 임베디드 jinja
        템플릿을 쓰는 create_chat_completion(stream=True) 가 정상 동작.

        반드시 reset(): 직전 invoke()/generate 호출이 남긴 KV 캐시 + `_scores`
        버퍼와 새 prompt 가 어긋나면 llama-cpp 가 `IndexError: index N is out
        of bounds for axis 0 with size 12` 로 죽음. 각 streaming 시작 전에
        상태를 초기화해 prefill 부터 새로 함 (8GB GPU 에서 ~수백 ms 추가).
        """
        self._llm.reset()
        # 호출 시점 override 허용: chat_model.stream(messages, temperature=0.6) 같은 사용.
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_new_tokens))
        if self.chat_format == "chatml-function-calling":
            # Qwen 우회 경로 (UTF-8 안전)
            prompt = self._messages_to_chatml_prompt(messages)
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
            if stop:
                stop_tokens.extend(stop)
            for chunk in self._llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_tokens,
                stream=True,
            ):
                text = chunk["choices"][0].get("text")
                if text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        else:
            # 표준 경로 — GGUF 임베디드 템플릿 사용 (Gemma 4 등)
            chat_messages = self._messages_to_openai_format(messages)
            for chunk in self._llm.create_chat_completion(
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    # ─────────────────────────────────────────────
    # 헬퍼
    # ─────────────────────────────────────────────
    def _build_tool_choice_param(self) -> Any:
        """llama.cpp chatml-function-calling 포맷 호환 변환.

        지원: "auto" (기본), 특정 함수 dict {"type":"function","function":{"name":"X"}}.
        미지원: "required", "any" → 첫 도구 강제로 우회 (어차피 우리는 단일 도구만 강제하는 케이스가 대부분).
        """
        tc = self._tool_choice
        if tc is None or tc == "auto":
            return "auto"
        if tc in ("required", "any"):
            # llama.cpp 가 "any" 미지원 → 첫 도구 강제로 폴백
            if self._tools_schema:
                first_name = self._tools_schema[0]["function"]["name"]
                logger.info(
                    f"[QwenLlamaCpp] tool_choice='required' 미지원 → "
                    f"첫 도구 '{first_name}' 강제로 폴백"
                )
                return {"type": "function", "function": {"name": first_name}}
            return "auto"
        if isinstance(tc, str):
            return {"type": "function", "function": {"name": tc}}
        if isinstance(tc, dict):
            return tc
        return "auto"

    @staticmethod
    def _messages_to_chatml_prompt(messages: List[BaseMessage]) -> str:
        """ChatML 프롬프트 문자열로 변환 (Qwen2.5 포맷).

        streaming 답변 생성용 — 마지막 turn 은 항상 빈 assistant 헤더로 끝남.
        tool_call AIMessage / ToolMessage 는 사람이 읽을 수 있는 형태로 평탄화.
        """
        parts: List[str] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                parts.append(f"<|im_start|>system\n{m.content}<|im_end|>")
            elif isinstance(m, HumanMessage):
                parts.append(f"<|im_start|>user\n{m.content}<|im_end|>")
            elif isinstance(m, AIMessage):
                if m.tool_calls:
                    names = ", ".join(tc["name"] for tc in m.tool_calls)
                    parts.append(f"<|im_start|>assistant\n[{names} 도구 호출]<|im_end|>")
                else:
                    parts.append(f"<|im_start|>assistant\n{m.content}<|im_end|>")
            elif isinstance(m, ToolMessage):
                parts.append(f"<|im_start|>user\n[{m.name or '도구'} 결과]\n{m.content}<|im_end|>")
            else:
                parts.append(f"<|im_start|>user\n{getattr(m, 'content', '')}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _messages_to_openai_format(messages: List[BaseMessage]) -> List[dict]:
        out = []
        for m in messages:
            if isinstance(m, SystemMessage):
                out.append({"role": "system", "content": str(m.content)})
            elif isinstance(m, HumanMessage):
                out.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                if m.tool_calls:
                    out.append(
                        {
                            "role": "assistant",
                            "content": str(m.content) if m.content else "",
                            "tool_calls": [
                                {
                                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                                    },
                                }
                                for tc in m.tool_calls
                            ],
                        }
                    )
                else:
                    out.append({"role": "assistant", "content": str(m.content)})
            elif isinstance(m, ToolMessage):
                out.append(
                    {
                        "role": "tool",
                        "name": m.name or "",
                        "tool_call_id": getattr(m, "tool_call_id", "") or "",
                        "content": str(m.content),
                    }
                )
            else:
                out.append({"role": "user", "content": str(getattr(m, "content", ""))})
        return out


def build_qwen_llama_cpp_chat(**kwargs) -> QwenLlamaCppChatModel:
    """편의 팩토리. QwenChatModel 와 같은 패턴."""
    return QwenLlamaCppChatModel(**kwargs)
