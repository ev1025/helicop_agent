"""
Qwen 전용 LangChain ChatModel.

LangChain 표준 ChatHuggingFace 가 tokenizer.apply_chat_template(tools=) 의
tools 파라미터를 전달하지 않아 Qwen native function calling 이 동작하지 않는
문제를 해결한다.

핵심:
  1. _generate() 에서 messages + tools 를 직접 chat template 에 주입.
  2. 모델 응답에서 <tool_call>...</tool_call> 태그를 정규식으로 추출.
  3. 추출된 tool call 을 AIMessage(tool_calls=[...]) 로 포장하여 반환.
  4. ToolMessage(role="tool") 가 들어오면 chat template 의 tool 메시지로 직렬화.

bind_tools(tools) 를 호출하면 LangChain BaseTool 리스트가 OpenAI-style
JSON Schema 로 변환되어 self._tools 에 저장된 뒤 _generate 호출 시 사용된다.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Iterator, List, Optional, Sequence

import torch
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)

# 도구 호출 패턴.  Qwen 의 BPE 토크나이저가 가끔 opening <tool_call> 토큰을
# 깨뜨리거나 누락시키는 현상이 관찰되어 opening tag 는 옵셔널로 둔다.
# 닫는 </tool_call> 만 있어도 매칭하고, 실패 시 fallback 으로 bare JSON 도 시도한다.
_TOOL_CALL_RE = re.compile(
    r"(?:<tool_call>)?\s*(\{[\s\S]*?\})\s*</tool_call>",
    re.DOTALL,
)
_BARE_TOOL_JSON_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{[\s\S]*?\}\s*\}",
    re.DOTALL,
)

# tool_call 주변 BPE 잔여 패턴: 짧은 영문/공백/단독 < 등으로만 이루어진 라인.
# 예: "ronics", "<", "oolnics", "ool_call" 같은 깨진 token.
_BPE_NOISE_RE = re.compile(r"^[<>\s]*[a-z_]{1,12}[<>\s]*$", re.IGNORECASE | re.MULTILINE)


def _strip_bpe_noise(text: str) -> str:
    """tool_call 매칭 후 남은 BPE 잔여 노이즈 라인을 제거."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _BPE_NOISE_RE.fullmatch(stripped):
            # 짧은 알파벳-only 라인은 BPE 잔여로 간주하여 버림
            continue
        lines.append(line)
    return "\n".join(lines).strip()


class QwenChatModel(BaseChatModel):
    """Qwen 모델을 LangChain ChatModel 인터페이스로 노출.

    내부적으로 transformers 의 model.generate() 와 tokenizer.apply_chat_template(tools=)
    을 직접 사용한다.
    """

    model_id: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    max_new_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=0.85)
    repetition_penalty: float = Field(default=1.1)
    do_sample: bool = Field(default=True)

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _tools_schema: Optional[List[dict]] = PrivateAttr(default=None)
    # tool_choice 값 (None | "auto" | "required" | {"type": "function", "function": {"name": "X"}})
    _tool_choice: Any = PrivateAttr(default=None)

    def __init__(self, *, tokenizer, model, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._model = model

    # ─────────────────────────────────────────────────────
    # LangChain BaseChatModel 추상 메서드
    # ─────────────────────────────────────────────────────
    @property
    def _llm_type(self) -> str:
        return "qwen-chat-native"

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs,
    ):
        """LangChain BaseTool 리스트를 OpenAI-style JSON Schema 로 변환해 보관.

        Args:
            tools: 바인딩할 LangChain BaseTool 리스트.
            tool_choice: 도구 선택 강제. 다음 값들을 지원한다.
                - None / "auto": 모델 자율 (기본값)
                - "required" / "any": 반드시 도구 중 하나는 호출
                - {"type": "function", "function": {"name": "X"}}: 특정 도구 강제
                - "X" (문자열): 특정 도구 이름 강제 (단축형)

        Returns:
            새 QwenChatModel 인스턴스 (불변성 유지). _tools_schema 와
            _tool_choice 가 채워진 상태.
        """
        schemas = [convert_to_openai_tool(t) for t in tools]
        bound = self.__class__(
            tokenizer=self._tokenizer,
            model=self._model,
            model_id=self.model_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
        )
        bound._tools_schema = schemas
        bound._tool_choice = tool_choice
        return bound

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        # 1) LangChain Message → Qwen chat 포맷 (dict 리스트)
        chat_messages = self._messages_to_chat_format(messages)

        # 2) tool_choice 가 강제 모드면 시스템 프롬프트에 지시 주입
        chat_messages = self._inject_tool_choice_instruction(chat_messages)

        # 3) chat template 적용 (tools 포함)
        prompt_text = self._tokenizer.apply_chat_template(
            chat_messages,
            tools=self._tools_schema,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 4) tool_choice 가 특정 도구 강제면 assistant prefix 로 <tool_call> 박기
        prefix = self._build_assistant_prefix()
        if prefix:
            prompt_text += prefix

        # 5) 토큰화 + generate
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_response = self._tokenizer.decode(new_tokens, skip_special_tokens=False)

        # 6) prefix 가 있었으면 응답 앞에 다시 붙여서 파싱 (모델은 prefix 다음부터 생성하므로)
        if prefix:
            raw_response = prefix + raw_response

        # 7) <tool_call> 태그 파싱
        ai_message = self._parse_response_to_ai_message(raw_response)

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ─────────────────────────────────────────────────────
    # tool_choice 처리
    # ─────────────────────────────────────────────────────
    def _inject_tool_choice_instruction(self, chat_messages: List[dict]) -> List[dict]:
        """tool_choice 가 강제 모드면 시스템 메시지에 지시문을 추가한다."""
        if self._tool_choice in (None, "auto"):
            return chat_messages

        forced_name = self._extract_forced_tool_name()
        if forced_name:
            instruction = (
                f"\n\n[도구 호출 강제] 이번 응답에서는 반드시 `{forced_name}` 도구를 "
                f"<tool_call>{{\"name\": \"{forced_name}\", \"arguments\": {{...}}}}</tool_call> "
                f"형식으로 호출해야 한다. 일반 텍스트 답변 금지."
            )
        else:
            # tool_choice == "required" / "any"
            tool_names = [t["function"]["name"] for t in (self._tools_schema or [])]
            instruction = (
                f"\n\n[도구 호출 강제] 이번 응답에서는 반드시 다음 도구들 중 하나를 "
                f"<tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call> 형식으로 "
                f"호출해야 한다. 사용 가능 도구: {tool_names}. 일반 텍스트 답변 금지."
            )

        new_messages = [dict(m) for m in chat_messages]
        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0]["content"] = str(new_messages[0].get("content", "")) + instruction
        else:
            new_messages.insert(0, {"role": "system", "content": instruction.lstrip()})
        return new_messages

    def _build_assistant_prefix(self) -> str:
        """tool_choice 가 특정 도구 지정이면 assistant 응답을 강제로 시작시키는 prefix.

        예: tool_choice={"type":"function","function":{"name":"final_answer"}}
            → '<tool_call>\n{"name": "final_answer", "arguments": '
        """
        forced_name = self._extract_forced_tool_name()
        if not forced_name:
            return ""
        return f'<tool_call>\n{{"name": "{forced_name}", "arguments": '

    def _extract_forced_tool_name(self) -> Optional[str]:
        """tool_choice 에서 특정 도구 이름을 추출 (있으면)."""
        tc = self._tool_choice
        if tc is None or tc in ("auto", "required", "any"):
            return None
        if isinstance(tc, str):
            return tc
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                return fn["name"]
        return None

    # ─────────────────────────────────────────────────────
    # 헬퍼
    # ─────────────────────────────────────────────────────
    @staticmethod
    def _messages_to_chat_format(messages: List[BaseMessage]) -> List[dict]:
        """LangChain BaseMessage 들을 Qwen chat template 형식(dict)으로 변환."""
        out = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                out.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                out.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # assistant 가 도구 호출한 턴은 chat template 에서 tool_calls 키를 사용
                    out.append(
                        {
                            "role": "assistant",
                            "content": str(msg.content) if msg.content else "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(
                                            tc.get("args", {}), ensure_ascii=False
                                        ),
                                    },
                                }
                                for tc in msg.tool_calls
                            ],
                        }
                    )
                else:
                    out.append({"role": "assistant", "content": str(msg.content)})
            elif isinstance(msg, ToolMessage):
                # Qwen chat template 은 role="tool" 메시지를 tool_call_id 와 함께 받음
                out.append(
                    {
                        "role": "tool",
                        "name": msg.name or "",
                        "content": str(msg.content),
                    }
                )
            else:
                # 알 수 없는 메시지 타입은 user 로 다운캐스트
                out.append({"role": "user", "content": str(getattr(msg, "content", ""))})
        return out

    @classmethod
    def _parse_response_to_ai_message(cls, raw: str) -> AIMessage:
        """모델 raw 응답에서 <tool_call> JSON 을 추출 → AIMessage 생성.

        파싱 우선순위:
          1) <tool_call>...</tool_call>  (정상 케이스)
          2) ...</tool_call>  (opening tag 누락 케이스 — Qwen BPE 토크나이저가
             가끔 opening 토큰을 깨뜨림)
          3) bare JSON {"name": ..., "arguments": {...}}  (마지막 fallback)

        파싱 못 하면 raw 텍스트를 그대로 content 로 반환 (tool_calls=[]).
        """
        cleaned = raw.replace("<|im_end|>", "").strip()
        logger.debug(f"[QwenChatModel] raw response (200자): {cleaned[:200]!r}")

        tool_calls = []
        matched_spans: list[tuple[int, int]] = []

        # 1단계 + 2단계: <tool_call>...</tool_call> 또는 ...</tool_call>
        for m in _TOOL_CALL_RE.finditer(cleaned):
            payload_str = m.group(1)
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                logger.warning(
                    f"[QwenChatModel] tool_call JSON 파싱 실패: {payload_str[:200]!r}"
                )
                continue
            name = payload.get("name")
            args = payload.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            if name:
                tool_calls.append(
                    {
                        "name": name,
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }
                )
                matched_spans.append(m.span())

        # 3단계: 위에서 못 찾았으면 bare JSON fallback
        if not tool_calls:
            for m in _BARE_TOOL_JSON_RE.finditer(cleaned):
                try:
                    payload = json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
                name = payload.get("name")
                args = payload.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                if name:
                    tool_calls.append(
                        {
                            "name": name,
                            "args": args,
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "tool_call",
                        }
                    )
                    matched_spans.append(m.span())

        # 매칭된 부분을 cleaned 에서 제거하여 content 텍스트 산출
        if matched_spans:
            content_parts = []
            cursor = 0
            for s, e in sorted(matched_spans):
                content_parts.append(cleaned[cursor:s])
                cursor = e
            content_parts.append(cleaned[cursor:])
            content_text = "".join(content_parts).strip()
            # tool_call 주변 BPE 잔여 토큰 정리:
            # 'ronics', '< ', 'oolnics' 같은 소수 알파벳-only 짧은 노이즈를 제거.
            # tool_calls 가 있을 때만 적용 (일반 텍스트는 보호).
            content_text = _strip_bpe_noise(content_text)
        else:
            content_text = cleaned

        return AIMessage(content=content_text, tool_calls=tool_calls)
