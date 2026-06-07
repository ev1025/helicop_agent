"""
Langfuse trace 통합 helper.

환경변수가 설정되지 않으면 no-op 으로 동작 — 운영 코드에 그대로 끼워두고 키만 설정하면
자동 trace 시작.

필수 env vars (cloud 또는 self-host 공통):
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=https://cloud.langfuse.com   # 또는 self-host URL

요청 1개 = trace 1개 로 묶는 패턴 (streaming generator 에서도 안전):
  trace_ctx, trace = start_request_trace("agent-pipeline", input=message)
  try:
      result = chain.invoke(x, config={"callbacks": fresh_callbacks(trace_ctx)})  # 매 호출 새 handler
      ...
  finally:
      trace.end(output=answer)
      flush()

핵심: trace_id 는 공유하되 LangChain `.invoke()`/`.stream()` 마다 **새 CallbackHandler 인스턴스**
를 만들어야 함 (handler 재사용하면 첫 run 만 trace 에 묶이고 나머지는 별도 trace 로 새어나감).
OTEL context-manager (start_as_current_span) 는 FastAPI threadpool generator 에서 yield 경계를
못 넘어가므로 사용하지 않음 — 명시적 trace_id + parent_span_id 만 전달.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Iterator, List

logger = logging.getLogger(__name__)

_handler = None
_initialized = False

# repo root = app/core/agent_v2/langfuse_handler.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_env() -> None:
    """repo 루트의 .env 를 명시적으로 로드 (서버는 main.py 가 이미 했어도 무해)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fallback: 기본 탐색


def _init() -> None:
    """최초 1회만 init. 키 없으면 _handler 는 None 으로 유지."""
    global _handler, _initialized
    if _initialized:
        return
    _initialized = True

    _load_env()

    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")

    if not (pub and sec):
        logger.info("[langfuse] env vars 미설정 → trace 비활성화 (no-op)")
        return

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Langfuse client init (env vars 자동 인식)
        Langfuse(public_key=pub, secret_key=sec, host=host or "https://cloud.langfuse.com")
        _handler = CallbackHandler()  # flat trace 용 (smoke 스크립트 등)
        logger.info(f"[langfuse] 활성화 host={host or 'cloud.langfuse.com'}")
    except Exception as e:
        logger.warning(f"[langfuse] 초기화 실패: {e}")
        _handler = None


def is_enabled() -> bool:
    _init()
    return _handler is not None


def flush() -> None:
    """프로세스 종료 / 요청 종료 시 trace 큐 강제 flush."""
    if not is_enabled():
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as e:
        logger.warning(f"[langfuse] flush 실패: {e}")


# ─────────────────────────────────────────────
# 요청 1개 = trace 1개 (async event_stream 안에서 사용)
#
# 핵심: async generator 는 event loop 에서 돌아 OTEL contextvars 가 yield/await 경계를
# 유지함 → request_span() 으로 root span 을 ambient OTEL context 로 세우면, 그 안에서
# (threadpool 로 감싸 호출하는) LangChain 의 CallbackHandler 가 자동으로 그 span 의 자식이 됨.
# sync generator + threadpool 조합에서는 매 .next() 가 새 context copy 라 이게 안 됐음 (#10 참고).
# ─────────────────────────────────────────────
@contextlib.contextmanager
def request_span(name: str, *, input: Any = None, metadata: Any = None) -> Iterator[Any]:
    """요청 1개를 감싸는 root span (ambient OTEL context). 비활성화 시 None yield.

    반드시 async event_stream 안에서 `with request_span(...) as span:` 으로 사용 —
    그 안에서 LLM 호출을 (sync 면 iterate_in_threadpool/run_sync 로 감싸서) 하면
    한 trace 에 묶임. span 은 LangfuseSpan (또는 None).
    """
    if not is_enabled():
        yield None
        return
    try:
        from langfuse import get_client
        client = get_client()
    except Exception:
        yield None
        return
    with client.start_as_current_observation(
        name=name, as_type="span", input=input, metadata=metadata
    ) as span:
        try:
            span.set_trace_io(input=input)
        except Exception:
            pass
        yield span


@contextlib.contextmanager
def observe_span(name: str, *, input: Any = None, metadata: Any = None) -> Iterator[Any]:
    """ambient OTEL context (있으면) 아래에 자식 span 1개 생성. 비활성화 시 None yield.

    LangChain 컴포넌트가 아닌 코드 블록을 trace 에 노출 (with 문 가능한 경우):
        with observe_span("rag.child_search", input=query) as s:
            results = ...
            if s: s.update(output={"n": len(results)})
    """
    if not is_enabled():
        yield None
        return
    try:
        from langfuse import get_client
        with get_client().start_as_current_observation(
            name=name, as_type="span", input=input, metadata=metadata
        ) as span:
            yield span
    except Exception:
        yield None


def maybe_span(name: str, *, input: Any = None, metadata: Any = None):
    """현재 OTEL context 아래에 span 생성 후 반환 (스스로 current 로는 안 만듦 — 순차 단계용).
    반드시 end_span() 으로 닫아야 함. 비활성화 시 None.

    with 문이 어색한 곳 (재인덴트 피하고 싶을 때):
        s = maybe_span("rag.parent_fetch", input={"n_child": n})
        ... 작업 ...
        end_span(s, output={"n_parent": m})
    """
    if not is_enabled():
        return None
    try:
        from langfuse import get_client
        return get_client().start_observation(
            name=name, as_type="span", input=input, metadata=metadata
        )
    except Exception:
        return None


def end_span(span, *, output: Any = None, metadata: Any = None) -> None:
    """maybe_span() 으로 만든 span 마무리. span 이 None 이면 no-op."""
    if span is None:
        return
    try:
        if output is not None or metadata is not None:
            span.update(output=output, metadata=metadata)
        span.end()
    except Exception:
        pass


def new_callbacks() -> List:
    """LangChain `.invoke()` / `.stream()` config 용 — 매 호출 새 CallbackHandler 1개.

    handler 는 ambient OTEL context (request_span 이 세운 span) 에 자동으로 자식으로 붙음.
    Langfuse 비활성화 시 []. (handler 재사용하면 root run 상태가 꼬이므로 매번 새로.)
    """
    _init()
    if _handler is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []


def get_callbacks() -> List:
    """하위호환 별칭 (CLI 스크립트용). new_callbacks() 와 동일."""
    return new_callbacks()
