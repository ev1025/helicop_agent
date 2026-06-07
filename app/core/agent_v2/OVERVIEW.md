# 🚀 [수리온 LLM] RAG 챗봇 시스템 아키텍처 개편 및 고도화 리포트

> **브랜치:** `jw/agent-refactoring` · **작업량:** main 대비 30 커밋 · **작업 기간:** 2026-05-08 ~ 2026-05-18
> **문서 링크:** [상세 변경 이력 (CHANGES.md)](https://www.google.com/search?q=./CHANGES.md) · [운영 가이드 (LANGFUSE_SETUP.md)](https://www.google.com/search?q=./LANGFUSE_SETUP.md)
> **스코프:** 시스템 아키텍처, 코드 구조, 인프라, 관측/평가 도구 등 **로직 중심의 기반 개선**
> *(※ RAG 검색 정확도 및 임베딩/LLM/리랭커 부품 성능 평가는 **AutoRAG 트랙에서 별도 진행**)*

기존 단일 에이전트 환경에서 발생하던 **구동 실패, 출력 형식 이탈, 극심한 응답 지연** 문제를 원천적으로 해결하기 위해 모델 백엔드, RAG 파이프라인, 관측 및 평가 시스템을 풀스택으로 재설계한 프로덕션 레벨 리팩토링 내역입니다.

---

## 1. 아키텍처 및 로직 전면 개편 (LangGraph & RAG)

### 1.1 멀티 에이전트(Multi-Agent) 아키텍처 전환

* **Native Function Calling 도입:** 불안정한 프롬프트 및 정규식 파싱 방식에서 벗어나 **Qwen Native Function Calling**을 채택. 형식 안정성을 확보하여 도구 호출 성공률을 60%에서 100%로 끌어올렸으며, BPE 노이즈를 완벽히 제거했습니다.
* **LangGraph 기반 역할 분담:** `Supervisor(라우팅) → Worker(답변 생성) → Critic(평가 및 피드백)` 구조의 멀티 에이전트 워크플로우를 구축했습니다.
* **실효성 있는 Feedback Loop (`#016-A`):** Critic이 단순 점수만 부여하는 대신 '구체적인 결함(Issues)'을 추출하여 Worker의 시스템 프롬프트에 주입함으로써 Retry의 품질을 극대화했습니다.

### 1.2 RAG 파이프라인 로직 최적화

* **문맥 단조 증가(Monotonic) 정렬:** 검색된 청크를 실제 문서 페이지(Page) 순서로 재정렬하여, LLM이 문서를 책 읽듯 자연스러운 흐름으로 인지하도록 개선했습니다.
* **Sentence-Window 로직:** 검색 매칭된 부모 청크의 앞뒤 페이지(±1)를 자동으로 회수하여 문맥 단절 현상을 해소했습니다.
* **도구 파라미터(Tool Args) 통제:** `top_k`, `reranker_top_k` 등의 검색 파라미터를 LLM이 임의 조작하지 못하도록 잠그고 Config 단일 소스로 관리합니다. LLM은 오직 `query` 결정에만 집중합니다.
* **시스템 프롬프트 분리:** 검색 쿼리 추출용(`SEARCH_QUERY_SYSTEM`)과 답변 작성용(`RESEARCHER_SYSTEM` / `PROCEDURE_SYSTEM`) 프롬프트를 분리하여 LLM의 역할 혼동을 방지했습니다.

---

## 2. 백엔드 성능 최적화 및 UX 극대화

### 2.1 초고속 GGUF 양자화 백엔드 도입

* **처리 속도 60배 향상:** 극도로 느렸던 기존 BNB 4bit 환경을 **GGUF Q4_K_M (llama.cpp)** 백엔드로 전면 교체하여 쿼리당 처리 시간을 304초에서 5.2초로 단축했습니다.
* **맞춤형 Wrapper Class 구현:** GGUF 모델을 LangGraph에 연동하기 위해 `QwenLlamaCppChatModel`을 직접 구현하였으며, `bind_tools`, `tool_choice`, `with_structured_output` 등 LangChain 표준 API를 모두 지원합니다.

### 2.2 유연한 멀티 백엔드 지원 아키텍처

* **다중 모델 템플릿 지원:** `qwen_llama_cpp_chat_model.py` 내에 `chat_format`을 파라미터화하여 Qwen, Gemma 등 상이한 템플릿을 단일 클래스에서 지원합니다.
* **즉각적인 백엔드 전환:** 환경변수 `CHAT_V2_BACKEND` (`gemma` / `qwen` / `bnb`)를 통해 시스템 재시작 없이 백엔드를 전환할 수 있습니다.
* `_stream` 메서드를 포맷별로 분기 처리하여, Qwen의 ChatML Function Calling 우회 경로와 표준 GGUF Jinja 경로를 동시에 지원합니다.

### 2.3 Token Streaming (점진적 출력) 구현

* 기존 SSE 일괄 전송 방식의 답답한 대기 시간을 해결하기 위해 한 글자씩 출력되는 Streaming UI(ChatGPT 방식)를 도입했습니다.
* 발견된 UTF-8 한글 다이트 깨짐 버그는 ChatML 프롬프트 직접 구성 및 `create_completion(stream=True)` 경로 우회를 통해 안정적으로 해결했습니다.

---

## 3. 관측성(Observability) 및 E2E 평가 자동화

### 3.1 Langfuse Trace 완벽 통합

* **비동기 추적 유실 완벽 해결:** FastAPI의 Threadpool 환경에서 OTEL `contextvars`가 비동기 경계를 넘지 못하는 문제를 해결하기 위해, `event_stream`을 `async def`로 전환하고 동기 LLM 호출을 `iterate_in_threadpool` 및 `anyio.to_thread.run_sync`로 감싸 Worker Thread로 컨텍스트가 안전하게 전파되도록 조치했습니다.
* **Funnel 단위 실시간 관측:** 질문 하나당 `agent-pipeline → Supervisor → Researcher → rag_search → child_to_parent → parent_rerank → sentence_window → 답변`으로 이어지는 전체 흐름을 Langfuse에서 한눈에 추적할 수 있습니다.
* 안전 폴백(Safe Fallback)을 적용하여 `.env` 환경변수가 없을 경우 모든 Trace 호출이 시스템에 영향을 주지 않도록(No-op) 처리했습니다.

### 3.2 스크립트 기반 CI/CD 평가 시스템

* Langfuse의 **Dataset / Experiment** 기능을 연동하여 자동화된 E2E 평가 환경을 구축했습니다.
* 단 한 줄의 스크립트(`dataset.run_experiment`) 실행으로 평가 데이터셋(`agent_v2_eval_7q`)에 대한 4개 지표(**Keyword Recall, Route Correctness, Latency, Answer Length**)를 자동 채점합니다.
* 코드나 시스템 로직 변경 시, Langfuse UI를 통해 Run 간의 **Side-by-Side 성능 비교**가 즉각적으로 가능합니다. (※ 라우팅 정확도 및 시스템 흐름 검증 목적)

---

## 4. 운영 안정성 및 트러블슈팅 (Troubleshooting)

* **Context Window 초과 (8192 토큰) 핫픽스:** `rag_search` 도구가 검색 결과를 `json.dumps(indent=2)`로 직렬화하면서 메타데이터로 인해 용량이 2배 이상 팽창하는 현상을 발견했습니다. LLM에는 핵심 텍스트(`[문서 N · p.PAGE]\n{내용}`)만 반환하도록 경량화(Lean)하여 에러를 차단했습니다.
* **검색 파라미터 변조 방지:** LLM이 임의로 `reranker_top_k=3`과 같은 값을 강제 삽입하여 검색 컨텍스트가 쪼그라드는 문제를 막기 위해, 도구 시그니처에서 해당 인자들을 완전히 제거했습니다.
* **API 호환성 수정:** Starlette 신버전 호환성에 맞춰 `pages.py`의 `TemplateResponse` 시그니처를 업데이트하여 `/chat` 라우팅의 500 에러를 해결했습니다.
* **레거시 클린업:** 더 이상 동작하지 않는 레거시 라우터(`chat.py`, `voice.py`)를 정리하고 불필요한 Git 추적 파일들을 제외했습니다.

---

## 5. 주요 기대 효과 (Impact)

* **👨‍💻 사용자 (UX):** 지루한 빈 화면 대기 없이, 질문 즉시 라우팅 및 검색 진행 상태(`🧭 Supervisor → 🔍 RAG 검색 중`)가 실시간 표시되며 답변이 점진적으로 스트리밍됩니다.
* **🛠️ 운영자 (Observability):** Langfuse 대시보드를 통해 모든 유저 요청의 파이프라인 전 단계를 마이크로초 단위로 펼쳐보고 병목 구간을 즉시 식별할 수 있습니다.
* **⚙️ 개발자 (DX):** 시스템 로직(프롬프트, 라우팅 룰, 인자 정책 등) 변경 시, CLI 스크립트 한 줄로 전체 E2E 파이프라인의 정량적 변화를 즉각 검증할 수 있습니다.

---

## 💡 핵심 인사이트 (Lessons Learned)

1. **제어권 분리의 원칙:** 검색 파라미터(`top_k` 등)를 LLM의 자율에 맡기면 시스템 일관성이 심각하게 훼손됩니다. **LLM은 오직 '쿼리 생성'만 담당하고, 파라미터 제어는 Config 기반으로 철저히 통제**하는 것이 최적의 관행입니다.
2. **프롬프트의 단일 책임 원칙 (SRP):** 검색 목적의 지시와 최종 답변 목적의 지시를 하나의 프롬프트에 혼재시키지 않고 분리했을 때, 응답 시간과 LLM의 지시 이행률이 눈에 띄게 향상되었습니다.
3. **데이터 직렬화의 함정:** RAG 도구 연동 시 무심코 사용하는 JSON 직렬화 포맷팅이 Context Window Bloat의 주원인이 될 수 있습니다. 에이전트 간 통신 페이로드는 최대한 경량화(Lean Text)해야 합니다.
4. **피드백 없는 재시도는 무의미:** Temperature=0 환경에서 Critic 에이전트의 단순 점수 부여를 통한 Retry는 동일한 오류를 반복할 뿐입니다. **구체적인 원인(Issues)을 텍스트로 피드백**해야만 실질적인 답변 개선이 이루어집니다.
5. **FastAPI + OTEL Context의 비동기 단절:** Async Generator와 OTEL Contextvars 조합은 Sync Threadpool 환경에서 자연스럽게 풀리지 않습니다. 추적 유실을 막으려면 정교한 비동기/동기 컨텍스트 래핑(`iterate_in_threadpool`)이 필수적입니다.