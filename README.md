# helicop_agent — 수리온 헬리콥터 표준교재 AI 튜터

> **헬리콥터 표준교재 PDF 1권을 RAG 인덱싱하고, FastAPI + GGUF(llama.cpp) 백엔드로 음성/텍스트 챗봇을 띄우는 단일 레포.**
> 운영 핵심은 `app/api/routes/chat_v2.py` 한 파일 — Supervisor 분기, RAG 강제 호출, Langfuse trace, SSE streaming 까지 포함.

---

## 0. 산출물 한눈 보기

| 산출물 | 위치 | 비고 |
|---|---|---|
| **운영 챗봇 서버** | `main.py` + `app/` | FastAPI, SSE streaming, GGUF 백엔드 |
| **에이전트 파이프라인** | `app/api/routes/chat_v2.py` | Supervisor → RAG 강제 → 답변 streaming |
| **GGUF ChatModel 래퍼** | `app/core/agent_v2/qwen_llama_cpp_chat_model.py` | LangChain `BaseChatModel` 호환, `bind_tools`/`with_structured_output` 지원 |
| **RAG 도구** | `app/core/agent_v2/tools.py` + `app/services/rag_service.py` | Parent-Child Retrieval, `top_k` 잠금 |
| **임베딩/벡터 DB** | `app/core/{embeddings,models,vector_store}.py` + `chroma_db_new/` | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| **음성 인터페이스** | `app/api/routes/voice.py` + `app/services/{audio,stt}_service.py` | STT(Whisper-small) → chat_v2 위임, 브라우저 TTS |
| **관측성** | `app/core/agent_v2/langfuse_handler.py` + `LANGFUSE_SETUP.md` | env 없으면 no-op |
| **변경 이력 / 인사이트** | `app/core/agent_v2/{CHANGES,OVERVIEW}.md` | Qwen native function calling 도입 등 |
| **OCR 벤치마크 (8종)** | `scripts/ocr_poc/` + `results/ocr_poc/` + `OCR_BENCHMARK.md` | Surya / Tesseract / MinerU 등 정량 비교 |
| **AutoRAG 실험 트랙** | `autorag/` + `autorag_bench/` + `AUTORAG_ANALYSIS_REPORT.md` | 별도 트랙. 운영 swap-in 은 `autorag/adapter.py` |
| **원본 PDF** | `data/pdfs/조종사표준교재(비행이론_헬리콥터).pdf` | 229페이지, 베이스 인덱스 소스 |

---

## 1. 빠른 실행

전제: 윈도우 + RTX 4060 + `.venv/` 안에 `torch 2.6+cu124`, `llama-cpp-python 0.3.22`, `webrtcvad-wheels` 등 설치돼 있다고 가정.

```powershell
# 0) (선택) HTTP 모드로 띄울 때만 — Playwright 등 자체서명 인증서 거부 환경 회피
#    HTTPS 로 띄우려면 ssl_certificates.disabled → ssl_certificates 로 이름 복원
cd "C:\Users\eg287\OneDrive\바탕 화면\project\LLM\helicop_agent"

# 1) venv 활성화
.\.venv\Scripts\Activate.ps1

# 2) 백엔드 LLM 선택 (gemma 권장 — Qwen 대비 ~38% 빠름)
$env:CHAT_V2_BACKEND = "gemma"     # 또는 "qwen"

# 3) 서버 띄우기 (config/server.json 의 port 사용, 기본 8001)
python main.py

# 4) 헬스체크
curl http://localhost:8001/health
# → {"status":"healthy","llm_loaded":false,"rag_loaded":true,"device":"cuda",...}

# 5) 브라우저
# http://localhost:8001/        모드 선택
# http://localhost:8001/chat    텍스트 채팅
# http://localhost:8001/speak   음성 채팅 (마이크)
```

첫 채팅 호출 시 GGUF 모델이 lazy load (~2초). 답변 시간은 질문 종류·길이에 따라 3~12초.

---

## 2. 아키텍처 한눈

```
[클라이언트 HTTP GET /chat/v2/stream?message=...]
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ chat_v2.stream_answer  (SSE async generator)                  │
│                                                               │
│  ① Rule-based pre-router (정규식, LLM 호출 X)                  │
│     인사·짧은 응답 → smalltalk 즉시 확정                       │
│                                                               │
│  ② (애매한 경우) Supervisor LLM                                │
│     with_structured_output(RouteDecision) → rag | smalltalk   │
│                                                               │
│  ── smalltalk → temp=0.6 token streaming                       │
│                                                               │
│  ── rag                                                        │
│     ③ tool_choice='rag_search' 강제 → query 추출              │
│     ④ Parent-Child Retrieval (child 60 → parent 43 → 7개)     │
│     ⑤ <context>+<question> XML → temp=0.0 token streaming     │
│        답변에 [p.PAGE] 자연스러운 인용                          │
│                                                               │
│  ⓦ SSE :ping heartbeat (5초 주기) 으로 keep-alive             │
│  ⓛ Langfuse `agent-pipeline` trace 로 전 단계 기록            │
└───────────────────────────────────────────────────────────────┘
```

세부 인사이트는 [app/core/agent_v2/OVERVIEW.md](app/core/agent_v2/OVERVIEW.md).

---

## 3. 모델 · 검색 설정 (현 운영값)

| 항목 | 값 | 비고 |
|---|---|---|
| LLM (기본) | `unsloth/gemma-4-E4B-it-GGUF` Q4_K_M | n_ctx=8192, temp=0.0 / 0.6 |
| LLM (대안) | `bartowski/Qwen2.5-7B-Instruct-GGUF` Q4_K_M | env `CHAT_V2_BACKEND=qwen` |
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | 한국어 특화 |
| 벡터 DB | Chroma (Parent-Child) | 1025 child / 256 parent |
| Chunk | child 300자 / parent 1500자 | overlap 30 / 150 |
| Vector top_k | 30 (내부 ×2=60 child) | LLM 변조 불가 — config 고정 |
| MAX_CONTEXT_LENGTH | 8000자 | rag_search 도구가 cap |
| Reranker | OFF (`use_reranker:false`) | `bge-reranker-v2-m3-ko` 옵션 보존 |
| Hybrid (BM25+vector) | OFF (`use_hybrid_search:false`) | `rank_bm25` 코드 보존 |
| Multi-Query | ON (자동 판단) | `services/multi_query.py` |
| Reranker / Hybrid 성능 | AutoRAG 트랙에서 별도 평가 | [autorag_bench/README.md](autorag_bench/README.md) |

---

## 4. 디렉토리 맵 (운영 vs 평가)

### 운영 (실제 서비스)
```
main.py                          서버 entry
app/api/main.py                  FastAPI 앱, lifespan 에서 임베딩+Chroma 로드
app/api/routes/
  chat_v2.py                     ★ 핵심 SSE 라우트
  voice.py                       STT → chat_v2 위임
  pages.py                       /, /chat, /speak, /mp3 HTML
app/core/agent_v2/               GGUF wrapper + Langfuse handler + tools
app/services/                    rag_service, hybrid_retrieval, multi_query, stt, audio
app/templates/                   Jinja2 HTML (select_mode / chat / speak / mp3)
config/                          server, models, rag, prompts, audio, logging
ssl_certificates(.disabled)/     HTTPS 인증서 (있으면 자동 HTTPS, 없으면 HTTP)
data/pdfs/                       베이스 PDF
chroma_db_new/                   인덱싱된 벡터 DB
```

### 평가 / 실험 (운영 swap-in 안 됨)
```
autorag/                         AutoRAG 평가 + adapter.py (명시적 호출 시 운영 rag_search 교체)
autorag_bench/                   AutoRAG 평가 자료 + 결과
scripts/agent_v2_poc/            agent_v2 PoC 스크립트 (검증·벤치)
scripts/ocr_poc/                 OCR 모델 8종 벤치마크 파이프라인
results/ocr_poc/                 OCR 결과 (dashboard.html 등)
```

### 문서 / 보고서
```
OCR_BENCHMARK.md                 OCR 정량 비교 — Phase 1+2 완료
AUTORAG_ANALYSIS_REPORT.md       AutoRAG 분석
AUTORAG_SETUP.md                 AutoRAG 셋업
app/core/agent_v2/OVERVIEW.md    아키텍처 개편 보고서
app/core/agent_v2/CHANGES.md     변경 이력
app/core/agent_v2/LANGFUSE_SETUP.md   Langfuse 활성화 가이드
수리온_AI튜터_분석정리.md          전체 배경 분석
docs/개발계획서_발췌/             원본 개발계획서
```

---

## 5. 환경변수

| 변수 | 기본 | 용도 |
|---|---|---|
| `CHAT_V2_BACKEND` | `gemma` | `gemma` / `qwen` / `gguf` (qwen 동의어) |
| `LANGFUSE_PUBLIC_KEY` | (없음) | 없으면 trace 비활성화 (no-op) |
| `LANGFUSE_SECRET_KEY` | (없음) | 위와 페어 |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | self-host 가능 |

`.env` 파일 자동 로드 ([app/api/main.py](app/api/main.py)).

---

## 6. 음성 인터페이스

- 라우트: `POST /speak/stream` (음성 업로드 → SSE 답변), `POST /voice_chat/` (JSON)
- STT: `app/services/my_whisper_small/` (HuggingFace Whisper small) — **현재 가중치 누락 상태**. 음성 모드 사용하려면 weights 다운 필요.
- TTS: **클라이언트 측 SpeechSynthesis API** (서버는 텍스트만 송출)

---

## 7. 평가 트랙

### OCR ([OCR_BENCHMARK.md](OCR_BENCHMARK.md))
- 8개 OCR 모델 (Surya / Tesseract / EasyOCR / PaddleOCR / TrOCR / Donut / Marker / MinerU) × 21장 샘플
- 정량 1등: **Surya** (PyPDF 대비 1.28x), 표·그림에서 +47%
- 정성 (CER) 측정은 Phase 3 대기

### AutoRAG ([autorag_bench/README.md](autorag_bench/README.md))
- BM25(ko_kiwi) / Vector / Hybrid / Reranker 평가
- 운영 swap-in 은 `autorag/adapter.py` 의 `install_into_agent_v2()` 명시 호출 필요 — 현재 startup 자동 호출 X

---

## 8. 알려진 함정

1. **Gemma 4 GGUF + `tool_choice="auto"`** → `llama-cpp-python` 이 native function calling 토큰을 OpenAI `tool_calls` 로 변환 못 함. **반드시 함수명 강제** (`tool_choice="rag_search"` 등). chat_v2 는 이 패턴 사용.
2. **`torch` CPU 빌드 + `llama-cpp-python` CUDA 빌드** 미스매치 시 `llama.dll` 의존성(`cudart64_*.dll`) 깨짐. `.venv` 의 torch 는 **반드시 `+cu124` 빌드**여야 함.
3. **HTTPS 자체서명 인증서 + Playwright** → `NET::ERR_CERT_AUTHORITY_INVALID` 로 거부. 테스트 시 `ssl_certificates` 폴더를 `.disabled` 로 일시 변경해 HTTP 모드로 띄움.
4. **8000 포트가 다른 프로젝트와 겹침** (예: cctv_memory). 본 레포는 `config/server.json` 의 `port: 8001` 기본.
5. **Whisper 모델 가중치 부재** → `app/services/stt_service.py` import 시 OSError 한 번 찍힘. 서버 부팅엔 영향 X (음성 모드만 못 씀).
6. **`use_reranker:false` 상태에선 `top_scores` 가 더미 `1.0` 배열** — 의미 없는 값.

---

## 9. 변경 이력 (이 레포 통합 후)

- `app/api/routes/chat_v2.py` — Supervisor + Rule-based pre-router + 2-step pipeline + SSE :ping heartbeat + 자연스러운 [p.PAGE] 인용
- `app/core/agent_v2/qwen_llama_cpp_chat_model.py` — 호출 시점 `temperature` override 지원
- `app/core/agent_v2/tools.py` — `final_answer` 더미 도구 제거 (`ALL_TOOLS = [rag_search]`)
- 가짜 멀티에이전트(`multi_graph`, `agents/`, `qwen_chat_model`, `runner`) 및 legacy(`orchestrator`, `tool_executor`, `tool_prompt`, `conversation`, `tools/`, `chat.py`) 제거
- 자세한 PoC 시기 변경은 [CHANGES.md](app/core/agent_v2/CHANGES.md)

---

## 10. 라이선스 / 데이터

- 코드: 사내 사용
- 원본 PDF: 헬리콥터 표준교재 (조종사 표준 교재 — 비행이론·헬리콥터). 외부 배포 금지.
- GGUF 모델: 각 모델 라이선스(Gemma TOU, Qwen2.5 Apache-2.0) 따름.
