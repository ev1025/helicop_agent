# helicop_agent — Claude 작업 가이드

이 레포는 **수리온 헬리콥터 표준교재(229p PDF)** 를 인덱싱하고 RAG 챗봇으로 서빙하는 단일 레포입니다. 운영 핵심은 `app/api/routes/chat_v2.py` 하나, 백엔드는 Gemma 4 E4B GGUF (Q4_K_M) + Chroma + Langfuse trace.

전체 산출물 맵은 [README.md](README.md) 참고. 이 문서는 **Claude 가 작업할 때 지켜야 할 약속**만 담습니다.

---

## 0. 절대 약속

### Git 정책
- 이 레포 원격은 **GitHub** (`ev1025/helicop_agent`).

### 디렉토리 / 환경
- 파이썬 환경은 **반드시 `.venv` 사용** (`.\.venv\Scripts\Activate.ps1`). 시스템 Python 호출 금지.
- `torch` 는 **반드시 `+cu124` 빌드**. `+cpu` 빌드면 `llama-cpp-python` 의 CUDA DLL 의존성 깨짐.
- 검색 결과 / chroma_db / `.venv` / GGUF 캐시(`~/.cache/huggingface/`) **건드리지 말 것** — 인덱싱이 사라지면 재구축에 분 단위 걸림.
- 다른 프로젝트(`cctv_memory`, `hailo_*`, `t2i`, `helicop_agent` 외) 디렉토리는 본 세션에서 다루지 말 것.

### 프로세스 종료
- `python` 프로세스 kill 전에 항상 **본인 프로세스인지 확인** (`Get-CimInstance Win32_Process -Filter "ProcessId=N" | Select CommandLine`). 다른 프로젝트 서버 죽이지 말 것.
- 본 레포 서버는 `config/server.json` 의 `port: 8001` 사용 (8000 은 다른 프로젝트와 충돌).

### 메모리·캐시
- HuggingFace 캐시 위치: `C:\Users\eg287\.cache\huggingface\hub`. GGUF 4.97GB 짜리가 들어있음. 절대 지우지 말 것.

---

## 1. 자주 하는 작업 명령

### 서버 띄우기 (HTTP 모드)
```powershell
cd "C:\Users\eg287\OneDrive\바탕 화면\project\LLM\helicop_agent"
.\.venv\Scripts\Activate.ps1
$env:CHAT_V2_BACKEND = "gemma"      # 또는 "qwen"
python main.py
```
- HTTPS 로 띄우려면 `ssl_certificates.disabled` → `ssl_certificates` 로 이름 복원.
- 부팅 로그는 `server_boot.log` 로 tee 돼있음 (백그라운드 실행 시).
- `until curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8001/health 2>/dev/null | grep -q "200"; do sleep 2; done` 로 ready 폴링.

### 채팅 동작 확인
```powershell
curl.exe -s --max-time 60 "http://localhost:8001/chat/v2/stream?message=%EB%B2%A0%EB%A5%B4%EB%88%84%EC%9D%B4"
```
SSE 응답: `data: {"type":"info"|"rag_info"|"text"|"done"|"error", ...}\n\n` + `:ping\n\n` (heartbeat).

### 본인 python 프로세스만 종료
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*helicop_agent*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## 2. 코드 작업 시 지킬 룰

### 운영 라우트 (`chat_v2.py`) 수정 원칙
- `tool_choice="auto"` 사용 금지 — Gemma 4 GGUF + `auto` 조합은 native function calling 토큰을 `tool_calls` JSON 으로 변환 못 함. **항상 함수명 강제** (`tool_choice="rag_search"` 또는 Pydantic 스키마 강제).
- 답변 system prompt 와 user message 에 **같은 지시 중복 금지** (Gemma chat template 은 system/user 한 turn 으로 합침 → 토큰 낭비).
- 새 LLM/RAG 호출 추가 시 `_drain_with_ping(task)` 로 감쌀 것 — 프록시 keep-alive timeout 방어.

### Prompt 변경 시
- `chat_v2.py` 상단의 4개 상수 (`SUPERVISOR_SYSTEM`, `SMALLTALK_SYSTEM`, `SEARCH_QUERY_SYSTEM`, `RESEARCHER_SYSTEM`) 가 유일한 운영 prompt.
- 인용 형식은 `[p.PAGE]` (대괄호). 매 문장 강제하지 말고 단락 끝/핵심 뒤에 자연스럽게 — RESEARCHER_SYSTEM 의 약속.
- RAG 답변 user message 는 `<context>` / `<question>` XML 태그 사용.

### RAG 검색 설정 변경
- `config/rag.json` 한 곳에만 — `top_k`, `reranker_top_k`, `max_context_length` 등.
- LLM 이 검색 파라미터를 임의 조작 못 하게 `rag_search` tool 인자는 `query` 하나뿐 — **이 정책 깨지 말 것**.
- Reranker / BM25 / Hybrid 켤 거면 `use_reranker` / `use_hybrid_search` 둘 다 켜야 동작 (`config/models.json` 의 `reranker_model: null` 도 함께 변경).

### LangChain ChatModel 호출
- 구조화 출력은 `chat_model.with_structured_output(Schema)` — 내부적으로 `bind_tools` + 함수 강제 사용해서 Gemma 에서도 정상 작동.
- 도구 호출은 `chat_model.bind_tools([tool], tool_choice="tool_name")` — `auto`/`required`/`any` 금지.
- `temperature` 호출 시 override: `chat_model.stream(messages, temperature=0.6)` — wrapper 가 kwargs 에서 받음.

---

## 3. 검증 흐름 (변경 후 항상)

1. **syntax**
   ```powershell
   Get-ChildItem -Path app, main.py -Recurse -Filter *.py |
     Where-Object { $_.FullName -notmatch '__pycache__' } |
     ForEach-Object { python -m py_compile $_.FullName }
   ```
2. **서버 부팅 + `/health`** — `device:cuda`, `rag_loaded:true` 확인.
3. **3 시나리오 회귀 테스트**
   - `안녕` → rule-based smalltalk (LLM 호출 없이 ~3초)
   - `VRS가 뭐야` → supervisor → rag (약어 → rag)
   - `자동회전 절차 알려줘` → supervisor → rag (절차 답변 + [p.X] 인용)
4. Langfuse trace 확인 (`cloud.langfuse.com`) — `agent-pipeline` trace 의 `rag.child_to_parent`, `rag.parent_rerank` span 안에 청크 미리보기 보임.

---

## 4. 자주 묻는 함정

| 증상 | 원인 | 처방 |
|---|---|---|
| 503 "모델 로드 실패" | torch +cpu 빌드라 `llama.dll` 의존성 미충족 | `pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu124 torch torchvision` |
| Chroma 새로 인덱싱이 시작됨 | `chroma_db_new/` 가 삭제됐거나 collection name 안 맞음 | 절대 `chroma_db_new/` 지우지 말 것. 인덱싱 5분 걸림 |
| `webrtcvad` ModuleNotFoundError | voice.py 가 audio_service import 시 깨짐 | `pip install webrtcvad-wheels` |
| LLM 이 `<\|tool_call>` 텍스트로 뱉음 | Gemma + tool_choice="auto" | tool_choice 에 함수명 강제 |
| `/health` 응답이 이상 (`vlm_loaded` 같은 필드) | 다른 프로젝트 서버가 8000 잡고 있음 | `netstat -ano | sls ":8000\s"` 로 PID 확인. 본 레포는 8001 |
| 답변에 `(p.142)` 가 매 문장에 박힘 | 옛 RESEARCHER_SYSTEM 의 강제 인용 | 최신 system prompt 는 `[p.X]` 자연스럽게 — 덮어쓰지 말 것 |
| Playwright `NET::ERR_CERT_AUTHORITY_INVALID` | HTTPS 자체서명 인증서를 Playwright 가 거부 | `ssl_certificates` → `ssl_certificates.disabled` 로 일시 변경 |

---

## 5. 외부 트랙 / 별도 처리

- **AutoRAG 평가**: `autorag/` + `autorag_bench/` — 운영 swap-in 은 `autorag/adapter.py::install_into_agent_v2()` **명시 호출 시에만** 동작. 현재 startup 에 자동 호출 없음. 이 트랙은 사용자 명시 요청 없으면 건드리지 말 것.
- **OCR 벤치마크**: `scripts/ocr_poc/` + `results/ocr_poc/dashboard.html`. PDF 인덱싱에는 PyPDF 만 사용 중. 표/그림 페이지 회수율 개선 PoC 결과는 [OCR_BENCHMARK.md](OCR_BENCHMARK.md).
- **agent_v2 PoC**: `scripts/agent_v2_poc/` — 과거 PoC. 운영에서 import 안 됨.
- **legacy 문서**: `app/solution.md`, `app/summary.md`, `수리온_AI튜터_분석정리.md`, `docs/개발계획서_발췌/` — 참조용. 운영 코드와 동기화 안 됨.

---

## 6. 관측성 — Langfuse

- env 3개 (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`) 필요. `.env` 로드 자동.
- 없으면 `langfuse_handler` 가 no-op 으로 동작 — 코드 동작엔 영향 없음.
- trace 이름 `agent-pipeline`, 자식 span: `rag.child_to_parent`, `rag.parent_rerank`. span output 에 child/parent 청크 본문 미리보기 (200자) 들어가 있음 — RAG 검색 품질 디버깅 용도.
- 자세한 셋업은 [app/core/agent_v2/LANGFUSE_SETUP.md](app/core/agent_v2/LANGFUSE_SETUP.md).
