# Langfuse 활성화 가이드

코드는 항상 callback 을 주입하지만 env vars 없으면 no-op (영향 0). 키만 설정하면 즉시 trace 시작.

## 옵션 1 — Cloud + `.env` 파일 (권장)

1. https://cloud.langfuse.com 가입 (Free tier 무제한)
2. 프로젝트 생성 → Settings → API Keys 에서 public/secret key 복사
3. repo 루트의 `.env.example` 을 `.env` 로 복사하고 값 채우기:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

   `.env` 는 `.gitignore` 에 있으므로 commit 안 됨. 서버(`app/api/main.py`)와 스크립트 (`langfuse_handler`) 가 자동으로 로드함 (`python-dotenv`).
4. 서버 재시작 후 `/chat/v2/stream` 호출 → 대시보드에서 trace 확인.

### `.env` 없이 쓰려면 (셸 env vars)

```powershell
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_HOST = "https://cloud.langfuse.com"
uvicorn app.api.main:app --reload
```

## 옵션 2 — Self-host (운영망/오프라인)

망분리 환경이거나 데이터 외부 송출 금지 시:

```bash
# Docker compose 한번 다운로드
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d   # Postgres + Clickhouse + Langfuse web 모두 기동

# 브라우저로 http://localhost:3000 접속 → 계정 생성 → API key 발급
```

env vars:

```powershell
$env:LANGFUSE_PUBLIC_KEY = "pk_lf_..."
$env:LANGFUSE_SECRET_KEY = "sk_lf_..."
$env:LANGFUSE_HOST = "http://localhost:3000"
```

## 동작 검증

```powershell
.\.venv\Scripts\python.exe scripts\agent_v2_poc\31_langfuse_smoke.py
```

env vars 미설정 시 출력:
```
Langfuse 활성화: False (callbacks=0개)
→ env vars 미설정이므로 no-op.
```

env vars 설정 시:
```
Langfuse 활성화: True (callbacks=1개)
→ trace flush 완료. <host> 확인.
```

## 대시보드에서 보이는 것

- **Sessions**: 사용자 1요청 = 1 session, 그 안에 모든 LLM 호출
- **Traces**: Supervisor → Researcher (RAG search → answer streaming) 노드별 latency 와 token
- **Generations**: 각 LLM 호출의 input/output, prompt, finish_reason
- **Datasets / Experiments**: 7질의 등록 후 코드 변경마다 자동 실행해 비교
