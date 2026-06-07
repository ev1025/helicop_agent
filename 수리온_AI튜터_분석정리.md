# 수리온 AI 튜터 (surion_llm) 프로젝트 분석 정리

> **작성일**: 2026-05-07
> **저장소**: `http://172.30.1.30:17580/llm/surion_llm.git`
> **대상 브랜치**: `gitlab-migration-fresh`

---

## 1. 브랜치 구조

| 브랜치 | 상태 |
|---|---|
| `main` | 거의 빈 상태 (Initial + refactoring 1만) |
| `gitlab-migration-fresh` | **실제 작업물** (Phase 1~6 누적) |

- 둘은 공통 조상 없는 별개 히스토리
- main의 `chroma_db_new` / Whisper 가중치는 자동 생성물

> **권장**: `gitlab-migration-fresh`를 받아 작업할 것.

---

## 2. 기술 스택

### LLM
- **현재 적용**: `Qwen/Qwen2.5-14B-Instruct`
  - 보고서엔 Llama-3-8B로 표기, 코드는 Qwen으로 변경됨
- **옵션**: Llama-3-Korean-Bllossom-8B, Qwen2.5-3B/7B/14B, EXAONE-3.5-7.8B, Yi-1.5-9B 등

### 임베딩
- **현재 적용**: `sentence-transformers/all-MiniLM-L6-v2`
- **옵션**: `dragonkue/snowflake-arctic-embed-l-v2.0-ko`, `jhgan/ko-sroberta-multitask` 등

### Reranker (현재 OFF, `use_reranker=false`)
- **후보 모델**:
  - `dragonkue/bge-reranker-v2-m3-ko` (한국어 특화)
  - `BAAI/bge-reranker-base`
  - `Dongjin-kr/ko-reranker`
- 코드는 모두 구비, config에서 비활성화만 한 상태
- `reranker_top_k=7`, `score_threshold=0.5` 설정값은 남아있음

### Hybrid Search (현재 OFF, `use_hybrid_search=false`)
- **구현**: `app/services/hybrid_retrieval.py`
- **방식**: Semantic + Keyword(BM25) 결합
- **동작**: Semantic 검색이 `min_results_threshold` 미만이면 Keyword 검색으로 보완해 합산 후 rerank
- **관련 모듈**: `keyword_search.py`, `multi_query.py`

### Vector DB / STT / Web
- **Vector DB**: Chroma (Parent-Child 모드, `chroma_db_new/`에 persist)
- **STT**: Whisper-small (로컬, `app/services/my_whisper_small/`)
- **Web**: FastAPI + SSE 스트리밍 (음성 기능은 HTTPS 필요)

---

## 3. 최종 모델 작동 방식

```
사용자 질문 (텍스트 또는 음성)
        ↓
   [음성이면] Whisper STT → 텍스트
        ↓
┌──────────────────────────────────┐
│ FastAPI /chat/stream (SSE)       │
└──────────────────────────────────┘
        ↓
[1] RAG 검색 (벡터DB)
    └─ Parent-Child Retrieval 모드
        ├─ Child(300자)로 정밀 검색  → top 30개 (×2=60)
        └─ parent_id로 Parent(1500자) 회수 (중복 제거)
        ↓
[2] (Reranker 비활성화 — 벡터 결과 그대로 top 7 사용)
        ↓
[3] 컨텍스트 구성 (최대 8000자)
        ↓
[4] 프롬프트 합성 (rag_focus 템플릿)
    "오직 위 문서의 내용만 사용하세요"
        ↓
[5] LLM 생성 (Qwen2.5-14B-Instruct)
    Temp 0.6, RepPen 1.1, Top-p 0.9, Max 512 tokens
    └─ TextIteratorStreamer로 토큰 단위 스트림
        ↓
[6] SSE 이벤트로 즉시 송출
    ├─ text 이벤트 (토큰)
    └─ tts 이벤트 (문장 종결 + TTS 적합 판정 시)
        ↓
브라우저: 화면 출력 + Web Speech API로 합성음 재생
```

---

## 4. Parent-Child Retrieval 핵심

> **원리**: 검색은 작은 조각으로 정밀하게, 답변에는 큰 조각으로 풍부하게.

### 저장 구조
- **벡터 DB(Chroma)**: Child 300자 청크만 임베딩 저장
- **`parent_store`** (메모리 dict): Parent 1500자 본문 보관
- 각 Child의 metadata에 `parent_id` 들어있음

### 검색 흐름
1. Child 60개 벡터 검색 (정밀 키워드 매칭)
2. 각 Child의 `parent_id` 추출
3. **같은 `parent_id`는 한 번만** (자연스러운 중복 제거)
   - → 같은 Parent에 여러 Child가 매칭되면 그 Parent가 더 관련 깊다는 강한 신호
4. `parent_store`에서 해당 Parent 본문(1500자) 회수
5. 회수된 Parent들을 LLM 컨텍스트에 합침

### 왜 Parent 직접 검색이 아닌가

| 이유 | 설명 |
|---|---|
| ① 토큰 제한 | 임베딩 모델은 보통 512토큰(~1200자) 한도. 1500자 Parent는 잘려서 임베딩됨. Child는 안전. |
| ② 점수 분포 | 긴 문서 임베딩은 점수가 평균값에 몰려 threshold 컷오프가 무의미. Child는 점수가 spread 되어 0.5 임계값이 의미를 가짐. |
| ③ 다면적 질문 | 한 질문이 여러 측면을 묻을 때 Child가 각 측면을 독립적으로 매칭. |
| ④ Long-context degradation | 임베딩 모델은 짧은 문장으로 학습되어 긴 텍스트는 의미 표현 정밀도가 떨어짐. |

> **연산 비용**: dict 조회 1회 추가 (사실상 공짜)

---

## 5. 같은 주제가 1500자를 넘을 때의 처리

원본 4500자 주제 → 3개 Parent로 분할 (#42, #43, #44), `overlap` 150자로 경계 정보 보존.

### 대응 메커니즘
1. **Overlap 150자**: 경계 문장 양쪽에 보존
2. **여러 Parent 동시 회수**: top 7로 인접 Parent 함께
3. **Multi-Query** (활성화 상태): 한 질문을 변형 쿼리로 분해해 인접 Parent까지 잡음

### 남은 한계
- 회수된 Parent의 순서가 유사도 점수 순 → **본문 순서 보장 안 됨** (LLM이 흐름을 잘못 잡을 위험)
- 검색 누락된 인접 Parent는 정보 손실
- 6개 넘는 Parent 회수 시 `max_context`로 잘림

---

## 6. 현재 시스템의 약점과 개선 제안

### 약점 1 — 회수된 Parent 순서가 유사도 순

본문 순서로 정렬되지 않아 LLM이 흐름을 잘못 읽음. 약한 LLM에서 답변 흐름 어색해짐.

**처방** — `rag_service.py`의 final return 직전 한 줄 추가:

```python
final_results.sort(
    key=lambda x: (x['page'],
                   x['metadata'].get('parent_chunk_id', 0))
)
```

`parent_id`가 `'parent_{i}'` 형태의 순차 번호이므로 정렬 가능. 페이지 번호도 metadata에 있음.

---

### 약점 2 — 같은 주제의 인접 Parent 누락

[5] 에서 본 것처럼 한 주제가 1500자를 넘으면 여러 Parent(#42, #43, #44)로 쪼개진다.
이 중 검색에 일부 Parent(#42, #43)만 잡히고 인접 Parent(#44)가 점수 미달로 빠지면, **같은 주제의 뒷부분 정보가 통째로 누락**된다.
현재는 Multi-Query가 우회 보완 역할만 함.

#### 처방 옵션 A — Sentence-Window Retrieval (윈도우 방식)

매칭된 Parent의 **앞뒤 N개**를 무조건 함께 회수.

```
검색 결과: Parent #43 매칭됨
   ↓ window=±1로 설정
회수: Parent #42, #43, #44 모두 가져옴
   ↓ window=±2면
회수: Parent #41, #42, #43, #44, #45 모두 가져옴
```

`parent_id`가 `parent_{i}` 형태의 순차 번호이므로 산술로 인접 ID 계산 가능.

**구현은 5줄 정도면 가능**:

```python
# 회수된 parent_chunk_id 모음
matched_ids = {doc['metadata']['parent_chunk_id'] for doc in final_results}

# 각 매칭에 대해 ±1 인접 청크 자동 추가
extended_ids = set()
for pid in matched_ids:
    extended_ids.update([pid - 1, pid, pid + 1])

# parent_store에서 추가 회수
for pid in extended_ids - matched_ids:
    parent_key = f"parent_{pid}"
    if parent_key in models.parent_store:
        # final_results에 추가...
```

| 장점 | 단점 |
|---|---|
| 단순, 확실 | 무관한 인접 청크도 가져옴 → 컨텍스트 낭비 |
| 즉시 적용 가능 | window 크기 키울수록 8000자 한도에 빨리 도달 |
| 추가 인프라 불필요 | 단원 경계를 모르므로 다른 챕터까지 끌고 올 수 있음 |

---

#### 처방 옵션 B — Auto-Merging Retrieval (LlamaIndex 방식)

같은 상위 단원에서 청크가 **임계 비율 이상** 매칭되면 자동으로 더 큰 단위(상위 Parent)로 병합.

```
계층 구조 (3-level chunking 필요):
  Section 3 (제3장 양력, 4500자)
    ├─ Parent #42
    ├─ Parent #43
    └─ Parent #44

검색 결과:
  Parent #42 ✓ (매칭)
  Parent #43 ✓ (매칭)
  Parent #44 ✗ (미매칭)

→ 3개 중 2개 매칭 (66%) → threshold (예: 50%) 초과
→ 자동으로 Section 3 통째로 회수 (Parent #44 포함)
```

매칭 비율이 낮으면 (예: 1/3 = 33%) 병합 안 함 → 무관한 확장 방지.

| 장점 | 단점 |
|---|---|
| 진짜 관련 있을 때만 확장 | 계층 구조를 인덱싱 단계에 미리 만들어야 함 |
| 단원 경계를 정확히 인식 | 3-level chunking 코드 추가 필요 |
| 컨텍스트 낭비 적음 | PDF 파싱 시 단원 경계 추출이 어려울 수 있음 |

---

#### A vs B 비교

| 기준 | Sentence-Window (A) | Auto-Merging (B) |
|---|---|---|
| 구현 난이도 | 낮음 (5줄) | 높음 (인덱싱 재설계) |
| 정확성 | 중간 (가까운 무관 청크도 포함) | 높음 (단원 단위로 정밀) |
| 컨텍스트 효율 | 낮음 | 높음 |
| 즉시 적용 | ✅ | ❌ (인덱스 재구축 필요) |

> **권장**: 일단 옵션 A로 빠르게 검증 → 효과 보이면 옵션 B로 정밀화.

---

### 약점 3 — 리랭커 비활성화 (가장 중요)

**현재 config**:
```json
{
  "use_reranker": false,
  "reranker_model": null,
  "vector_search_top_k": 30,
  "reranker_top_k": 7
}
```

**문제**: 정밀 필터링 단계 자체가 빠진 상태. 거친 벡터 점수만으로 7개 자르고 LLM에 던짐.

**교과서적 RAG 파이프라인**:
```
[1] Vector Search: 그물 넓게 (top 100~200)
[2] Reranker: cross-encoder로 정밀 필터 → 상위 5~7만 남김
[3] LLM Context: 작고 정확하게 (8000자 이하)
```

> **핵심 원칙**: 싸고 넓은 검색 → 비싸고 정밀한 필터링 → 작고 집중된 컨텍스트

**권장 설정 변경**:

| 항목 | 현재 | 권장 |
|---|---|---|
| `vector_search_top_k` | 30 | **100** |
| `use_reranker` | false | **true** |
| `reranker_model` | null | **`BAAI/bge-reranker-v2-m3` 또는 `dragonkue/bge-reranker-v2-m3-ko`** |
| `reranker_top_k` | 7 | **5** |
| `reranker_score_threshold` | 0.5 | 0.5 (유지) |
| `max_context_length` | 8000 | 8000 유지 (또는 7000 축소) |

---

## 7. Phase 7 후보 작업 (우선순위 순)

1. **Reranker 다시 켜기** (가장 큰 효과 예상)
2. `vector_search_top_k` 30 → 100 확장
3. `reranker_top_k` 7 → 5 축소
4. Parent를 `page` / `parent_chunk_id` 순으로 정렬
5. 인접 청크 자동 회수 (Sentence-Window) 도입

**예상 효과**
- Phase 6 보고서의 "개념 설명 후퇴 -15점" 회복 가능성
- 정밀도 + 흐름 동시 개선
- 추가 비용: 리랭커 호출 1~3초 (Phase 6 로그상 3.5초)

---

## 8. 운영 메모

- `main`에만 `chroma_db_new`와 Whisper 모델 가중치 존재
  - → `gitlab-migration-fresh`로 이동 시 첫 실행에서 재생성
- `ssl_certificates/` 필요 (음성 기능은 HTTPS만 작동)
- `ffmpeg` 시스템 설치 필요 (WebM → WAV 변환)
- `requirements_v2.txt`에서 `webrtcvad` 주석 처리됨
  - → 로컬 빌드 의존성 문제, 별도 설치 필요할 수 있음

### 문서 읽는 순서
1. `.agents/PHASE6_STATUS.md`
2. `PHASE6_FINAL_STATUS.md`
3. `milestone1vs2.md`
4. `EVALUATION_REPORT.md`

---

## 9. PoC v1 결과 (브랜치 `jw/agent-refactoring`, 커밋 `d9ee73b`)

> **목적**: 기존 수동 구현 orchestrator (prompt + 정규식)의 한계를 LangGraph + Qwen native function calling 으로 대체할 수 있는지 검증.

### 9.1 작성된 코드

```
app/core/agent_v2/
├─ state.py            # LangGraph AgentState
├─ tools.py            # LangChain @tool 래퍼 (rag_search, final_answer)
├─ qwen_chat_model.py  # 커스텀 BaseChatModel (핵심)
├─ llm_qwen.py         # 4-bit transformers 로드 + 래핑
├─ graph.py            # StateGraph + ToolNode + 조건부 엣지
└─ runner.py           # 외부 호출용 entry point

scripts/agent_v2_poc/
├─ 01_qwen_native_tool_check.py        # Qwen <tool_call> 출력 확인
├─ 02_langgraph_integration_test.py    # ChatHuggingFace 한계 진단
├─ 03_custom_chatmodel_integration.py  # QwenChatModel + LangGraph 한 사이클
└─ 04_tool_choice_test.py              # 4가지 tool_choice 모드 검증
```

### 9.2 발견 사항

| 발견 | 의미 |
|---|---|
| **LangChain `ChatHuggingFace.bind_tools()` 가 Qwen native function calling 에 tools 미전달** | 직접 ChatModel 작성 필요 |
| **Qwen 의 `<tool_call>` BPE 토큰이 종종 깨져서 디코드됨** | 정규식 fallback (opening tag 옵셔널) + bare JSON fallback 으로 해결 |
| **`tool_choice="X"` 강제 시 args 누락 빈번** | Assistant prefix `<tool_call>\n{"name":"X","arguments":` 박기로 완벽 해결 |
| **`with_structured_output()` 자동 동작** | bind_tools 위에 얹힌 기능이라, 우리 구현으로 자연 호환 |

### 9.3 검증된 것 vs 안 된 것

| 항목 | 상태 |
|---|---|
| Qwen2.5-7B 4-bit 로드 (RTX 4060 8GB → 5.45 GB 사용) | ✅ |
| Qwen native `<tool_call>` 형식 응답 | ✅ |
| 커스텀 QwenChatModel + LangGraph ToolNode 한 사이클 | ✅ (mock RAG) |
| `tool_choice` 4가지 모드 | ✅ |
| **실제 chroma_db / 임베딩 RAG 연결** | ❌ 모든 테스트 mock |
| **chat.py / FastAPI 라우터 통합** | ❌ 웹 UI 는 여전히 옛 코드 사용 |
| **SSE 토큰 스트리밍** | ❌ 현재는 invoke 한 번에 답 |
| **기존 orchestrator.py 제거** | ❌ 그대로 |

→ **"기술적 가능성 확인"까지만**. 운영 적용 아님.

---

## 10. 다음 세션 작업 (멀티에이전트 + 통합 + 평가)

### 10.1 멀티에이전트 패턴: **Supervisor + Workers** (결정됨)

```
[사용자]
   ↓
┌─────────────────────────┐
│   Supervisor Agent       │ ← 라우터: 어느 워커로 보낼지 결정
│   (Qwen + structured    │
│    routing output)       │
└──┬──────┬──────┬─────────┘
   │      │      │
   ▼      ▼      ▼
┌────────┐ ┌────────┐ ┌────────┐
│Researcher│ │Procedure│ │Critic  │
│- rag_   │ │- rag_   │ │- 답변  │
│  search │ │  search │ │  품질  │
│- 답변   │ │- 단계   │ │  평가  │
│         │ │  추출   │ │        │
└────┬────┘ └────┬───┘ └────┬───┘
     │           │           │
     └───────────┴───────────┘
              ↓
         [사용자에게 답변]
```

각 워커 에이전트는 **현재 만든 `QwenChatModel` 을 공유** (모델 1개, 시스템 프롬프트와 도구만 차이).

### 10.2 작업 순서 (다음 세션)

1. **`app/core/agent_v2/agents/` 디렉토리 신규**
   - `supervisor.py` — 라우팅 결정 (Pydantic + `with_structured_output`)
   - `researcher.py` — 일반 사실 검색·답변
   - `procedure.py` — 절차/단계별 답변
   - `critic.py` — 답변 품질 평가
2. **`graph.py` 재작성** — Supervisor 노드 + 워커 노드들 + 조건부 라우팅
3. **실제 RAG 연결** — `app/services/rag_service.py` 의 함수를 `tools.py` 의 `@tool rag_search` 안에서 그대로 호출 (이미 코드는 그렇게 짜둠, mock 만 활성화한 상태)
4. **chat.py 통합** — `/chat/v2/stream` 신규 라우트 추가, `runner` 호출
5. **SSE 스트리밍** — LangGraph `app.stream()` 을 SSE 이벤트로 변환
6. **기존 orchestrator.py 제거 검토** — chat.py 가 더 이상 참조 안 하게 되면 제거

### 10.3 chat.py 통합이 의미하는 것

```
[사용자 브라우저]
        ↓ HTTP
[FastAPI 서버]
        ↓
[app/api/routes/chat.py /chat/stream]
        ↓
   현재:  rag_service.rag_search() + llm.stream()  ← 옛 단순 파이프라인
   목표:  agent_v2.runner.get_runner().run(message)  ← LangGraph 멀티에이전트
        ↓
[브라우저로 SSE 스트림]
```

이 한 줄 변경 안 하면 만든 코드는 영원히 잠자는 상태. **반드시 필요한 단계**.

---

## 11. 성능 비교 측정 원칙 (모든 작업의 전제)

> **새 시스템이 기존보다 어느 차원에서 어떻게 좋은지 / 나쁜지를 숫자로 보여야 함.**
> 좋다고 주장만 하면 안 됨.

### 11.1 측정 차원

| 차원 | 지표 | 측정 방법 |
|---|---|---|
| **속도** | 첫 토큰 지연 (TTFT), 전체 응답 시간 | `time.perf_counter()` 로 endpoint 호출~응답 끝 측정 |
| **답변 정확도** | F1, BLEU, ROUGE, 정답 키워드 포함률 | 기존 `tests/rag_cases.json` + `scripts/comprehensive_evaluation_phase2.py` 재활용 |
| **답변 유사도** | embedding 기반 cosine 유사도 (정답 vs 생성 답변) | sentence-transformers 임베딩 후 cosine |
| **RAG 품질** | 회수된 문서 적중률, 페이지 정확도 | Phase 2 평가 항목 그대로 사용 |
| **도구 호출 신뢰도** | `<tool_call>` 파싱 성공률, JSON 형식 위반 비율 | agent_v2 의 파서 로그 집계 |
| **응답 길이/장황함** | Phase 2 의 "장황함 점수" 재사용 | `comprehensive_evaluation_phase2.py` |
| **반복/할루시네이션** | n-gram 반복률, 문서 외 정보 비율 | 기존 평가 스크립트 |
| **자원 사용** | VRAM 점유, RAM, CPU | `nvidia-smi`, `psutil` |

### 11.2 비교 대상 (반드시 두 시스템 같은 입력으로)

```
입력: tests/rag_cases.json 의 21문항 (Phase 6 평가 기준 동일)

[기존 시스템]              [agent_v2 시스템]
chat.py 옛 파이프라인       chat.py 신규 라우트 (/chat/v2)
  │                         │
  │ RAG → LLM 단순           │ Supervisor → Worker → Critic
  │                         │
  ▼                         ▼
응답 + metric 기록           응답 + metric 기록
  │                         │
  └────────┬────────────────┘
           ▼
  같은 metric 으로 비교 (벤치마크 스크립트)
```

### 11.3 출력 형태

`results/agent_v2_vs_old_YYYYMMDD.csv` 에 다음 컬럼:

```
case_id, question,
old_answer, new_answer,
old_ttft_ms, new_ttft_ms,
old_total_ms, new_total_ms,
old_f1, new_f1,
old_cosine_sim, new_cosine_sim,
old_rag_quality, new_rag_quality,
old_tool_parse_success, new_tool_parse_success,
old_vram_peak_mb, new_vram_peak_mb,
winner_by_metric  # f1: new, latency: old, ...
```

추가로 `results/agent_v2_vs_old_summary.md` 에 평균/등급별 분포 보고서.

### 11.4 평가 스크립트 위치 (기존 재활용)

| 도구 | 용도 |
|---|---|
| `scripts/comprehensive_evaluation_phase2.py` | 메인 채점 (장황함, 반복, F1, RAG 품질 등) |
| `scripts/run_phase2_bench.py` | 벤치 러너 (시간 측정 포함) |
| `tests/rag_cases.json` | 평가 케이스 21개 |
| `results/best_model/` | 기존 베스트 결과 (비교 베이스라인) |

→ **다음 세션 첫 작업 중 하나로 "agent_v2 호출 → 평가 스크립트 입력" 어댑터 만들기**.

---

## 12. 다음 세션 시작 시 체크리스트

```
[ ] git checkout jw/agent-refactoring  (현재 브랜치 확인)
[ ] git log -5  (마지막 커밋 d9ee73b 확인)
[ ] cd team_gitlab; .venv/Scripts/activate  (venv 활성화)
[ ] python -c "import torch; torch.cuda.is_available()"  (환경 확인)
[ ] 본 문서 9~11장 다시 읽고 시작
[ ] todo list 새로 작성:
    1. agent_v2/agents/ 디렉토리 + Supervisor, Researcher, Critic 구현
    2. graph.py 멀티에이전트로 재작성
    3. mock 제거하고 실제 chroma_db / 임베딩 연결 (단, main 브랜치 chroma_db 가 없으면 새로 인덱싱 필요)
    4. tests/rag_cases.json 으로 새/옛 시스템 비교 벤치 스크립트 작성
    5. chat.py /chat/v2/stream 라우트 추가 + SSE 스트리밍
    6. 결과 비교 보고서 (CSV + Markdown)
[ ] 기존 orchestrator.py 제거는 모든 검증 끝난 후 마지막에
```
