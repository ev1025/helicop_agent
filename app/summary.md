# main.py 데이터 흐름 분석

## 1. 입력 데이터

### 1.1 음성 채팅 입력
- **형식**: 
  - WebM, WAV 등 오디오 파일 (UploadFile 객체)
  - 최대 10MB 제한
- **저장 위치**: 
  - `tempfile.mkstemp(suffix='.{suffix}')` - 임시 파일 시스템
  - 처리 후 자동 삭제 (finally 블록)
- **전처리 방식**:
  1. `validate_file_info()` - 파일 정보 검증
  2. `AudioProcessor.webm_to_wav()` - FFmpeg로 16kHz 모노 WAV 변환
  3. `AudioProcessor.extract_voice()` - webrtcvad로 음성 구간 추출 (VAD)
  4. `transcribe_audio()` - Whisper로 STT 처리
- **관련 함수**: `speak_stream_enhanced()`, `voice_chat()`

### 1.2 텍스트 채팅 입력
- **형식**: 
  - JSON 요청 본문 (`TextMessageRequest`, `ChatRequest`)
  - 문자열 쿼리 파라미터
- **저장 위치**: 
  - 메모리 (Python 변수)
- **전처리 방식**:
  - 직접 처리 (별도 전처리 없음)
- **관련 함수**: `chat_stream()`, `speak_text_stream()`

### 1.3 PDF 문서 (RAG 초기화)
- **형식**: 
  - PDF 파일
- **저장 위치**: 
  - `조종사표준교재(비행이론_헬리콥터).pdf`
  - 프로젝트 루트 디렉토리
- **전처리 방식**:
  1. `PyPDFLoader.load()` - PDF 텍스트 추출
  2. `RecursiveCharacterTextSplitter` - 청킹 (chunk_size=950, overlap=50)
  3. `get_embeddings()` - 벡터 임베딩 생성
- **관련 함수**: `load_models()`

---

## 2. 중간 데이터 (임베딩, 벡터 등)

### 2.1 문서 임베딩 벡터
- **형식**: 
  - List[List[float]] - 8192차원 밀집 벡터
  - L2 정규화된 임베딩
- **저장 위치**: 
  - Chroma 벡터 데이터베이스 (`./chroma_db_new`)
  - 컬렉션명: `"new_manual"`
- **사용 방식**:
  1. `CustomEmbeddings.embed_documents()` - 문서 임베딩
  2. `Chroma.from_documents()` - 벡터 DB 저장
  3. `collection.similarity_search()` - 코사인 유사도 검색
- **관련 클래스/함수**: `CustomEmbeddings`, `get_embeddings()`, `load_models()`

### 2.2 쿼리 임베딩
- **형식**: 
  - List[float] - 8192차원 벡터
  - "query: " prefix 추가
- **저장 위치**: 
  - 메모리 (일시적)
- **사용 방식**:
  - `CustomEmbeddings.embed_query()` - 쿼리 임베딩 생성
  - Chroma 검색에 즉시 사용 후 폐기
- **관련 함수**: `get_embeddings(is_query=True)`

### 2.3 Re-rank 점수
- **형식**: 
  - List[Tuple[str, float]] - (문서, 관련도 점수)
  - 점수 범위: 0.0~1.0 (sigmoid 적용)
- **저장 위치**: 
  - 메모리 (함수 반환값)
- **사용 방식**:
  1. `rerank_documents()` - 쿼리-문서 쌍 재평가
  2. 0.7 이상 점수만 컨텍스트 구성
  3. 점수 내림차순 정렬
- **관련 함수**: `rerank_documents()`, `rag_search_with_rerank()`

### 2.4 RAG 컨텍스트
- **형식**: 
  - 문자열 (최대 2500자)
  - 형식: `"[문서 {i} - {source} 페이지 {page}] {content}"`
- **저장 위치**: 
  - 메모리 (함수 로컬 변수)
- **사용 방식**:
  - LLM 프롬프트의 "참고 문서" 섹션에 삽입
  - 1500자 초과시 문장 경계로 절단
- **관련 함수**: `rag_search_with_rerank()`, `llm_streamer_with_rag_and_tts_v3()`

### 2.5 LLM 입력 토큰
- **형식**: 
  - PyTorch Tensor (torch.int64)
  - 최대 6000 토큰
- **저장 위치**: 
  - GPU/CPU 메모리 (`model.device`)
- **사용 방식**:
  1. `tokenizer.apply_chat_template()` - 채팅 템플릿 적용
  2. `tokenizer.encode()` - 토큰화 (fallback)
  3. Attention mask 생성
- **관련 함수**: `llm_streamer_with_rag_and_tts_v3()`, `generate_chat_style_response()`

---

## 3. 출력 데이터

### 3.1 스트리밍 텍스트 응답
- **형식**: 
  - Server-Sent Events (SSE)
  - JSON 페이로드: `{'type': 'text', 'content': str, 'lang': 'ko'}`
- **저장/전달 방식**:
  1. `TextIteratorStreamer` - 토큰 단위 실시간 생성
  2. `StreamingResponse` - HTTP 청크 전송
  3. 클라이언트 측 EventSource API 수신
- **관련 함수**: `llm_streamer_with_rag_and_tts_v3()`, `chat_stream()`, `speak_stream_enhanced()`

### 3.2 TTS 데이터
- **형식**: 
  - JSON 메타데이터:
    ```python
    {
        'type': 'tts',
        'content': str,  # 정제된 문장
        'lang': 'ko-KR',
        'rate': 1.0,
        'pitch': 1.0,
        'volume': 0.8
    }
    ```
- **저장/전달 방식**:
  1. `is_complete_sentence_v2()` - 문장 완성 판단
  2. `clean_tts_text()` - 마크다운/특수문자 제거
  3. SSE로 클라이언트 전송
  4. 클라이언트 Web Speech API로 음성 합성
- **관련 함수**: `llm_streamer_with_rag_and_tts_v3()`, `preprocess_text_for_tts()`, `clean_tts_text()`

### 3.3 RAG 검색 결과 메타데이터
- **형식**: 
  - JSON 배열:
    ```python
    {
        'type': 'rag_info',
        'content': str,  # 요약 메시지
        'doc_count': int
    }
    {
        'type': 'doc_details',
        'content': str,  # 상세 정보
        'documents': List[str]  # 문서 목록
    }
    ```
- **저장/전달 방식**:
  - SSE 이벤트로 클라이언트 전송
  - 응답 생성 전 참고 문서 정보 제공
- **관련 함수**: `chat_stream()`, `speak_stream_enhanced()`, `speak_text_stream()`

### 3.4 JSON 응답 (비스트리밍)
- **형식**: 
  - `VoiceChatResponse` Pydantic 모델:
    ```python
    {
        'user_text': str,      # STT 결과
        'ai_response': str,    # 완성된 LLM 응답
        'status': 'success'
    }
    ```
- **저장/전달 방식**:
  - FastAPI JSONResponse
  - 한 번에 완성된 응답 반환
- **관련 함수**: `voice_chat()`, `generate_chat_style_response()`

### 3.5 상태 이벤트
- **형식**: 
  - JSON:
    ```python
    {'type': 'info', 'content': str}          # 진행 상황
    {'type': 'user_speech', 'content': str}   # STT 결과
    {'type': 'tts_complete', 'total_items': int}  # TTS 완료
    {'type': 'done'}                          # 전체 완료
    {'type': 'error', 'content': str}         # 오류
    ```
- **저장/전달 방식**:
  - SSE 스트림으로 전송
  - 클라이언트 UI 업데이트용
- **관련 함수**: 모든 스트리밍 엔드포인트

---

## 데이터 흐름 요약

```
[음성 입력] 
  → tempfile → WebM→WAV 변환 → VAD 처리 → STT
  → [텍스트]

[텍스트 입력/STT 결과]
  → RAG 검색 (임베딩 → Chroma → Re-rank)
  → [컨텍스트 구성]

[컨텍스트 + 쿼리]
  → 프롬프트 생성 → 토큰화
  → LLM 추론 (스트리밍)
  → [토큰 스트림]

[토큰 스트림]
  → 문장 버퍼링 → TTS 전처리
  → SSE (텍스트 + TTS 메타데이터)
  → [클라이언트]
```

**핵심 특징**:
- 임시 데이터는 메모리/tempfile 활용 후 즉시 삭제
- 벡터 DB만 영구 저장 (`./chroma_db_new`)
- 실시간 스트리밍으로 지연 시간 최소화
- 클라이언트 TTS 방식으로 서버 부하 감소

---
---

# 핵심 기능별 코드 분석

## 3.1. RAG 파이프라인

### 📌 핵심 함수: `rag_search_with_rerank()`

```python
def rag_search_with_rerank(query: str, top_k: int = 5, rerank_top_k: int = 3):
```

**파이프라인 구조**:

```
1단계: 벡터 검색 (Vector Search)
├─ collection.similarity_search(query, k=top_k * 2)
├─ 임베딩 모델: dragonkue/snowflake-arctic-embed-l-v2.0-ko
└─ 코사인 유사도 기반 검색

2단계: 재순위화 (Re-ranking)
├─ rerank_documents(query, documents, rerank_model, ...)
├─ 모델: dragonkue/bge-reranker-v2-m3-ko
└─ 관련도 점수 0.7 이상만 선택

3단계: 컨텍스트 구성
├─ 최대 길이: 2500자
├─ 문서별 메타데이터 포함 (source, page, score)
└─ 형식: "[문서 {i} - {source} 페이지 {page}] {content}"
```

**초기화 (load_models())**:
```python
# 1. PDF 로딩 및 청킹
loader = PyPDFLoader(pdf_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=950,    # 청크 크기
    chunk_overlap=50   # 중복 구간
)
texts = splitter.split_documents(docs)

# 2. 벡터 DB 생성
collection = Chroma.from_documents(
    texts,
    embedding=CustomEmbeddings(embedding_model, ...),
    persist_directory="./chroma_db_new",
    collection_name="new_manual"
)
```

---

## 3.2. LLM 추론

### 📌 핵심 함수: `llm_streamer_with_rag_and_tts_v3()`

**모델 정보**:
- **모델**: `MLP-KTLim/llama-3-Korean-Bllossom-8B`
- **양자화**: 4bit NF4 (BitsAndBytesConfig)
- **디바이스**: GPU (CUDA) / CPU fallback

**추론 파이프라인**:

```python
# 1. 프롬프트 구성
messages = [
    {"role": "system", "content": PROMPT},
    {"role": "user", "content": instruction}  # RAG 컨텍스트 포함
]

# 2. 토큰화
input_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 3. 스트리밍 생성
streamer = TextIteratorStreamer(
    tokenizer, 
    skip_prompt=True,
    skip_special_tokens=True
)

generation_kwargs = {
    "max_new_tokens": 1024,
    "temperature": 0.6,      # 창의성 조절
    "top_p": 0.85,           # Nucleus sampling
    "repetition_penalty": 1.1,
    "streamer": streamer
}

# 4. 별도 스레드에서 생성
generation_thread = threading.Thread(
    target=model.generate, 
    kwargs=generation_kwargs
)
generation_thread.start()

# 5. 실시간 토큰 수신 및 전송
for new_text in streamer:
    yield f"data: {json.dumps({'type': 'text', 'content': new_text})}\n\n"
```

**프롬프트 구조**:
```python
# RAG 모드
instruction = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{rag_context}

질문: {prompt}

답변 시 다음 사항을 지켜주세요:
1. 참고 문서의 내용을 바탕으로 정확하게 답변하세요.
2. 문서에 없는 내용은 추측하지 마세요.
3. 한국어로 자연스럽게 답변하세요.
4. 답변 마지막에 "[참고 문서 기반 답변]"을 추가하세요."""
```

---

## 3.3. 파인튜닝

### ❌ **해당 기능 없음**

코드에 파인튜닝 관련 로직 없음. 사전 학습된 모델을 그대로 사용:
- LLaMA-3 한국어 모델 (이미 파인튜닝된 상태)
- 임베딩/리랭크 모델도 사전 학습 모델 사용

---

## 3.4. 리랭킹

### 📌 핵심 함수: `rerank_documents()`

```python
def rerank_documents(query: str, documents: List[str], 
                     rerank_model, rerank_tokenizer, device,
                     top_k: int = 5) -> List[Tuple[str, float]]:
```

**리랭킹 프로세스**:

```python
# 1. 쿼리-문서 쌍 생성
query_doc_pairs = [[query, doc] for doc in documents]

# 2. 토큰화
features = rerank_tokenizer(
    query_doc_pairs, 
    padding=True, 
    truncation=True, 
    return_tensors="pt", 
    max_length=512  # 임베딩보다 짧은 길이
).to(device)

# 3. 관련성 점수 계산
rerank_model.eval()
with torch.no_grad():
    logits = rerank_model(**features).logits
    scores = torch.sigmoid(logits).squeeze().cpu().tolist()

# 4. 정렬 및 상위 K개 선택
doc_score_pairs = list(zip(documents, scores))
doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
return doc_score_pairs[:top_k]
```

**모델 세부사항**:
- **모델**: `dragonkue/bge-reranker-v2-m3-ko`
- **타입**: `AutoModelForSequenceClassification`
- **출력**: 0~1 사이의 관련도 점수 (sigmoid 적용)
- **임계값**: 0.7 이상만 컨텍스트에 포함

**Vector Search vs Re-rank 차이**:
```
Vector Search (1차 필터링)
├─ 속도: 빠름 (코사인 유사도)
├─ 정확도: 보통
└─ 결과: top_k * 2 개 (여유있게 검색)

Re-rank (2차 정밀 평가)
├─ 속도: 느림 (트랜스포머 추론)
├─ 정확도: 높음 (쿼리-문서 상호작용 모델링)
└─ 결과: rerank_top_k 개 (최종 선택)
```

---

## 3.5. 임베디드 배포

### ⚠️ **부분적 지원**

**현재 구현**:
```python
# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU 모드 fallback (GPU 없는 환경)
if not torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # CPU용 정밀도
        device_map={"": "cpu"},
        trust_remote_code=True
    )
```

**제한사항**:
- ✅ CPU 실행 가능
- ❌ 경량화/최적화 없음 (ONNX, TensorRT 등)
- ❌ 모바일/엣지 디바이스 미지원
- ⚠️ 8B 모델은 임베디드 환경에 너무 큼

**개선 방향**:
```python
# (예시) 실제 임베디드 배포를 위한 최적화 필요
# - 모델 양자화: INT8, INT4
# - 프루닝: 불필요한 가중치 제거
# - 지식 증류: 더 작은 모델로 변환
# - ONNX Runtime 또는 TFLite 변환
```

---

## 3.6. Agentic AI

### ❌ **해당 기능 없음**

**현재 시스템 특징**:
- 단일 턴 질의응답 (Single-turn Q&A)
- 도구 사용 없음 (No tool calling)
- 계획 수립 없음 (No planning)
- 멀티 에이전트 없음 (No multi-agent)

**Agentic AI가 되려면 필요한 것**:
```python
# (예시) 구현 필요 기능
1. Tool Calling
   - 함수 호출 능력 (계산기, 검색, API 호출 등)
   
2. Planning & Reasoning
   - ReAct, Chain-of-Thought
   - 다단계 문제 해결
   
3. Memory & Context
   - 대화 히스토리 관리
   - 장기 메모리
   
4. Self-Correction
   - 오류 감지 및 재시도
   - 피드백 기반 개선
```

**현재 코드의 한계**:
```python
# 현재: 단순 RAG + LLM
rag_context = rag_search_with_rerank(query)
response = llm_generate(query, rag_context)

# Agentic 시스템이라면:
# 1. 질문 분석 → 필요한 도구 선택
# 2. 도구 실행 (RAG, 계산, 외부 API 등)
# 3. 결과 종합 → 추가 정보 필요시 반복
# 4. 최종 답변 생성
```

---

## 📊 기능 구현 현황 요약

| 기능 | 구현 여부 | 완성도 | 비고 |
|------|----------|--------|------|
| **RAG 파이프라인** | ✅ 완전 구현 | ⭐⭐⭐⭐⭐ | Vector Search + Re-rank 2단계 |
| **LLM 추론** | ✅ 완전 구현 | ⭐⭐⭐⭐⭐ | 실시간 스트리밍 지원 |
| **파인튜닝** | ❌ 미구현 | - | 사전 학습 모델 사용 |
| **리랭킹** | ✅ 완전 구현 | ⭐⭐⭐⭐⭐ | BGE Reranker 활용 |
| **임베디드 배포** | ⚠️ 부분 지원 | ⭐⭐ | CPU fallback만 존재 |
| **Agentic AI** | ❌ 미구현 | - | 단순 Q&A 시스템 |

---
---

# 설정 파일 및 환경 분석

## 4.1. 주요 라이브러리

### **AI/ML 코어**
```python
torch                    # PyTorch 딥러닝 프레임워크
transformers            # Hugging Face 모델 (LLaMA-3, 임베딩, 리랭크)
bitsandbytes           # 4bit 양자화 (BitsAndBytesConfig)
```

### **RAG 스택**
```python
langchain-community     # PDF 로더 (PyPDFLoader)
langchain-text-splitters  # 텍스트 청킹 (RecursiveCharacterTextSplitter)
langchain-chroma       # 벡터 DB (Chroma)
```

### **음성 처리**
```python
webrtcvad              # 음성 활동 감지 (VAD)
wave                   # WAV 파일 처리
numpy                  # 오디오 신호 처리
# whisper (app.whisper_util에서 사용 - 별도 모듈)
# pyttsx3 (TTS - 현재 비활성화)
```

### **웹 프레임워크**
```python
fastapi                # 웹 API 프레임워크
uvicorn               # ASGI 서버
jinja2                # HTML 템플릿 엔진
pydantic              # 데이터 검증
```

### **시스템 유틸**
```python
subprocess            # FFmpeg 실행 (오디오 변환)
tempfile             # 임시 파일 관리
threading            # LLM 스트리밍 멀티스레딩
asyncio              # 비동기 처리
```

---

## 4.2. config 파일

### ❌ **별도 config 파일 없음**

**설정이 코드에 하드코딩된 상태**:

```python
# === 모델 설정 (load_models 함수 내) ===
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
emb_name = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
rr_name = "dragonkue/bge-reranker-v2-m3-ko"

# === 양자화 설정 ===
qcfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# === 텍스트 분할 설정 ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=950,
    chunk_overlap=50
)

# === 벡터 DB 설정 ===
DB_PATH = "./chroma_db_new"
collection_name = "new_manual"

# === 오디오 처리 설정 (AudioProcessor.__init__) ===
self.sample_rate = 16000      # 16kHz
self.frame_duration = 30      # 30ms
self.silence_sec = 1.0        # 1초 무음 기준

# === LLM 생성 파라미터 ===
generation_kwargs = {
    "max_new_tokens": 1024,
    "temperature": 0.6,
    "top_p": 0.85,
    "repetition_penalty": 1.1,
}

# === RAG 검색 설정 (rag_search_with_rerank) ===
top_k = 10                    # Vector 검색 결과
rerank_top_k = 3              # Re-rank 후 선택
max_context_length = 2500     # 컨텍스트 최대 길이
score_threshold = 0.7         # 관련도 임계값

# === 파일 업로드 제한 ===
max_file_size = 10 * 1024 * 1024  # 10MB

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

---

## 4.3. 환경변수

### **API 키**
```bash
# ❌ 없음 - 로컬 모델만 사용
# OpenAI API, Anthropic API 등 외부 API 미사용
```

### **모델 경로**
```bash
# ❌ 명시적 환경변수 없음
# Hugging Face Hub에서 자동 다운로드:
# - ~/.cache/huggingface/hub/ 에 저장
# - 모델명으로 자동 관리
```

### **GPU 디바이스 설정**
```python
# 환경변수로 GPU 사용 여부 제어
USE_EMBEDDING_GPU = os.getenv('USE_EMBEDDING_GPU', 'true')  # default: True
USE_RERANK_GPU = os.getenv('USE_RERANK_GPU', 'true')        # default: True

# 사용 예시:
# export USE_EMBEDDING_GPU=false  # 임베딩 CPU 사용
# export USE_RERANK_GPU=true      # 리랭크 GPU 사용
```

### **디버그 모드**
```python
DEBUG_VERBOSE = os.getenv('DEBUG_VERBOSE', 'false')  # default: False

# 사용 예시:
# export DEBUG_VERBOSE=true  # 상세 로그 활성화
```

### **기타 시스템 환경변수 (암묵적)**
```bash
# PyTorch CUDA 관련
CUDA_VISIBLE_DEVICES=0,1  # 사용할 GPU 지정

# Hugging Face Hub
HF_HOME=/path/to/cache    # 모델 캐시 경로
HF_TOKEN=hf_xxx           # Private 모델 접근용 (현재 미사용)

# FFmpeg
PATH=/usr/bin/ffmpeg      # 오디오 변환에 필요
```

---

## 📋 설정 개선 제안

### **현재 문제점**:
1. ❌ 하드코딩된 설정 → 유지보수 어려움
2. ❌ 환경변수 최소 사용 → 배포 환경별 분리 어려움
3. ❌ Config 파일 없음 → 설정 변경시 코드 수정 필요

### **개선 방안**:

```yaml
# config.yaml (제안)
models:
  llm:
    name: "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    quantization: "4bit"
    device: "auto"
  embedding:
    name: "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    device: "cuda"
  reranker:
    name: "dragonkue/bge-reranker-v2-m3-ko"
    device: "cuda"

rag:
  chunk_size: 950
  chunk_overlap: 50
  vector_db_path: "./chroma_db_new"
  top_k: 10
  rerank_top_k: 3
  score_threshold: 0.7
  max_context_length: 2500

generation:
  max_new_tokens: 1024
  temperature: 0.6
  top_p: 0.85
  repetition_penalty: 1.1

audio:
  sample_rate: 16000
  frame_duration_ms: 30
  silence_threshold_sec: 1.0
  max_file_size_mb: 10

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"
  log_file: "app.log"
```

```python
# .env (제안)
# GPU 설정
USE_EMBEDDING_GPU=true
USE_RERANK_GPU=true
CUDA_VISIBLE_DEVICES=0

# 디버그
DEBUG_VERBOSE=false

# 모델 캐시
HF_HOME=/data/models/huggingface

# (선택) 외부 API
# OPENAI_API_KEY=sk-xxx
# ANTHROPIC_API_KEY=sk-ant-xxx
```

```python
# 로드 방법 (제안)
import yaml
from dotenv import load_dotenv

load_dotenv()
with open('config.yaml') as f:
    config = yaml.safe_load(f)

model_id = config['models']['llm']['name']
chunk_size = config['rag']['chunk_size']
```

---

## 🎯 핵심 요약

| 항목 | 현재 상태 | 비고 |
|------|----------|------|
| **라이브러리** | 완전함 | PyTorch, Transformers, LangChain 스택 |
| **Config 파일** | ❌ 없음 | 코드에 하드코딩 |
| **환경변수** | 최소 사용 | GPU 설정 2개, 디버그 1개만 |
| **API 키** | ❌ 불필요 | 로컬 모델 전용 |
| **모델 경로** | 자동 관리 | Hugging Face Hub 활용 |
| **설정 유연성** | ⭐⭐ | 개선 필요 (YAML/ENV 분리 권장) |

---
---

# 리팩토링 파일 구성 3가지 제안

## 📁 제안 1: 최소 리팩토링 (기능별 분리)
**난이도**: ⭐ (쉬움)  
**목표**: 현재 구조 유지하면서 config만 분리

```
project/
├── main.py                    # FastAPI 라우트만 (300줄)
├── config.py                  # 모든 설정 집중 (100줄)
├── models.py                  # 모델 로딩 로직 (200줄)
├── rag.py                     # RAG 파이프라인 (150줄)
├── audio.py                   # 오디오 처리 (200줄)
├── llm.py                     # LLM 추론 + 스트리밍 (250줄)
├── utils.py                   # 유틸 함수들 (100줄)
├── .env                       # 환경변수
└── app/
    ├── templates/             # HTML
    ├── whisper_util.py
    └── tts_pyttsx3_worker.py
```

### 핵심 변경사항:
```python
# config.py
class Config:
    # 모델
    LLM_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    EMBEDDING_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    RERANK_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
    
    # RAG
    CHUNK_SIZE = 950
    CHUNK_OVERLAP = 50
    TOP_K = 10
    RERANK_TOP_K = 3
    SCORE_THRESHOLD = 0.7
    
    # LLM
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.6
    TOP_P = 0.85

# main.py
from config import Config
from models import load_all_models
from rag import rag_search_with_rerank
from llm import llm_streamer

model, tokenizer, collection = load_all_models()

@app.get("/chat/stream")
async def chat_stream(message: str):
    rag_results = rag_search_with_rerank(message, Config.TOP_K)
    return StreamingResponse(llm_streamer(message, rag_results))
```

**장점**: 
- 기존 로직 거의 변경 없음
- 빠른 적용 (1-2시간)
- 설정만 한 곳에서 관리

**단점**: 
- 여전히 절차적 프로그래밍
- 테스트 어려움

---

## 📁 제안 2: 중간 리팩토링 (레이어 분리)
**난이도**: ⭐⭐⭐ (보통)  
**목표**: Service Layer 패턴 적용

```
project/
├── main.py                    # FastAPI 엔트리포인트 (100줄)
├── config/
│   ├── __init__.py
│   ├── settings.py           # Pydantic Settings
│   └── prompts.py            # 프롬프트 템플릿
├── core/
│   ├── __init__.py
│   ├── models.py             # 모델 싱글톤
│   └── dependencies.py       # FastAPI 의존성
├── services/
│   ├── __init__.py
│   ├── rag_service.py        # RAG 비즈니스 로직
│   ├── llm_service.py        # LLM 비즈니스 로직
│   └── audio_service.py      # 오디오 처리 로직
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── chat.py           # /chat 엔드포인트
│   │   ├── voice.py          # /speak 엔드포인트
│   │   └── health.py         # /health
│   └── schemas.py            # Pydantic 모델
├── utils/
│   ├── __init__.py
│   ├── text_processing.py    # TTS 전처리
│   └── validators.py         # 입력 검증
├── .env
└── app/
    └── templates/
```

### 핵심 변경사항:
```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 모델
    llm_model: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    embedding_model: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    
    # RAG
    chunk_size: int = 950
    top_k: int = 10
    rerank_top_k: int = 3
    
    # 환경변수 자동 로드
    use_embedding_gpu: bool = True
    debug_verbose: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()

# services/rag_service.py
class RAGService:
    def __init__(self, collection, rerank_model):
        self.collection = collection
        self.rerank_model = rerank_model
    
    def search(self, query: str) -> List[Document]:
        results = self.collection.similarity_search(
            query, 
            k=settings.top_k * 2
        )
        return self._rerank(query, results)
    
    def _rerank(self, query, documents):
        # 리랭크 로직
        pass

# services/llm_service.py
class LLMService:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    async def stream_response(self, prompt: str, context: str):
        # 스트리밍 로직
        yield from self._generate_stream(prompt, context)

# api/routes/chat.py
from fastapi import APIRouter, Depends
from services.rag_service import RAGService
from services.llm_service import LLMService
from core.dependencies import get_rag_service, get_llm_service

router = APIRouter()

@router.get("/chat/stream")
async def chat_stream(
    message: str,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    rag_results = rag_service.search(message)
    context = rag_service.build_context(rag_results)
    return StreamingResponse(
        llm_service.stream_response(message, context)
    )

# main.py
from fastapi import FastAPI
from api.routes import chat, voice, health
from core.models import initialize_models

app = FastAPI()
app.include_router(chat.router)
app.include_router(voice.router)
app.include_router(health.router)

@app.on_event("startup")
async def startup():
    await initialize_models()
```

**장점**:
- 명확한 레이어 분리 (API - Service - Core)
- 의존성 주입으로 테스트 용이
- 설정을 Pydantic으로 타입 안전성

**단점**:
- 코드 이동량 많음 (2-3일 소요)
- 약간의 학습 곡선

---

## 📁 제안 3: 완전 리팩토링 (DDD + Clean Architecture)
**난이도**: ⭐⭐⭐⭐⭐ (어려움)  
**목표**: 도메인 중심 설계 + 확장성

```
project/
├── main.py                    # 진입점만
├── config/
│   ├── settings.py
│   └── logging.py
├── domain/                    # 핵심 비즈니스 로직 (프레임워크 독립)
│   ├── models/
│   │   ├── document.py       # Document 엔티티
│   │   ├── conversation.py   # Conversation 엔티티
│   │   └── audio.py          # Audio 값 객체
│   ├── services/
│   │   ├── rag_domain_service.py
│   │   └── llm_domain_service.py
│   └── repositories/         # 인터페이스 (추상)
│       ├── vector_repository.py
│       └── model_repository.py
├── application/               # 유스케이스 (애플리케이션 로직)
│   ├── use_cases/
│   │   ├── process_text_chat.py
│   │   ├── process_voice_chat.py
│   │   └── search_documents.py
│   └── dto/
│       ├── chat_request.py
│       └── chat_response.py
├── infrastructure/            # 외부 의존성 구현
│   ├── llm/
│   │   ├── huggingface_llm.py
│   │   └── openai_llm.py     # 추후 확장용
│   ├── vector_db/
│   │   ├── chroma_repository.py
│   │   └── pinecone_repository.py  # 추후 확장용
│   ├── audio/
│   │   ├── whisper_stt.py
│   │   └── elevenlabs_tts.py # 추후 확장용
│   └── cache/
│       └── redis_cache.py
├── presentation/              # API 레이어
│   ├── api/
│   │   ├── v1/
│   │   │   ├── chat.py
│   │   │   ├── voice.py
│   │   │   └── admin.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── rate_limit.py
│   ├── schemas/
│   │   └── api_models.py
│   └── dependencies.py
├── shared/                    # 공통 유틸
│   ├── utils/
│   ├── exceptions/
│   └── validators/
└── tests/                     # 테스트
    ├── unit/
    ├── integration/
    └── e2e/
```

### 핵심 변경사항:
```python
# domain/models/document.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Document:
    content: str
    page: int
    source: str
    relevance_score: Optional[float] = None
    
    def is_relevant(self, threshold: float = 0.7) -> bool:
        return self.relevance_score >= threshold if self.relevance_score else False

# domain/repositories/vector_repository.py (인터페이스)
from abc import ABC, abstractmethod

class VectorRepository(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int) -> List[Document]:
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        pass

# infrastructure/vector_db/chroma_repository.py (구현체)
class ChromaVectorRepository(VectorRepository):
    def __init__(self, collection, embedding_fn):
        self.collection = collection
        self.embedding_fn = embedding_fn
    
    async def search(self, query: str, top_k: int) -> List[Document]:
        results = self.collection.similarity_search(query, k=top_k)
        return [Document(
            content=r.page_content,
            page=r.metadata.get('page'),
            source=r.metadata.get('source')
        ) for r in results]

# application/use_cases/process_text_chat.py
class ProcessTextChatUseCase:
    def __init__(
        self, 
        vector_repo: VectorRepository,
        llm_service: LLMDomainService,
        reranker: RerankerService
    ):
        self.vector_repo = vector_repo
        self.llm_service = llm_service
        self.reranker = reranker
    
    async def execute(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        # 1. RAG 검색
        documents = await self.vector_repo.search(request.message, top_k=10)
        
        # 2. 리랭크
        ranked_docs = await self.reranker.rerank(request.message, documents)
        relevant_docs = [d for d in ranked_docs if d.is_relevant()]
        
        # 3. 컨텍스트 구성
        context = self._build_context(relevant_docs)
        
        # 4. LLM 스트리밍
        async for chunk in self.llm_service.stream(request.message, context):
            yield ChatResponse(
                type="text",
                content=chunk,
                metadata={"doc_count": len(relevant_docs)}
            )

# presentation/api/v1/chat.py
from fastapi import APIRouter, Depends
from application.use_cases.process_text_chat import ProcessTextChatUseCase

router = APIRouter(prefix="/api/v1/chat")

@router.get("/stream")
async def stream_chat(
    message: str,
    use_case: ProcessTextChatUseCase = Depends(get_text_chat_use_case)
):
    request = ChatRequest(message=message)
    
    async def event_generator():
        async for response in use_case.execute(request):
            yield f"data: {response.json()}\n\n"
    
    return StreamingResponse(event_generator())

# main.py
from fastapi import FastAPI
from presentation.api.v1 import chat, voice
from infrastructure.llm.huggingface_llm import HuggingFaceLLM
from infrastructure.vector_db.chroma_repository import ChromaVectorRepository
from application.use_cases.process_text_chat import ProcessTextChatUseCase

app = FastAPI()

# 의존성 컨테이너 (DI)
class Container:
    def __init__(self):
        # Infrastructure
        self.llm = HuggingFaceLLM(model_name=settings.llm_model)
        self.vector_repo = ChromaVectorRepository(...)
        
        # Use Cases
        self.text_chat_use_case = ProcessTextChatUseCase(
            vector_repo=self.vector_repo,
            llm_service=self.llm,
            reranker=...
        )

container = Container()

def get_text_chat_use_case():
    return container.text_chat_use_case

app.include_router(chat.router)
app.include_router(voice.router)
```

**장점**:
- 완벽한 관심사 분리
- 프레임워크 독립적 (FastAPI 교체 가능)
- 테스트 매우 용이 (Mock 주입)
- 확장성 극대화 (새 LLM, Vector DB 쉽게 추가)

**단점**:
- 초기 구조 설계 시간 (1주일+)
- 과도한 추상화 (작은 프로젝트엔 오버엔지니어링)
- 팀원 학습 필요

---

## 🎯 추천 선택 기준

| 상황 | 추천 | 이유 |
|------|------|------|
| **빠른 정리 목적** | 제안 1 | 1-2시간 작업, 즉시 적용 가능 |
| **유지보수 중시** | 제안 2 | 균형잡힌 구조, 테스트 가능 |
| **팀 프로젝트/장기** | 제안 3 | 확장성, 여러 개발자 협업 |
| **PoC/데모** | 제안 1 | 오버헤드 최소화 |
| **프로덕션** | 제안 2 | 실용적 선택 |
| **대규모 시스템** | 제안 3 | 아키텍처 견고함 |

---

## 💡 개인적 추천: **제안 2 (중간 리팩토링)**

**이유**:
1. ✅ 현재 코드를 크게 바꾸지 않으면서도 구조화
2. ✅ Service Layer 패턴은 학습 곡선 낮음
3. ✅ 테스트 작성 가능해짐
4. ✅ 2-3일이면 완료 가능
5. ✅ 추후 제안 3으로 진화 가능

**우선순위**:
```
1단계 (1일차): config 분리 + settings.py
2단계 (2일차): services 레이어 분리
3단계 (3일차): API 라우터 분리 + 의존성 주입
```

---

## 📋 마이그레이션 가이드

### 1단계: 파일 생성 (10분)
```bash
# 프로젝트 루트에서
touch config.py models.py rag.py audio.py llm.py utils.py .env
```

### 2단계: 코드 이동 (1시간)
1. **config.py**: 기존 main.py의 모든 상수값 복사
2. **models.py**: `load_models()`, `cleanup_models()`, 전역변수 이동
3. **rag.py**: `rag_search_with_rerank()`, `rerank_documents()` 이동
4. **audio.py**: `AudioProcessor` 클래스 이동
5. **llm.py**: `llm_streamer_...()`, `generate_chat_style_response()` 이동
6. **utils.py**: TTS 전처리 함수들 이동

### 3단계: main.py 정리 (30분)
1. 모든 import를 새 모듈로 변경
2. 라우트만 남기고 나머지 삭제
3. 동작 테스트

### 4단계: 검증 (30분)
```bash
# 서버 실행
python main.py

# 테스트
curl http://localhost:8000/health
# 웹 브라우저: http://localhost:8000
```

---

## ✅ 체크리스트

- [ ] config.py 생성 및 모든 상수 이동
- [ ] models.py 생성 및 모델 로딩 로직 이동
- [ ] rag.py 생성 및 RAG 함수 이동
- [ ] audio.py 생성 및 AudioProcessor 이동
- [ ] llm.py 생성 및 스트리밍 함수 이동
- [ ] utils.py 생성 및 유틸 함수 이동
- [ ] main.py에서 import 수정
- [ ] main.py에서 불필요한 코드 삭제
- [ ] .env 파일 생성
- [ ] 서버 실행 테스트
- [ ] 텍스트 채팅 테스트
- [ ] 음성 채팅 테스트
- [ ] /health 엔드포인트 확인

---

## 🎯 리팩토링 효과

| 항목 | 개선 전 | 개선 후 |
|------|---------|---------|
| **main.py 줄 수** | 1200+ | ~300 |
| **설정 관리** | 분산 | 한 곳 집중 |
| **코드 재사용** | 어려움 | 모듈 import |
| **테스트** | 불가능 | 각 모듈 개별 테스트 가능 |
| **유지보수** | 어려움 | 파일별 관심사 분리 |

이제 각 파일을 복사해서 사용하시면 됩니다!