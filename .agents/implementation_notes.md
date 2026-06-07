# Implementation Notes

이 문서는 **프로젝트별 구현 특이사항**을 기록합니다. 소스코드 수정 시 **반드시 먼저 확인**하여 반복적인 실수를 방지하세요.

---

## 📌 목차

1. [Config 시스템 (싱글톤)](#1-config-시스템-싱글톤)
2. [전역 모델 캐싱 시스템](#2-전역-모델-캐싱-시스템)
3. [Parent-Child Document Retrieval](#3-parent-child-document-retrieval)
4. [Chroma DB 재생성 요구사항](#4-chroma-db-재생성-요구사항)
5. [타입 안정성 (Chroma)](#5-타입-안정성-chroma)
6. [노트북 환경에서 로그 출력](#6-노트북-환경에서-로그-출력)

---

## 1. Config 시스템 (싱글톤)

### 구현 방식
```python
# app/config.py
class Config:
    def _load_configs(self):
        # JSON 파일에서 설정 로드하여 인스턴스 속성 업데이트
        ...

# 싱글톤 인스턴스 (모듈 레벨)
config = Config()
```

### ✅ 올바른 사용법

```python
# 1. Import
from app.config import config

# 2. Config 파일 수정 후 재로드
import json
with open('config/models.json', 'w') as f:
    json.dump(new_data, f)

config._load_configs()  # ✅ 싱글톤 인스턴스 업데이트

# 3. 값 접근
print(config.LLM_MODEL)  # ✅ 업데이트된 값 사용됨
```

### ❌ 흔한 실수

```python
# 실수 1: 모듈을 import
from app import config  # ❌ 모듈을 import (AttributeError 발생)
print(config.LLM_MODEL)  # AttributeError: module has no attribute 'LLM_MODEL'

# 올바름:
from app.config import config  # ✅ 싱글톤 인스턴스 import
```

### 💡 핵심 특징

1. **싱글톤**: 프로세스 전체에서 단일 인스턴스 공유
2. **동적 재로드**: `config._load_configs()` 호출 시 인스턴스 내부 값 업데이트
3. **전역 공유**: 한 곳에서 reload하면 모든 모듈에서 업데이트된 값 사용
4. **importlib.reload() 불필요**: `_load_configs()`로 충분

### 🧪 검증 방법

```python
# Config 변경 전후 값 확인
print(f"Before: {config.TEMPERATURE}")
config._load_configs()
print(f"After: {config.TEMPERATURE}")
```

### 📝 관련 파일
- `app/config.py` - Config 클래스 정의
- `config/*.json` - 설정 파일들
  - `models.json` - LLM, embedding, reranker 모델 및 생성 파라미터
  - `rag.json` - RAG 파라미터, chunk size 등
  - `prompts.json` - 시스템 프롬프트
  - `server.json`, `audio.json`, `logging.json`

---

## 2. 전역 모델 캐싱 시스템

### 구현 방식
```python
# app/core/models.py
# 전역 변수로 모델 캐싱
model = None
tokenizer = None
embedding_model = None
embedding_tokenizer = None
rerank_model = None
rerank_tokenizer = None
collection = None
parent_store = None

def load_llm_model():
    global model, tokenizer
    model, tokenizer = _load_llm(config.LLM_MODEL)
```

### ⚠️ 주의사항

1. **한 번 로드하면 캐싱됨**: 같은 프로세스에서 `load_llm_model()` 여러 번 호출해도 첫 호출 때 로드된 모델 재사용
2. **Config 변경 후 모델 재로드 필요**: Config만 변경하고 모델 재로드 안 하면 **이전 모델 사용**
3. **Phase 테스트 시 캐싱 문제**: 여러 모델 조합 테스트 시 명시적 재로드 필요

### ✅ Phase 테스트에서 올바른 사용법

```python
# Phase 2-6 노트북에서
for config_item in test_configs:
    # 1. Config 파일 수정
    apply_config(config_item)

    # 2. Config 재로드
    from app.config import config
    config._load_configs()

    # 3. 모델 강제 재로드 (캐싱 방지)
    from app.core import models

    # GPU 메모리 클리어
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Reranker 재로드 (Phase 2 예시)
    models.load_rerank_model()  # ✅ 전역 변수 덮어쓰기

    # 4. 벤치마크 실행 (LLM/Embedding은 여기서 로드)
    run_bench(...)
```

### 🔍 실제 로드된 모델 검증

```python
# scripts/run_phase2_bench.py의 get_actual_model_info()
def get_actual_model_info():
    model_info = {
        'config_llm_model': config.LLM_MODEL,
        'actual_llm_model': models.model.config.name_or_path if models.model else 'Not loaded',
        # Config와 실제 메모리의 모델 비교
    }
    return model_info
```

### 📝 Phase별 재로드 전략

| Phase | 재로드 대상 | 이유 |
|-------|------------|------|
| Phase 2 | Reranker만 | RAG 파라미터만 변경, LLM/Embedding 고정 |
| Phase 3 | LLM 파라미터 | temperature, repetition_penalty 변경 → 모델 재생성 불필요 |
| Phase 4 | Embedding | Chunk size 변경 → **Chroma DB 재생성** 필요 |
| Phase 5 | 모두 | 모델 조합 변경 |

---

## 3. Parent-Child Document Retrieval

### 구현 방식
- **Child Chunk**: 검색에 사용 (작은 단위, 빠른 탐색)
- **Parent Chunk**: 컨텍스트 제공 (큰 단위, 풍부한 문맥)
- **Two-step retrieval**: Child로 검색 → Parent 반환

### 🔧 설정 위치
```json
// config/rag.json
{
  "parent_document_mode": true,
  "child_chunk_size": 500,
  "child_chunk_overlap": 100,
  "parent_chunk_size": 2500,
  "parent_chunk_overlap": 250
}
```

### ⚠️ 주의사항

1. **Chunk size 변경 시 Chroma DB 재생성 필수**
2. **Parent:Child 비율**: 일반적으로 1:3 ~ 1:5 권장
3. **Overlap**: Chunk size의 10-20% 권장

### 📝 관련 파일
- `app/core/vector_store.py` - `_create_parent_doc_db()`
- `app/services/rag_service.py` - `_rag_search_parent_child()`

---

## 4. Chroma DB 재생성 요구사항

### ⚠️ 반드시 재생성 필요한 경우

1. **Chunk size 변경** (child/parent 모두)
2. **Embedding 모델 변경**
3. **PDF 문서 변경**

### ✅ Phase 4에서 올바른 구현

```python
# Phase 4: Chunk Size Optimization
def force_rebuild_chroma_db():
    """Chroma DB 강제 재생성"""
    db_path = project_root / "chroma_db_new"

    if db_path.exists():
        print("🔄 Removing existing Chroma DB...")
        shutil.rmtree(db_path)  # ✅ 기존 DB 완전 삭제
        print("   ✅ Chroma DB removed")

    # GPU 메모리 클리어
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("   ℹ️  Chroma DB will be rebuilt on first query")

# 각 테스트마다
for config in chunk_configs:
    apply_chunk_config(config)
    force_rebuild_chroma_db()  # ✅ 매번 재생성
    run_bench(...)
```

### ❌ 흔한 실수

```python
# 실수: DB 재생성 없이 chunk size만 변경
apply_chunk_config(new_chunk_size)
run_bench()  # ❌ 이전 chunk size로 만들어진 DB 사용
```

### 📝 DB 재생성 흐름

1. Config 변경 → `config._load_configs()`
2. 기존 DB 삭제 → `shutil.rmtree(db_path)`
3. 벤치마크 실행 → 첫 쿼리 시 자동 재생성
4. `app/core/vector_store.py`의 `get_or_create_collection()` 호출

---

## 5. 타입 안정성 (Chroma)

### 문제점: String → Int 자동 변환 버그

Chroma의 `similarity_search(k=...)` 파라미터는 **int만 허용**하지만, Python은 문자열 연결 시 자동으로 string이 됨.

### ❌ 버그 사례

```python
# app/services/multi_query.py
top_k_per_query = 10  # int
results = rag_search_with_rerank(
    query=sub_query,
    top_k=10,  # "10" + "10" → "1010" (문자열 연결!)
    rerank_top_k=top_k_per_query
)

# 에러: Expected requested number of results to be a int, got 1010 in query
```

### ✅ 수정 방법

```python
# 모든 top_k 파라미터에 int() 명시
results = rag_search_with_rerank(
    query=sub_query,
    top_k=int(10),  # ✅ 명시적 int 변환
    rerank_top_k=int(top_k_per_query)  # ✅
)

# app/services/rag_service.py
search_results = models.collection.similarity_search(
    query,
    k=int(top_k) * 2  # ✅ 연산 전에 int() 변환
)
```

### 📝 수정된 파일
- `app/services/multi_query.py:242-243`
- `app/services/rag_service.py:130, 209`

### 💡 교훈
- **숫자 파라미터는 항상 int() 변환** (특히 외부 라이브러리 호출 시)
- **연산 전에 타입 확정** (`int(x) * 2`가 `x * 2`보다 안전)

---

## 6. 모델 검증 타이밍 (2025-11-28 수정)

### ❌ 잘못된 검증 타이밍

```python
# 노트북에서 run_bench() 전에 검증
model_info = get_actual_model_info()  # ❌ 모델이 아직 로드 안 됨
print(f"LLM: {model_info['actual_llm_model']}")  # "Not loaded"

run_bench(...)  # ✅ 여기서 실제 모델 로드
```

**문제**: 모델은 `run_bench()` → `initialize_models()` 내부에서 로드되므로, 그 전에 검증하면 "Not loaded" 표시

### ✅ 올바른 검증 타이밍

```python
# scripts/run_phase2_bench.py
def initialize_models():
    models.load_all_models()  # 1. 모델 로드

    model_info = get_actual_model_info()  # 2. 로드 직후 검증 ✅

    logging.info("Config vs Actual Models:")
    logging.info(f"Config LLM: {model_info['config_llm_model']}")
    logging.info(f"Actual LLM: {model_info['actual_llm_model']}")  # ✅ 정상 표시

    return model_info  # 3. 반환

def run_bench(...):
    model_info = initialize_models()  # 검증 정보 받기
    bench_result['model_info'] = model_info  # 4. 결과에 저장
```

```python
# 노트북에서
result_path = run_bench(...)  # 모델 로드 + 검증 완료

# 벤치마크 결과에서 모델 정보 가져오기
with open(result_path) as f:
    bench_result = json.load(f)
model_info = bench_result['model_info']  # ✅ 실제 사용된 모델 정보

print(f"LLM: {model_info['actual_llm_model']}")  # ✅ 정상 표시
```

### 📝 원칙

**"실제 모델 사용 직전"에 검증**
- 모델 로딩 함수 내부에서 로드 직후 검증
- 벤치마크 결과에 포함시켜 저장
- 노트북에서는 결과 파일에서 읽어오기

---

## 6. 노트북 환경에서 로그 출력

### 문제점: logging이 노트북 output에 안 보임

Jupyter Notebook에서 `logging.info()` 등의 로그가 **output cell에 표시되지 않을 수 있음**.

### 원인
- Jupyter는 **stdout만 capture**함
- `logging` 모듈은 기본적으로 **stderr** 또는 별도 handler로 출력
- `verbose=False`로 호출 시 logging 레벨이 INFO여도 노트북에서 안 보일 수 있음

### ✅ 해결 방법 1: logging을 stdout으로 리다이렉트 (권장)

```python
# scripts/run_phase2_bench.py - setup_logging()
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    # 기존 handler 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # stdout으로 출력 (노트북에서 보이도록) ✅
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)
```

**효과**:
- ✅ 모든 `logging.info()` 출력이 노트북 output cell에 표시됨
- ✅ `app/core/models.py`, `app/core/model_loader.py` 등의 로그도 자동으로 보임
- ✅ 코드 수정 최소화

### ✅ 해결 방법 2: logging + print 둘 다 사용 (특정 메시지만)

```python
# 특정 중요 메시지만 확실히 보이게
def initialize_models():
    models.load_all_models()
    model_info = get_actual_model_info()

    msg = [
        "=" * 60,
        "Config vs Actual Models:",
        f"Config LLM: {model_info['config_llm_model']}",
        f"Actual LLM: {model_info['actual_llm_model']}",
        "=" * 60
    ]

    for line in msg:
        logging.info(line)  # 로그 파일용
        print(line)  # 노트북용 ✅

    return model_info
```

**사용 시기**: 특정 메시지만 강조해서 보여줄 때

### 💡 원칙

**노트북에서 확실히 보여야 하는 정보:**
- ✅ **print() 사용** (stdout → output cell)
- ✅ logging도 함께 사용 (로그 파일/콘솔용)

**로그만 남기면 되는 정보:**
- ❌ logging만 사용 (디버그, 일반 진행 상황)

### 📝 적용 위치
- `scripts/run_phase2_bench.py:39-56` - `setup_logging()` - **stdout 리다이렉트 (필수)**
- `scripts/run_phase2_bench.py:149-171` - `initialize_models()` - logging + print 병용

### ❌ 흔한 실수

```python
# 노트북에서 중요한 정보를 logging만 사용
logging.info("Model verification result...")  # ❌ 노트북에서 안 보임
```

### 🔍 디버깅 팁

로그가 안 보이면:
1. `print()`로 직접 출력 확인
2. 로그 파일 확인 (있다면)
3. `verbose=True`로 변경해서 테스트

---

## 🔄 문서 업데이트 규칙

1. **새로운 구현 특이사항 발견 시** → 즉시 이 문서에 추가
2. **반복적인 실수 발견 시** → "흔한 실수" 섹션에 기록
3. **구현 방식 변경 시** → 해당 섹션 업데이트 + 날짜 표시

---

## 📚 참고 문서

- [agents.md](.agents/agents.md) - 프로젝트 현재 상태 및 작업 기록
- [agents_meta.md](.agents/agents_meta.md) - 프로젝트 메타데이터 및 아키텍처
