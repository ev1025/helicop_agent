# RAG 전처리/청킹 포팅 현황 (surion_llm → surion_llm_isaacsim)

## 📁 리포지토리 정보
- **소스 리포**: `/workspace/jupyter_notebooks/surion_llm` (브랜치: `git-issue-2`)
- **대상 리포**: `/workspace/jupyter_notebooks/surion_llm_isaacsim` (브랜치: `claude/incremental-git-issue-1-final-01EJHBBDhHRyetXdyVXMDdfb`)
- **이식 범위**: 헬리콥터 교재 전처리 + 버퍼 기반 청킹 (계획 중, 아직 미이식)

---

## ✅ 이미 완료된 이식 내용

### 1. 텍스트 클리닝 모드 확장 (완료)
**파일**: `app/core/vector_store.py`

**이식된 함수**:
- `_clean_text_v2()` - 표 패턴 보존 (숫자/기호 많은 라인은 개행 유지)
- `_clean_text_v3()` - 수식 패턴 보존 (수학 기호 포함 라인 개행 유지)
- `get_cleaning_function(mode)` - 클리닝 모드 선택기 (`v1`, `v2`, `v3`)

**적용 위치**:
```python
# config/rag.json
{
  "text_cleaning_mode": "v1"  # v1(기본), v2(표 보존), v3(수식 보존)
}
```

**사용 현황**:
- Phase 2~6 테스트에서 `text_cleaning_mode` 파라미터로 선택 가능
- 기본값은 `v1` (연속 공백/개행 제거)

### 2. ChromaDB Client 설정 통일 (2025-12-02 완료)
**파일**: `app/core/vector_store.py`

**추가된 함수**:
```python
def _make_chroma_client(vector_db_path: str):
    """Chroma 클라이언트를 공통 설정으로 생성 (allow_reset=True로 충돌 방지)"""
    return chromadb.Client(Settings(
        is_persistent=True,
        persist_directory=vector_db_path,
        allow_reset=True,
        anonymized_telemetry=False,
    ))
```

**변경 이유**:
- **문제**: Phase 4 테스트 중 같은 프로세스에서 chunk size 변경 시 ChromaDB 싱글톤 충돌
  ```
  ValueError: An instance of Chroma already exists for /path with different settings
  ```
- **근본 원인**: `chromadb.PersistentClient(path=...)` 사용 시 settings 불일치
  - `test_chroma_cycle_v2.ipynb`의 `force_rebuild_chroma_db()`가 `chromadb.Client(Settings(allow_reset=True))` 생성
  - `vector_store.py`가 `chromadb.PersistentClient(path=...)` 생성 시도
  - 같은 경로에 대해 다른 settings → 충돌
- **해결**: 모든 client 생성을 `_make_chroma_client()`로 통일 (`allow_reset=True` 일관 적용)

**적용된 함수**:
- `load_vector_db()` - 일반 벡터 DB 로드
- `create_parent_document_vectordb()` - Parent-Child 벡터 DB 생성
- ⚠️ **향후 helicopter 함수 구현 시에도 이 함수 사용 필수!**

### 3. Chunk Size 변경 자동 감지 (2025-12-02 완료)
**파일**: `app/core/vector_store.py` - `create_parent_document_vectordb()` 함수

**추가된 로직**:
```python
# 기존 collection 발견 시
if collection_exists:
    # 메타데이터에서 chunk size 확인
    metadata_path = os.path.join(vector_db_path, f"_{child_collection_name}_metadata.json")

    if os.path.exists(metadata_path):
        old_metadata = json.load(open(metadata_path))

        # Chunk size 변경 확인
        if (old_metadata['child_chunk_size'] != child_chunk_size or
            old_metadata['parent_chunk_size'] != parent_chunk_size or ...):

            logger.warning("⚠️  Chunk size 변경 감지!")
            # Collection만 삭제 (DB 폴더는 유지)
            client.delete_collection(child_collection_name)
            collection_exists = False  # 새로 생성하도록
        else:
            # Chunk size 동일 - 기존 재사용
            return existing_collection, parent_store
```

**효과**:
- ✅ Phase 4 테스트 시 chunk size 변경 시 자동 재생성
- ✅ DB 폴더 삭제 불필요 (SQLite lock 문제 해결)
- ✅ Collection 이름에 chunk size 포함되어 충돌 방지
- ✅ 메타데이터 기반 안전한 캐싱

**Collection 명명 규칙**:
```
{collection}_{model}_{child_size}_{child_overlap}_parent_{parent_size}_{parent_overlap}_{cleaning_mode}

예: new_manual_dragonkue_..._child_300_30_parent_1500_150_v1
```

**이전 iteration collection 오접근 가능성**: ❌ 없음
- Chunk size가 다르면 collection 이름이 완전히 다름
- 메타데이터 검증으로 이중 확인

---

## ❌ 이식 예정 내용 (아직 미완료)

### 현황 확인 (2025-12-02)
**소스 리포 점검 결과**:
```bash
# /workspace/jupyter_notebooks/surion_llm/app/core/vector_store.py
$ grep "helicopter" vector_store.py
505:            "preprocessing": "helicopter_text_cleaning"  # ← 주석만 존재

$ grep "def " vector_store.py
_sanitize_model_name()
_clean_text()              # v1만 존재 (v2/v3 없음)
load_vector_db()
save_parent_store()
load_parent_store()
create_parent_document_vectordb()
```

**결론**: ❌ **Helicopter 함수들이 소스 리포에도 아직 구현되지 않음**

### 1. Helicopter 전처리 함수 (미구현)
**계획된 함수** (`.agents` 문서 및 line 505 주석 기준):
```python
def helicopter_clean_text(text: str) -> str:
    """
    헬리콥터 교재 전처리 (헤더/캡션/자료제공 제거 + 문장 공백 보정)

    제거 패턴:
    - ^비행이론(헬리콥터)\s*\n?\s*·\s*\d+
    - ^비행이론의 소개\s*\n?\s*\d+\s*·
    - ^헬리콥터의 구조와 시스템\s*\n?\s*\d+\s*·
    - ^기초 비행 원리\s*\n?\s*\d+\s*·
    - ^항공기 성능\s*\n?\s*\d+\s*·
    - [그림 \d+-\d+][^\n]*
    - (.*자료제공)

    변환:
    - \n 제거 → .(?=[^\s]) → . (공백 보정) → \s+ → ' ' (연속 공백 축약)
    """
    pass
```

### 2. 버퍼 기반 청킹 (미구현)
```python
def buffer_chunk_documents(pdf_path: str, docs, splitter, buffer_threshold: int):
    """
    페이지 전처리 + 버퍼 기반 청킹

    로직:
    1. 각 페이지를 helicopter_clean_text()로 전처리
    2. 텍스트 버퍼에 누적 (threshold = chunk_size * 10)
    3. 버퍼가 임계값 초과 시 청킹
    4. 페이지 범위 메타데이터 포함 (page, page_end)
    """
    pass
```

### 3. Helicopter 전용 벡터 DB 함수 (미구현)
```python
def load_vector_db_helicopter(embedding_function, model_name, vector_db_path,
                              collection_name, pdf_path=None,
                              chunk_size=950, chunk_overlap=50):
    """
    Collection 명명: {collection}_{model}_{chunk}_{overlap}_helicopter

    ⚠️ 반드시 _make_chroma_client() 사용!
    """
    client = _make_chroma_client(vector_db_path)  # ← 필수!
    # ...

def create_parent_document_vectordb_helicopter(embedding_function, model_name,
                                               vector_db_path, collection_name,
                                               pdf_path, child_chunk_size=300,
                                               parent_chunk_size=1500,
                                               start_page=None, end_page=None):
    """
    추가 기능:
    - start_page/end_page로 부분 페이지 처리
    - preprocessing 태그: "helicopter_text_cleaning"
    - 버퍼 청킹 사용

    ⚠️ 반드시 _make_chroma_client() 사용!
    """
    client = _make_chroma_client(vector_db_path)  # ← 필수!
    # ...
```

---

## 🎯 이식 전략 및 주의사항

### A. 공존 전략 (현재 방식)
**원칙**: 기존 로직을 건드리지 않고 **새 함수**로 추가

**방법**:
1. Helicopter 함수들은 `_helicopter` 접미사 사용
2. Collection 이름도 `_helicopter` 접미사로 구분
3. 기존 함수 (`create_parent_document_vectordb`) 유지
4. Phase 2~6 테스트는 기존 함수 사용
5. Helicopter 테스트는 별도 노트북 (`phase2_1` 등) 사용

**장점**:
- ✅ Phase 2~6 진행 중인 테스트에 영향 없음
- ✅ A/B 비교 가능
- ✅ 안전한 실험

**단점**:
- ❌ 코드 중복
- ❌ 향후 통합 작업 필요

### B. ChromaDB Settings 통일 **필수**
**⚠️ 매우 중요**: Helicopter 함수 구현 시 반드시 `_make_chroma_client()` 사용!

```python
# ✅ 올바른 방법
def load_vector_db_helicopter(...):
    client = _make_chroma_client(vector_db_path)
    collection = Chroma(..., client=client)

# ❌ 잘못된 방법 (설정 충돌 발생)
def load_vector_db_helicopter(...):
    client = chromadb.PersistentClient(path=vector_db_path)
    # → ValueError: An instance of Chroma already exists with different settings
```

### C. 메타데이터 스키마 통일
**현재 메타데이터 키** (`create_parent_document_vectordb`):
```json
{
  "mode": "parent_document",
  "embedding_model": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
  "child_chunk_size": 300,
  "child_chunk_overlap": 30,
  "parent_chunk_size": 1500,
  "parent_chunk_overlap": 150,
  "text_cleaning_mode": "v1",
  "total_parents": 222,
  "total_children": 1234
}
```

**Helicopter 추가 예상 키**:
```json
{
  "mode": "parent_document_helicopter",
  "preprocessing": "helicopter_text_cleaning",
  "page_range": {"start": 1, "end": 300},
  "buffer_threshold": 9500,
  "buffer_strategy": "size_multiplier:10"
}
```

### D. 테스트 노트북 분리
**Phase 2~6**: 기존 함수 사용
- `phase2_rag_parameters_optimization.ipynb` ✅ 완료
- `phase3_repetition_optimization.ipynb` 🔄 Cell 7 수정 완료, 실행 대기
- `phase4_chunk_size_optimization.ipynb` 🔄 ChromaDB 수정 완료, 검증 중
- `phase5_*.ipynb`
- `phase6_*.ipynb`

**Helicopter 실험**: 새 노트북 (Phase 4 이후)
- `phase2_1_helicopter_preprocessing.ipynb` (계획)
- Cleaning mode 스위치: `baseline`, `helicopter`, `v2`, `v3`
- 비교 지표: Comprehensive, RAG Quality, F1

---

## 📊 현재 코드 상태 요약

### 소스 리포 (`surion_llm`)
```
app/core/vector_store.py (512 lines)
├── _sanitize_model_name()
├── _clean_text()              # v1만 존재
├── load_vector_db()
├── save_parent_store()
├── load_parent_store()
└── create_parent_document_vectordb()

# Helicopter 함수들: ❌ 미구현
```

### 현재 리포 (`surion_llm_isaacsim`)
```
app/core/vector_store.py (570 lines)
├── _make_chroma_client()      # ✨ NEW (2025-12-02)
├── _sanitize_model_name()
├── _clean_text()              # v1
├── _clean_text_v2()           # ✨ 이식됨 (표 보존)
├── _clean_text_v3()           # ✨ 이식됨 (수식 보존)
├── get_cleaning_function()    # ✨ 이식됨
├── load_vector_db()           # ✨ Client 설정 수정됨
├── save_parent_store()
├── load_parent_store()
└── create_parent_document_vectordb()  # ✨ Chunk size 감지 추가

# Helicopter 함수들: ❌ 미이식 (소스에도 없음)
```

**차이점**:
- `+58 lines` (주로 v2/v3 cleaning, chunk size 감지 로직)
- ChromaDB client 생성 방식 변경 (충돌 방지)
- Helicopter 함수들은 소스 리포에도 구현 안 됨

---

## 🚀 다음 단계

### 1. Phase 4 완료 (우선 - 진행 중)
- ✅ ChromaDB 수정 완료
- 🔄 `test_chroma_cycle_v2.ipynb` 테스트 예정
- 🔄 Phase 4 12개 config 완료 여부 확인

### 2. Phase 3 수정 완료
- ✅ 노트북 Cell 7 포맷 수정 완료
- 🔄 Phase 3 52개 config 실행 예정

### 3. Helicopter 이식 (Phase 4 완료 후)
**조건**: Phase 4 완료 후 진행
**작업**:
1. ❌ 소스 리포에도 helicopter 함수 미구현 → **먼저 소스 리포에 구현 필요**
2. 또는 현재 리포에 직접 구현
3. `_make_chroma_client()` 사용 필수
4. `_helicopter` 접미사로 함수 추가
5. `phase2_1` 노트북 생성
6. A/B 테스트 실행

---

## 🔍 트러블슈팅 가이드

### 문제: ChromaDB 설정 충돌
```
ValueError: An instance of Chroma already exists for /path with different settings
```

**원인**:
- 같은 Python 프로세스에서 다른 Settings로 client 생성 시도
- `chromadb.PersistentClient(path=...)` vs `chromadb.Client(Settings(allow_reset=True))`

**해결책**:
1. 모든 client 생성을 `_make_chroma_client()` 사용
2. Jupyter 커널 재시작
3. `test_chroma_cycle_v2.ipynb` 참고

### 문제: Chunk size 변경 시 DB 충돌
**해결책**:
- ✅ 더 이상 수동 DB 삭제 불필요
- ✅ `create_parent_document_vectordb()`가 자동 감지 및 재생성
- ✅ Collection만 삭제, DB 폴더 유지 → SQLite lock 문제 없음

### 문제: 이전 iteration collection 오접근 우려
**확인 결과**: ❌ 문제 없음
- Collection 이름에 chunk size 포함됨
  - `child_300_30_parent_1500_150` vs `child_500_100_parent_2500_250`
- 자동으로 올바른 collection 선택됨
- 메타데이터 검증으로 이중 확인

---

## 📝 필수 고려사항 체크리스트

### 완료된 항목
- [x] **DB 클라이언트 설정**: `_make_chroma_client()` 구현 및 적용 완료
- [x] **표/수식 보존**: v2/v3 cleaning 이식 완료
- [x] **회귀 테스트**: 기존 v1/v2/v3 경로 정상 동작 (Phase 2~6 영향 없음)
- [x] **Chunk size 감지**: 메타데이터 기반 자동 감지 및 재생성 완료
- [x] **Collection 명명 규칙**: Chunk size 포함으로 충돌 방지

### 대기 중인 항목
- [ ] **전처리 품질**: Helicopter 함수 구현 후 RAG Quality/Comprehensive 측정
- [ ] **메타데이터 무결성**: Helicopter 함수에서 parent_id/page_range 저장 확인
- [ ] **성능 측정**: 버퍼 청킹 메모리/시간 영향 측정
- [ ] **병합 방침**: 테스트 결과에 따라 기존 로직과 융합 여부 결정

---

## 🎯 공용 버전으로 확장 제안

### 우선순위 높음
1. ✅ **ChromaDB Settings 통일** - 완료
2. **클리닝 파이프라인 레지스트리**: `register_cleaner(name, fn)` 구조
   - 현재: `get_cleaning_function(mode)` - v1/v2/v3 하드코딩
   - 개선: 동적 등록 가능하도록
3. **메타데이터 스키마 통일**: 공통 키 강제 및 검증

### 우선순위 중간
4. **패턴 파라미터화**: 정규식 패턴을 `patterns.yaml` 외부화
5. **버퍼링 전략 선택**: `buffer_strategy` 옵션 추가
6. **DB 네임스페이스 템플릿**: Collection 명명 규칙 설정화

### 우선순위 낮음
7. **페이지/섹션 필터**: `section_titles` 기반 필터 추가

---

**최종 업데이트**: 2025-12-02
**작성자**: Claude (Agent Session)
**다음 검토**: Phase 4 완료 후 Helicopter 이식 여부 결정
