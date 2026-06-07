# Phase 6 상태 - 100시간 종합 테스트

**업데이트:** 2025-11-20
**현재:** Phase 6 세팅 완료, 테스트 미실행

---

## ✅ 완료: Phase 5 - 통합 최적화

### 결과
- **최고 점수**: 55.54
- **최고 설정**:
  - Embedding: `BAAI/bge-large-en-v1.5`
  - Rerank: `BAAI/bge-reranker-base` (top_k=3)
  - Child Chunk: 500/100
  - Parent Chunk: 2500/250
- **RAG 품질**: 26.87 (목표 40에서 13.13점 부족)
- **결론**: 60점 달성 위해 Phase 6 필요

---

## 🚀 준비 완료: Phase 6 - 100시간 종합 테스트

### 목표
**7단계 전략적 테스트로 60점+ 달성**

### 테스트 전략
각 단계에서 동적 top-N 선택 (상대 2% + 통계적 유의성):

1. **Stage 1**: LLM Model Selection (3h) - 7개 모델 (8B~70B)
2. **Stage 2**: Chunk Size Optimization (8h)
3. **Stage 3**: Retrieval Parameter Tuning (6h)
4. **Stage 4**: Algorithm Improvement (4h) - text_cleaning_mode v1/v2/v3
5. **Stage 5**: Prompt Optimization (3h) - 4가지 변형
6. **Stage 6**: LLM Generation Parameters (4h) - temperature × repetition_penalty
7. **Stage 7**: Final Combination Test (40-60h) - 전체 조합 (2^6 = 64개)

**총 예상**: 70-100시간

### 구현 상태

#### ✅ 완료
1. **Prompt 변형 시스템** (`config/prompts.json`)
   - 10가지 프롬프트 변형 추가 (`_prompt_variants`)
   - baseline, concise_v1/v2, no_repeat, rag_focus, balanced_v1~v4, bullet_points

2. **Text Cleaning Mode 시스템** (`app/core/vector_store.py`)
   - v1: 기본 (모든 개행 → 공백)
   - v2: 표 보존 (숫자/기호 50%+ 라인 → 개행 유지)
   - v3: 수식+표 보존 (수식 기호, 표 패턴, 숫자 30%+ → 개행 유지)
   - 각 모드별 별도 collection 생성

3. **Phase 6 노트북** (`jupyter_notebooks/phase6_comprehensive_100h_test.ipynb`)
   - 7단계 전체 구현 (25개 셀)
   - apply_test_config() 함수: prompt variant, text_cleaning_mode 지원
   - 동적 top-N 선택 알고리즘
   - 단계별 진행 상황 자동 저장
   - 최종 분석 및 상세 통계

4. **설정 파일** (`temp_config_phase6.json`)
   - 7개 LLM 모델 (Llama-3, Llama-3.1, Qwen2.5, Mixtral)
   - Chunk 조합 (3×4 = 12가지)
   - Retrieval 파라미터 (3×3 = 9가지)
   - 3가지 text cleaning mode
   - 4가지 prompt variant
   - 3×3 LLM generation params

---

## ⚠️ 최종 확정 시 추가 필요 사항

### Context Window 제약사항 명시 필요

**현재 상태**: 암묵적으로 안전하지만 명시되지 않음

#### 1. Embedding 모델 토큰 제한

| 모델 | Max Tokens | Chunk Size 제약 |
|------|-----------|----------------|
| BAAI/bge-large-en-v1.5 | 512 | 영어: ~1750자, 한국어: ~1200자 |
| BAAI/bge-m3 | 8192 | 제약 없음 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 32768 | 제약 없음 |

**현재 설정**: child_chunk_size 최대 700자 → **모든 모델 안전** ✅

#### 2. LLM Context Window 제한

| LLM 모델 | Context Window | 최대 허용 RAG Context |
|---------|----------------|---------------------|
| Llama-3-8B | 8,192 | ~20,000자 (5K 토큰) |
| Llama-3.1-8B | 128,000 | 제약 없음 |
| Llama-3.1-70B | 128,000 | 제약 없음 |
| Qwen2.5-14B | 32,768 | ~80,000자 (20K 토큰) |
| Qwen2.5-32B | 32,768 | ~80,000자 (20K 토큰) |
| Qwen2.5-72B | 128,000 | 제약 없음 |
| Mixtral-8x7B | 32,768 | ~80,000자 (20K 토큰) |

**최악의 시나리오**:
- vector_search_top_k: 30개
- parent_chunk_size: 4,000자
- **Total**: 30 × 4,000 = 120,000자 ≈ 30,000-60,000 토큰
- **Llama-3-8B는 초과** ❌

**실제 안전한 이유**:
- `rerank_top_k` 최종 필터: 최대 7개
- **실제 전달**: 7 × 4,000 = 28,000자 ≈ 7,000-14,000 토큰
- **모든 모델 안전** ✅

#### 3. 최종 확정 시 검증 체크리스트

**프로덕션 배포 전 필수 검증**:

```python
# 1. Embedding 모델 검증
def validate_embedding_limits(chunk_size: int, embedding_model: str):
    """청크 크기가 임베딩 모델 토큰 제한 내인지 검증"""
    model_limits = {
        "BAAI/bge-large-en-v1.5": 512,
        "BAAI/bge-m3": 8192,
        "Alibaba-NLP/gte-Qwen2-7B-instruct": 32768,
    }
    max_tokens = model_limits.get(embedding_model, 512)
    estimated_tokens = chunk_size * 0.25  # 영어 기준 (한국어는 0.4)

    assert estimated_tokens <= max_tokens, \
        f"Chunk size {chunk_size} exceeds {embedding_model} limit ({max_tokens} tokens)"

# 2. LLM Context Window 검증
def validate_llm_context(rerank_top_k: int, parent_chunk_size: int, llm_model: str):
    """최종 RAG context가 LLM context window 내인지 검증"""
    model_limits = {
        "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
        "meta-llama/Meta-Llama-3.1-8B-Instruct": 128000,
        "meta-llama/Meta-Llama-3.1-70B-Instruct": 128000,
        "Qwen/Qwen2.5-14B-Instruct": 32768,
        "Qwen/Qwen2.5-32B-Instruct": 32768,
        "Qwen/Qwen2.5-72B-Instruct": 128000,
        "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    }
    max_tokens = model_limits.get(llm_model, 8192)

    # 한국어 기준: 1자 ≈ 0.5토큰, 여유 2배
    estimated_tokens = rerank_top_k * parent_chunk_size * 0.5 * 2

    assert estimated_tokens <= max_tokens * 0.8, \
        f"RAG context ({estimated_tokens} tokens) exceeds {llm_model} limit ({max_tokens})"

# 3. 전체 파이프라인 검증
def validate_phase6_config(config: dict):
    """Phase 6 설정 전체 검증"""
    # Embedding 검증
    for chunk_size in config['chunk_configs']['child_chunk_size']:
        validate_embedding_limits(chunk_size, config['embedding_model'])

    # LLM 검증
    for llm_model in config['llm_models']:
        for rerank_k in config['retrieval_configs']['rerank_top_k']:
            for parent_size in config['chunk_configs']['parent_chunk_size']:
                validate_llm_context(rerank_k, parent_size, llm_model)

    print("✅ All validations passed!")
```

**config/models.json에 추가 권장**:
```json
{
  "_model_limits": {
    "embedding_models": {
      "BAAI/bge-large-en-v1.5": {
        "max_tokens": 512,
        "max_chunk_size_chars": 1750
      }
    },
    "llm_models": {
      "meta-llama/Meta-Llama-3-8B-Instruct": {
        "context_window": 8192,
        "max_rag_context_tokens": 6000
      }
    }
  }
}
```

---

## 📊 예상 결과

### 현재 (Phase 5)
- **종합**: 55.54
- **RAG**: 26.87
- **목표 Gap**: 4.46점

### 목표 (Phase 6)
- **종합**: **60.0+** ✅
- **RAG**: **35.0+**
- **LLM**: 70B 모델 사용

### 예상 개선
| 항목 | 개선치 | 근거 |
|------|--------|------|
| LLM 70B | +5~10점 | RAG 품질 대폭 향상 |
| Chunk 최적화 | +1~2점 | 더 나은 조합 발견 |
| Retrieval 튜닝 | +1~2점 | top_k 최적화 |
| Text Cleaning v2/v3 | +1~2점 | 표/수식 보존 |
| Prompt 최적화 | +1~2점 | 반복/환각 감소 |
| LLM Params | +0.5~1점 | temperature/rep_pen |
| **총 예상** | **+9~18점** | - |
| **최종 예상** | **64~73점** 🎯 | **목표 달성** |

---

## 🔧 파일 및 커밋

### 파일
- 노트북: `jupyter_notebooks/phase6_comprehensive_100h_test.ipynb`
- 설정: `temp_config_phase6.json`
- Prompt 변형: `config/prompts.json` (`_prompt_variants`)
- Text Cleaning: `app/core/vector_store.py` (v1/v2/v3)
- 가이드: `PHASE6_IMPLEMENTATION_GUIDE.md`
- 결과 (생성 예정): `results/phase6/*.json`, `results/phase6/stage*_summary.csv`

### 커밋
```bash
56a308d - Phase 6 노트북 수정: 올바른 셀 구조로 재작성
5a0fdd0 - Phase 6 완전 구현: Stage 5-6 추가 및 전체 통합
c71d575 - Phase 6: Prompt 변형 & Text Cleaning Mode 시스템 완성
0d9a110 - Phase 6: 100시간 종합 테스트 인프라 구축
b061c18 - Phase 5 베스트 결과 확인 노트북 추가
```

---

## 🎯 다음 단계

### Phase 6 실행
```bash
cd jupyter_notebooks
jupyter notebook phase6_comprehensive_100h_test.ipynb
```

### 실행 후 작업
1. **최고 조합 확인**: `results/phase6/stage7_final_summary.csv`
2. **설정 파일 업데이트**:
   - `config/models.json` - 최고 LLM, generation params
   - `config/rag.json` - 최고 chunk, retrieval params, text_cleaning_mode
   - `config/prompts.json` - 최고 prompt variant를 기본으로
3. **검증 코드 추가**: Context window 제약사항 검증 (위 체크리스트)
4. **프로덕션 배포**

### Phase 7 (필요 시)
- 60점 미달 시: Hybrid Search (BM25 + Vector)
- 추가 개선: LLM Fine-tuning, 답변 후처리

---

**마지막 업데이트:** 2025-11-20 (Phase 6 세팅 완료)
