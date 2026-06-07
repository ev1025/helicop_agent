# LLM 개선 자동화 - 결과 분석 가이드

**목적**: `results/testcase_XXXX.json`으로 생성되는 벤치마크를 일관된 방식으로 평가하고, 수치적 변화뿐 아니라 실제 답변 품질과 언어적 완성도를 동시에 점검하기 위한 절차를 정의한다.

---

## 1. 분석 준비
1. `ls -t results/testcase_*.json | head -n 2`로 최신 파일과 직전 파일을 확인한다.
2. `python` 스크립트 또는 노트북으로 두 파일을 로드하여 동일한 키(`metrics`, `timings`, `rewrite`)를 비교할 수 있게 한다.
3. 분석 산출물은 `.agents/projects/llm_improvement_automation/analysis/testcase_XXXX_analysis.md`로 저장하며, 보고서 상단에 “주요 변경 요약”을 포함한다.

---

## 2. 필수 평가 항목
| 구분 | 내용 |
|------|------|
| **정량 지표** | Precision, Recall, F1, Keyword Overlap, Source Count, Retrieval/Generation Latency. 이전 테스트케이스와의 차이와 변화율을 표로 제시한다. |
| **Decomposition 통계** | `rewrite.sub_questions` 개수, `complexity` 비율, RAG 사용률을 산출하여 관련 변경(예: prompt rewriting)과 연결한다. |
| **케이스별 진단** | 최상위/최하위 F1 케이스, 회귀(regression) 케이스를 선정하여 원인(검색 실패, 답변 누락, 프롬프트 이슈 등)을 기록한다. |
| **언어 품질** | 답변 본문에서 매끄럽지 못한 문장, 의미 없는 공백 괄호, 한자/영문/베트남 알파벳 등 비한글 토큰이 과도하게 섞인 경우를 찾아 케이스별로 정리한다. 필요시 `re` 기반 패턴(`A-Za-z`, `\u4E00-\u9FFF`, `\u00C0-\u024F`)을 이용해 자동 감지하고 샘플 스니펫을 첨부한다. |

---

## 3. 보고서 구성 권장안
1. **요약**: 변경사항(예: Prompt Rewriting 개선)과 핵심 결과를 bullet 2-3줄로 작성.
2. **전체 지표 비교 표**: 최신 vs 직전 테스트케이스 수치를 % 단위로 정리.
3. **Decomposition/지연 통계**: 질문 수 대비 적용 비율, 평균 latency 변화를 해석.
4. **언어 품질 점검**: 문제 케이스 목록 + 발화 예시 + 영향.
5. **케이스별 하이라이트**: 개선/악화 케이스 2~3개씩 정리.
6. **권장 조치**: 다음 실험/코드 수정 제안.

---

## 4. 언어 품질 판정 기준
- **불필요한 영문/기호**: 단순 용어 설명을 넘어 본문이 영어 위주이거나 `'s`, `(of …)` 같은 잔여 토큰이 반복적으로 등장할 경우 경고.
- **한자/기타 문자 삽입**: `力的`, `的` 등 원문에서 오염된 한자가 산발적으로 섞이면 원인 추적 필요.
- **베트남/라틴 확장 알파벳**: `đ`, `tự động`, `cần` 등이 나타나면 프롬프트/토큰화 이상 징후로 분류.
- **매끄럽지 않은 문장**: 주어/서술어가 빠진 bullet, “** **”처럼 비어 있는 토큰, 같은 문장 반복 등은 모두 품질 이슈로 기록.

---

## 5. 기술적 참조 사항

### JSON 구조

**메트릭 필드명 (정확한 이름 사용 필수)**:
- ✅ `f1_score` (NOT `f1`)
- ✅ `precision`
- ✅ `recall`
- ✅ `keyword_overlap`
- ✅ `source_count`

**타이밍 정보**:
- `timings.retrieval_ms` - 검색 시간 (밀리초)
- `timings.generation_ms` - 생성 시간 (밀리초)

**케이스 구조**:
```json
{
  "id": "Q0",
  "question": "...",
  "ideal_answer": "...",
  "answer": "...",
  "metrics": {
    "f1_score": 0.1234,  // ⚠️ f1이 아님!
    "precision": 0.1234,
    "recall": 0.1234,
    ...
  },
  "timings": {
    "retrieval_ms": 123.45,
    "generation_ms": 456.78
  }
}
```

### 분석 파일 네이밍 및 비교 원칙

- **저장 위치**: `.agents/projects/llm_improvement_automation/analysis/`
- **파일명 형식**: `testcase_XXXX.md` (NOT `testcase_XXXX_analysis.md`)
- **⚠️ 필수 비교**: 모든 testcase 분석은 **이전 최신 testcase와 반드시 비교**
- **비교 항목**: 전체 메트릭, 케이스별 F1 변화, 회귀/개선 케이스 목록

### 언어 품질 정규식

```python
# 빈 볼드
r'\*\*\s+\*\*'

# 빈 괄호 (다양한 패턴)
r'\(\s*(?:of|in|the|for)?\s*\)'
r'\(\s*\)'
r'\[\s*\]'

# 한자 (CJK Unified Ideographs)
r'[\u4E00-\u9FFF]'

# 베트남/라틴 확장
r'[\u00C0-\u024F]'

# 영어 잔여물
r'(?<!\w)(in|of|the|for|\'s)(?!\w)'
```

---

---

## 6. 5가지 종합 평가 기준 (2025-11-10 추가)

**목적**: LLM 답변 품질을 체계적으로 정량화하여 testcase 간 비교 가능한 점수 체계 제공

**적용 범위**: testcase_0000 이후 모든 테스트

### 평가 항목 (각 100점 만점)

#### 1. 조사 단독 출현 및 빈 괄호 (Grammatical Particles Issue)

**검출 대상**:
- 조사만 단독으로 나오는 경우: `은 `, `는 `, `이 `, `가 `, `을 `, `를 `, `에 `, `의 `, 등
- 빈 괄호: `()`, `( )`, `[]`, `[ ]`

**점수 계산**:
```
점수 = 100 - (조사_단독_개수 × 15) - (빈_괄호_개수 × 10)
점수 = max(0, 점수)
```

**평가 도구**: `scripts/comprehensive_evaluation.py` - `evaluate_grammatical_particles()`

#### 2. 반복 문구 (Exact Repetition)

**검출 대상**:
- 완전히 동일한 문장 (10자 이상) 2회 이상 반복
- 완전히 동일한 구절 (5자 이상) 3회 이상 반복

**점수 계산**:
```
점수 = 100 - (문장_반복_개수 × 20) - (구절_반복_개수 × 10)
점수 = max(0, 점수)
```

**평가 도구**: `scripts/comprehensive_evaluation.py` - `evaluate_exact_repetition()`

#### 3. 유사 표현 중복 (Semantic Redundancy)

**검출 대상**:
- 의미가 동일한 표현의 반복 (Jaccard 유사도 > 0.5)
- 불필요한 부연 설명

**점수 계산**:
```
점수 = 100 - (유사_쌍_개수 × 15)
점수 = max(0, 점수)
```

**평가 도구**: `scripts/comprehensive_evaluation.py` - `evaluate_semantic_redundancy()`

#### 4. 장황함 (Verbosity)

**평가 기준**: 이상적 답변 대비 길이 비율

**점수 범위**:
- 0.8~1.2x: 100점 (완벽)
- 1.5~2.0x: 60-80점 (장황)
- 3.0x 이상: 0-30점 (극도로 장황)

**평가 도구**: `scripts/comprehensive_evaluation.py` - `evaluate_verbosity()`

#### 5. RAG 기반 답변 품질 (RAG-based Answer Quality)

**평가 요소**:
- F1 Score: 40점
- RAG 활용도: 30점 (RAG 문서에서 정보 추출 정도)
- Precision: 15점
- Recall: 15점
- 보너스/페널티: 출처 태그 ±5점, 환각 -10점

**기준 RAG 문서**: testcase_0020의 RAG 검색 결과 사용 (일관성 유지)

**평가 도구**: `scripts/comprehensive_evaluation.py` - `evaluate_rag_quality()`

### 종합 점수 산출 (가중 평균)

```
종합_점수 = (
    조사_빈괄호_점수 × 0.10 +
    반복_문구_점수 × 0.10 +
    유사_표현_점수 × 0.10 +
    장황함_점수 × 0.15 +
    RAG_기반_답변_품질 × 0.55
)
```

**가중치 근거**:
- **RAG 기반 답변 품질 (55%)**: 질문에 올바르게 답했는지가 최우선
- **장황함 (15%)**: 사용자 경험에 직접 영향
- **조사/반복/유사표현 (각 10%)**: 언어 품질 기본 요소

### 등급 체계

| 점수 범위 | 등급 | 평가 |
|----------|------|------|
| 90-100점 | S | 탁월 - 거의 완벽한 답변 |
| 80-89점 | A | 우수 - 대부분 만족스러운 답변 |
| 70-79점 | B | 양호 - 약간의 개선 필요 |
| 60-69점 | C | 보통 - 여러 부분 개선 필요 |
| 50-59점 | D | 미흡 - 상당한 개선 필요 |
| 0-49점 | F | 불합격 - 전면 재검토 필요 |

### 자동 평가 도구

**스크립트**: `scripts/comprehensive_evaluation.py`

**사용법**:
```bash
# 단일 testcase 평가 (자체 RAG 사용)
python scripts/comprehensive_evaluation.py results/testcase_0024.json

# 참조 RAG 기준 평가 (testcase_0020의 RAG 사용)
python scripts/comprehensive_evaluation.py results/testcase_0024.json results/testcase_0020.json

# 출력: JSON 형식으로 5가지 점수, 종합 점수, 상세 분석 정보
```

**출력 예시**:
```json
{
  "testcase_id": "testcase_0024",
  "avg_scores": {
    "grammatical_particles": 100.0,
    "exact_repetition": 78.6,
    "semantic_redundancy": 93.6,
    "verbosity": 66.6,
    "rag_quality": 6.5,
    "comprehensive": 40.8
  }
}
```

### 상세 가이드 참조

**전체 평가 기준 및 예시**: `.agents/projects/llm_improvement_automation/analysis/evaluation_guidelines.md`
- 각 항목별 상세 평가 방법
- 점수 계산 예시
- 평가 도구 사용법
- 비교 분석 방법

---

## 7. 변경 이력
- **2025-11-07**: testcase_0013 분석부터 언어 품질 점검과 보고서 상단 Summary를 필수 항목으로 지정 (작성: Codex).
- **2025-11-07 (후반)**: JSON 구조 참조 및 필수 비교 원칙 추가. 언어 품질 정규식 명시.
- **2025-11-10**: 5가지 종합 평가 기준 추가 (조사/반복/유사표현/장황함/RAG품질). 자동 평가 도구 개발 (comprehensive_evaluation.py).
