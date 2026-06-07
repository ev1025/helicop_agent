# 방법0

# RAG + LLM 테스트 결과 분석 보고서

## 📊 테스트 개요
- **대상**: 4가지 난이도별 쿼리 유형 (무관 질의, 저신뢰 히트, 정확 매칭, 다문서 통합)
- **총 문항**: 10개 (완료 8개, 진행중 2개)
- **환경**: CPU 전용, 생성 파라미터 max_tokens=1024, temp=0.6, top_p=0.85

---

## 🔴 핵심 성능 지표

### ⏱️ 속도 분석
| 구간 | 평균 시간 | 비고 |
|------|----------|------|
| **벡터 검색** | 1.2초 | 양호 |
| **리랭킹** | 8.7초 | ⚠️ 병목 구간 |
| **토큰 생성 시간(TTFT)** | 39.2초 | 🔴 심각 (목표: 1-3초) |
| **스트리밍 수행** | 3-11분 | 🔴 심각 |
| **평균 생성 속도** | 0.45 tok/s | 🔴 심각 (목표: 20+ tok/s) |

### 📄 RAG 검색 성능
- **RAG 활성화율**: 25% (8건 중 2건만 문서 검색 성공)
- **정확 매칭 검색률**: 33% (3건 중 1건)
- **다문서 통합 검색률**: 0% (3건 모두 실패)

---

## 🎯 유형별 문제점 분석

### 1️⃣ 무관 질의 (오늘 점심 뭐 먹을까? / 고래는 포유류야?)
**결과**: ❌ 한국어 오타 반복  
**문제**: "포유륜류..." 등 생성 품질 저하  
**원인**: CPU 속도 저하 + 한국어 토크나이저 품질

---

### 2️⃣ 저신뢰 히트 (플래핑이 뭐였지? / CG는 뭐야?)
**결과**: ⚠️ RAG 0건 → 일반 지식 기반 답변  
**문제점**:
- 약어/전문용어 검색 실패 (CG = Center of Gravity)
- 도메인 문맥 오판
- "플래핑"처럼 한글 전문용어 매칭 실패

---

### 3️⃣ 정확 매칭 (Coaxial Rotor / 유압 고장 / 가버너 고장)
**결과**: ⚠️ RAG 평균 1.3건 검색, **하지만 답변 품질 저하**

#### 문제 (1): Coaxial Rotor 방향 조종
- ❌ 답변: "회전수(RPM) 차이로 조종"
- ✅ 정답: "차등 컬렉티브(토크 불균형)"
- **원인**: RAG 검색은 성공했으나, 핵심 개념 추출/서술 왜곡

#### 문제 (2): 유압 계통 고장 조치
- ❌ 답변: 증상 나열 + "자동 활공" 권장
- ✅ 정답: 유압 스위치/CB 점검 + 활주착륙(rolling)
- **원인**: 근거 섹션 스니펫 선택 실패 + 안전 절차 우선순위 부재

#### 문제 (3): 가버너 고장 시 회전속도 변화
- ❌ 답변: '전동기' 일반론 환각
- ✅ 정답: RPM 상승 → 스로틀 감소 + 오토로테이션
- **원인**: RAG 0건인데 긴급/절차형 질문에 일반지식 폴백

---

### 4️⃣ 다문서 통합 (기화기 결빙 / 출력 제어 이상 / 진동 분석)
**결과**: 🔴 **전면 실패** (RAG 0건)

#### 핵심 원인
1. **용어 불일치**: 
   - 검색어: "기화기 결빙" → 문서: "Carburetor Icing"
   - 검색어: "유압 계통 고장" → 문서: "Hydraulic Failure"
   - 검색어: "엔진 출력 제어장치" → 문서: "Governor or Fuel Control Failure"

2. **LLM 환각**: 
   - RAG 0건 → 일반 기계론(냉각 시스템, 보조제어시스템) 서술
   - 실제 절차(RPM·Manifold 압력↓, 활주착륙) 누락

3. **절차형 질문 정책 부재**:
   - "근거 없음 고지" 미작동
   - 핵심 수치/우선순위 누락

4. **다문서 통합 불가능**:
   - 검색 단계 실패 → 통합 자체가 불가능
   - 비행교범(RFM) / 운용 매뉴얼 / 이론서 결합 X

---

## 🛠️ 개선 방향

### 🔴 긴급 (Critical) - 즉시 적용

#### 1. 속도 최적화
```python
# 목표: TTFT 39초 → 5초, 생성속도 0.45 → 2-5 tok/s
- 입력 토큰 제한: 957 → 512
- 생성 토큰 제한: 1024 → 256
- 리랭킹 스킵 옵션 추가 (8.7초 절약)
- GGUF 양자화 모델 전환 (3-5배 속도 향상)
```

#### 2. RAG 검색 개선
```python
# 다국어 용어 매칭
- 한영 동의어 사전 구축
  예: "기화기 결빙" = "Carburetor Icing" = "Carb Ice"
  
- 약어 확장 검색
  예: CG → Center of Gravity, Weight and Balance
  
- 임베딩 전 쿼리 전처리
  def preprocess_query(query):
      query = expand_abbreviations(query)  # CG → Center of Gravity
      query = add_english_terms(query)     # 유압 고장 + Hydraulic Failure
      return query
```

#### 3. 절차형 질문 정책
```python
# RAG 0건 + 절차형 질문 → 답변 거부
if doc_count == 0 and is_procedural_question(query):
    return "해당 내용은 교범에서 확인이 필요합니다. 검색된 문서가 없어 정확한 답변을 드릴 수 없습니다."
```

---

### 🟡 단기 (High Priority) - 1주 내

#### 4. 문서 인덱싱 재구축
```python
# 청크 전략 개선
- 현재: 단순 분할
- 개선: 섹션 기반 분할 (3.7.3, 3.7.4 등 절차 단위)
- 메타데이터 강화: 
  {
    "section": "3.7.4",
    "title_ko": "가버너 또는 연료 제어 고장",
    "title_en": "Governor or Fuel Control Failure",
    "keywords": ["RPM", "스로틀", "오토로테이션", "고정", "High", "Low"]
  }
```

#### 5. 핵심 개념 추출 로직
```python
# RAG 결과에서 핵심 절차 우선 추출
def extract_key_procedures(docs):
    priority_patterns = [
        r"점검|확인|Check",
        r"착륙|Landing",
        r"스위치|Switch|CB",
        r"RPM|회전수|속도"
    ]
    # 우선순위 정렬 후 반환
```

#### 6. 다문서 통합 전략
```python
# 관련 섹션 자동 확장
if query_type == "multi_document":
    docs = search_primary_docs(query)
    docs += search_related_sections(docs)  # 3.7.3 검색 시 3.7.4도 포함
    docs = merge_and_prioritize(docs)      # RFM > 운용매뉴얼 > 이론서
```

---

### 🟢 중기 (Medium Priority) - 1개월 내

#### 7. 한국어 품질 개선
- 더 나은 한국어 모델 평가 (Qwen 2.5, EXAONE 등)
- 후처리 맞춤법 검사기 추가

#### 8. 검색 정확도 모니터링
```python
# 로그에 검색 품질 지표 추가
logger.info(f"쿼리 유형: {query_type}")
logger.info(f"예상 필요 문서 수: {expected_docs}")
logger.info(f"실제 검색 문서 수: {actual_docs}")
logger.info(f"검색 성공률: {actual_docs/expected_docs*100:.1f}%")
```

---

## 📈 예상 개선 효과

| 지표 | 현재 | 목표 | 개선율 |
|------|------|------|--------|
| TTFT | 39초 | 5초 | 87% ↓ |
| 생성 속도 | 0.45 tok/s | 2-5 tok/s | 400-1000% ↑ |
| RAG 검색 성공률 | 25% | 70%+ | 280% ↑ |
| 정확 매칭 정확도 | 33% | 80%+ | 240% ↑ |
| 다문서 통합 성공률 | 0% | 60%+ | 신규 기능 |

---

# 방법1

맞습니다! 훨씬 더 나은 접근입니다. 수정된 개선 방향입니다:

---

### 🔴 긴급 (Critical) - 즉시 적용

#### 3. 절차형 질문 대화형 명확화 전략

```python
# rag.py - RAG 검색 실패 시 명확화 질문 생성
def generate_clarification_questions(query: str, doc_count: int):
    """
    검색 실패 시 사용자에게 맥락 파악을 위한 질문 생성
    """
    if doc_count == 0 and is_procedural_question(query):
        # 약어/애매한 용어 감지
        ambiguous_terms = detect_ambiguous_terms(query)
        
        clarifications = []
        
        # 약어 명확화
        if "CG" in query:
            clarifications.append({
                "question": "CG가 무엇을 의미하나요?",
                "options": [
                    "Center of Gravity (무게중심)",
                    "Center of Gravity Limits (무게중심 한계)",
                    "다른 의미"
                ]
            })
        
        # 절차 맥락 확인
        if any(word in query for word in ["고장", "이상", "문제", "조치"]):
            clarifications.append({
                "question": "어떤 상황에 대해 알고 싶으신가요?",
                "options": [
                    "비행 중 긴급 상황 조치",
                    "지상 점검 절차",
                    "정비 매뉴얼 내용",
                    "이론적 설명"
                ]
            })
        
        # 기체 종류 확인
        clarifications.append({
            "question": "어떤 헬리콥터 기종에 대한 질문인가요?",
            "options": [
                "Robinson R22/R44",
                "Bell 206",
                "일반적인 헬리콥터",
                "특정 기종 무관"
            ]
        })
        
        return clarifications
    
    return None


# llm.py - 명확화 대화 처리
def llm_streamer_with_clarification(
    prompt: str,
    use_rag: bool = False,
    rag_context: str = "",
    conversation_history: list = None  # 대화 이력 추가
):
    """
    RAG 검색 실패 시 명확화 질문으로 재검색 시도
    """
    
    # 1차 RAG 검색
    rag_results = rag_search_with_rerank(prompt)
    
    if len(rag_results) == 0:
        logger.info("RAG 검색 실패 - 명확화 질문 생성")
        
        # 명확화 질문 생성
        clarifications = generate_clarification_questions(prompt, len(rag_results))
        
        if clarifications:
            # 사용자에게 명확화 질문 전송
            yield f"data: {json.dumps({
                'type': 'clarification',
                'questions': clarifications,
                'message': '질문을 더 정확히 이해하기 위해 몇 가지 확인하고 싶습니다:'
            }, ensure_ascii=False)}\n\n"
            
            # 여기서 대기 (사용자 응답 대기)
            return  # 제너레이터 중단
    
    # RAG 성공 시 정상 처리
    # ... (기존 로직)


# 사용자 응답 후 재검색
def retry_search_with_context(
    original_query: str,
    clarification_answers: dict
):
    """
    명확화 응답을 바탕으로 쿼리 재구성 및 재검색
    
    Args:
        original_query: 원본 질문
        clarification_answers: {
            "CG_meaning": "Center of Gravity",
            "situation": "비행 중 긴급 상황 조치",
            "aircraft": "일반적인 헬리콥터"
        }
    """
    
    # 쿼리 재구성
    enhanced_query = reconstruct_query(original_query, clarification_answers)
    
    logger.info(f"재구성된 쿼리: {original_query} → {enhanced_query}")
    
    # 재검색
    rag_results = rag_search_with_rerank(enhanced_query)
    
    if len(rag_results) > 0:
        logger.info(f"재검색 성공: {len(rag_results)}개 문서")
        return rag_results
    else:
        logger.warning("재검색도 실패")
        return None


def reconstruct_query(original_query: str, answers: dict) -> str:
    """
    명확화 응답으로 쿼리 재구성
    """
    enhanced = original_query
    
    # 약어 확장
    if "CG_meaning" in answers:
        if answers["CG_meaning"] == "Center of Gravity":
            enhanced = enhanced.replace("CG", "Center of Gravity 무게중심")
    
    # 맥락 추가
    if "situation" in answers:
        situation_context = {
            "비행 중 긴급 상황 조치": "Emergency Procedure 긴급절차",
            "지상 점검 절차": "Pre-flight Check 지상점검",
            "정비 매뉴얼 내용": "Maintenance 정비",
            "이론적 설명": "Theory 이론"
        }
        context = situation_context.get(answers["situation"], "")
        enhanced = f"{context} {enhanced}"
    
    # 기종 정보 추가
    if "aircraft" in answers and answers["aircraft"] != "특정 기종 무관":
        enhanced = f"{answers['aircraft']} {enhanced}"
    
    return enhanced
```

---

### 📱 프론트엔드 (chat.html) 수정

```javascript
// SSE 이벤트 리스너에 명확화 처리 추가
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'clarification') {
        // 명확화 질문 UI 표시
        displayClarificationQuestions(data.questions, data.message);
        
    } else if (data.type === 'text') {
        // 기존 텍스트 처리
        appendToMessage(data.content);
        
    } // ... 기타 타입
};

function displayClarificationQuestions(questions, message) {
    const clarificationDiv = document.createElement('div');
    clarificationDiv.className = 'clarification-box';
    clarificationDiv.innerHTML = `
        <p><strong>${message}</strong></p>
        ${questions.map((q, idx) => `
            <div class="clarification-question">
                <p>${q.question}</p>
                ${q.options.map((opt, optIdx) => `
                    <label>
                        <input type="radio" 
                               name="clarification_${idx}" 
                               value="${opt}">
                        ${opt}
                    </label>
                `).join('')}
            </div>
        `).join('')}
        <button onclick="submitClarification()">답변 제출</button>
    `;
    
    document.getElementById('messages').appendChild(clarificationDiv);
}

function submitClarification() {
    const answers = {};
    
    // 사용자 선택 수집
    document.querySelectorAll('.clarification-question').forEach((q, idx) => {
        const selected = q.querySelector('input[type="radio"]:checked');
        if (selected) {
            answers[`answer_${idx}`] = selected.value;
        }
    });
    
    // 서버에 재검색 요청
    fetch('/api/retry_search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            original_query: lastQuery,
            clarification_answers: answers
        })
    }).then(response => {
        // 재검색 결과로 LLM 재실행
        // ... 
    });
}
```

---

### 🎯 실제 동작 예시

#### Case 1: "CG는 뭐야?"

**1차 검색 실패 → 명확화 질문**
```
🤖 질문을 더 정확히 이해하기 위해 몇 가지 확인하고 싶습니다:

1. CG가 무엇을 의미하나요?
   ○ Center of Gravity (무게중심)
   ○ Center of Gravity Limits (무게중심 한계)
   ○ 다른 의미

2. 어떤 내용을 알고 싶으신가요?
   ○ CG 계산 방법
   ○ CG 한계 범위
   ○ CG 이동 시 영향
   ○ CG 관련 비행 제한사항
```

**사용자 선택 → 쿼리 재구성**
```
원본: "CG는 뭐야?"
재구성: "Center of Gravity 무게중심 계산 방법"
```

**재검색 성공 → 정확한 답변**

---

#### Case 2: "가버너 고장 시 조치"

**1차 검색 실패 → 명확화**
```
🤖 더 정확한 답변을 위해 확인하고 싶습니다:

1. 어떤 상황인가요?
   ○ 비행 중 고장 발생
   ○ 지상 점검 중 발견
   ○ 이론적 설명 필요

2. 가버너가 어느 위치에서 고장났나요?
   ○ High (높은 쪽)
   ○ Low (낮은 쪽)
   ○ 모르겠음
```

**사용자 선택 → 재구성**
```
원본: "가버너 고장 시 조치"
재구성: "Emergency Procedure Governor Failure High 비행 중 조치"
```

**재검색 → "Governor or Fuel Control Failure" 섹션 히트 → 정확한 답변**

---

### 📊 예상 개선 효과

| 지표 | 현재 | 명확화 전략 적용 후 | 개선율 |
|------|------|-------------------|--------|
| RAG 검색 성공률 | 25% | **70-80%** | 280-320% ↑ |
| 정확 매칭 정확도 | 33% | **85%+** | 260% ↑ |
| 다문서 통합 성공률 | 0% | **65%+** | 신규 활성화 |
| 사용자 만족도 | 낮음 | **대폭 향상** (능동적 소통) |

---

이 방식이 훨씬 더 실용적이고 사용자 경험도 좋습니다! 구현 코드를 작성해드릴까요?

---

# 방법2

좋은 질문입니다! 두 가지 모두 중요한 포인트네요.

---

## 1️⃣ 키워드 필터링 + 유사도 검색 전략

### ❌ 오해: 속도 개선이 목적이 아닙니다

제가 제안한 키워드 전략은 **속도 개선이 아니라 정확도 개선**이 목적입니다.

### ✅ 실제 의도: 2단계 하이브리드 검색

```python
# 현재 방식 (문제)
def current_search(query):
    # 1단계: 벡터 유사도 검색만 사용
    results = vector_search(query, k=20)  # 811ms
    # → "CG"를 검색하면 "Center of Gravity"를 못 찾음
    # → "기화기 결빙"을 검색하면 "Carburetor Icing"을 못 찾음
    return results

# 제안 방식 (개선)
def hybrid_search(query):
    # 1단계: 키워드 필터링 (BM25 또는 메타데이터 필터)
    keyword_candidates = keyword_filter(query)  # +50-100ms
    
    # 2단계: 필터링된 후보군에서 벡터 검색
    if keyword_candidates:
        results = vector_search(query, filter=keyword_candidates, k=20)  # 811ms
    else:
        results = vector_search(query, k=20)  # 폴백
    
    return results
```

---

### 📊 속도 영향 분석

| 단계 | 현재 방식 | 하이브리드 방식 | 차이 |
|------|----------|----------------|------|
| 키워드 필터 | - | +50-100ms | 추가 |
| 벡터 검색 | 811ms | 811ms | 동일 |
| 리랭킹 | 8,067ms | 8,067ms | 동일 |
| **총합** | **8,878ms** | **8,978ms** | **+100ms (1%)** |

**결론: 속도는 거의 동일하지만, 정확도가 크게 향상됩니다.**

---

### 🎯 정확도 개선 효과

#### Before (현재)
```python
# 쿼리: "CG는 뭐야?"
vector_search("CG는 뭐야?")
# → 결과: 0건 (임베딩이 "CG"라는 짧은 약어를 제대로 인코딩 못함)
```

#### After (하이브리드)
```python
# 쿼리: "CG는 뭐야?"
# 1단계: 키워드 확장
expanded_query = expand_abbreviations("CG")
# → ["CG", "Center of Gravity", "무게중심", "Weight and Balance"]

# 2단계: 메타데이터 필터
keyword_filter(query="CG", keywords=["CG", "Center of Gravity"])
# → 후보 문서: 3.2절, 4.1절 등 15개 문서로 좁힘

# 3단계: 좁혀진 범위에서 벡터 검색
vector_search("CG는 뭐야?", candidates=15개)
# → 결과: 3-5건 (정확도 향상)
```

---

### 💡 실제 구현 방식

```python
# 방법 1: 메타데이터 활용 (추천 ✅)
# 문서 인덱싱 시 메타데이터에 키워드 추가
def create_documents_with_keywords(chunks):
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk['text'],
            metadata={
                'page': chunk['page'],
                'section': chunk['section'],
                'keywords': extract_keywords(chunk['text']),
                'keywords_en': ['CG', 'Center of Gravity', 'Weight'],
                'keywords_ko': ['무게중심', '중심', '밸런스']
            }
        )
        documents.append(doc)
    return documents

# 검색 시 메타데이터 필터 사용
def rag_search_with_keyword_filter(query: str, top_k: int = 20):
    # 쿼리에서 키워드 추출
    query_keywords = extract_keywords_from_query(query)
    
    # Chroma 메타데이터 필터
    if query_keywords:
        search_results = collection.similarity_search(
            query, 
            k=top_k,
            filter={
                "$or": [
                    {"keywords_en": {"$in": query_keywords}},
                    {"keywords_ko": {"$in": query_keywords}}
                ]
            }
        )
    else:
        search_results = collection.similarity_search(query, k=top_k)
    
    return search_results
```

```python
# 방법 2: BM25 + 벡터 검색 결합 (더 고급)
from rank_bm25 import BM25Okapi

def hybrid_search_bm25(query: str, alpha=0.5):
    # BM25 검색 (키워드 기반)
    bm25_results = bm25_search(query, top_k=50)  # +50ms
    
    # 벡터 검색
    vector_results = vector_search(query, top_k=50)  # 811ms
    
    # 점수 결합 (alpha: BM25 가중치, 1-alpha: 벡터 가중치)
    combined = combine_scores(bm25_results, vector_results, alpha)
    
    return combined[:20]  # 상위 20개만
```

---

## 2️⃣ 한국어 모델 변경 시 문제점

### ❌ 맞습니다. 모델 변경은 리스크가 있습니다.

```python
# 현재 사용 중인 모델 (추정)
model_name = "meta-llama/Llama-3.2-8B-Instruct"  # 또는 유사 모델

# 다른 모델로 변경 시 문제점
```

### 🔴 발생 가능한 문제들

#### 1. **출력 형식 변경**
```python
# Llama 모델
"헬리콥터의 Center of Gravity는..."  # 자연스러운 문장

# 다른 모델 (예: Qwen)
"## CG 설명\n1. 정의: ...\n2. 계산: ..."  # 마크다운 형식 선호

# 또 다른 모델
"<answer>CG는 무게중심입니다.</answer>"  # XML 태그 사용
```

#### 2. **토크나이저 차이**
```python
# Llama 토크나이저
tokenizer.apply_chat_template(messages)
# → 정상 작동

# 다른 모델 토크나이저
tokenizer.apply_chat_template(messages)
# → AttributeError: 'GPT2Tokenizer' object has no attribute 'apply_chat_template'
```

#### 3. **특수 토큰 차이**
```python
# Llama
eos_token = "<|eot_id|>"

# Qwen
eos_token = "<|im_end|>"

# EXAONE
eos_token = "</s>"
```

#### 4. **프롬프트 템플릿 차이**
```python
# Llama 형식
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"

# Qwen 형식
"<|im_start|>system\n{system}<|im_end|>"

# ChatGPT 형식
"### System:\n{system}\n\n### User:\n{user}"
```

---

### ✅ 모델 변경을 안전하게 하려면

```python
# models.py - 모델별 설정 추상화
MODEL_CONFIGS = {
    'llama': {
        'eos_tokens': ['<|eot_id|>', '<|end_of_text|>'],
        'chat_template': True,
        'output_format': 'natural',
        'best_for': ['general', 'instruction']
    },
    'qwen': {
        'eos_tokens': ['<|im_end|>', '<|endoftext|>'],
        'chat_template': True,
        'output_format': 'markdown',  # 마크다운 선호
        'best_for': ['korean', 'multilingual']
    },
    'exaone': {
        'eos_tokens': ['</s>'],
        'chat_template': False,  # 수동 템플릿 필요
        'output_format': 'natural',
        'best_for': ['korean']
    }
}

def load_model_safe(model_name: str):
    """모델별 설정을 자동으로 적용"""
    model_type = detect_model_type(model_name)
    config = MODEL_CONFIGS.get(model_type)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 설정 적용
    if not config['chat_template'] and not hasattr(tokenizer, 'apply_chat_template'):
        tokenizer.apply_chat_template = create_custom_template(model_type)
    
    return model, tokenizer, config


# llm.py - 모델별 처리
def llm_streamer_adaptive(prompt: str, model_config: dict):
    """모델 설정에 따라 적응적으로 처리"""
    
    # 토크나이저 처리
    if model_config['chat_template']:
        input_ids = tokenizer.apply_chat_template(messages, ...)
    else:
        input_ids = manual_chat_template(messages, model_config)
    
    # EOS 토큰 처리
    eos_tokens = model_config['eos_tokens']
    eos_token_id = [tokenizer.convert_tokens_to_ids(t) for t in eos_tokens]
    
    # 출력 후처리
    if model_config['output_format'] == 'markdown':
        output = clean_markdown_artifacts(generated_text)
    
    return output
```

---

### 🎯 추천 방향

#### 현재 모델 유지 + 최적화 우선
```python
# 1순위: 속도 최적화 (모델 변경 없이)
- GGUF 양자화 (llama.cpp)
- 입력/출력 토큰 제한
- 배치 처리 개선

# 2순위: RAG 정확도 개선 (모델 변경 없이)
- 키워드 필터링 추가
- 쿼리 재구성 로직
- 명확화 대화

# 3순위: 모델 변경 검토 (GPU 도입 후)
- 여러 모델 A/B 테스트
- 한국어 품질 vs 속도 트레이드오프 평가
```

---

### 📊 결론

| 항목 | 키워드 필터링 | 모델 변경 |
|------|-------------|----------|
| **속도 영향** | +1% (거의 없음) | 알 수 없음 (모델마다 다름) |
| **정확도 개선** | ✅ 크게 향상 | ⚠️ 불확실 |
| **리스크** | ✅ 낮음 | 🔴 높음 (출력/토큰/템플릿 이슈) |
| **구현 난이도** | ✅ 쉬움 | 🔴 어려움 (테스트 필요) |
| **추천 우선순위** | 🔴 1순위 (즉시 적용) | 🟢 3순위 (GPU 후 검토) |

**답변: 키워드 필터링은 속도는 거의 그대로지만 정확도가 크게 향상되고, 모델 변경은 리스크가 커서 현재 단계에서는 권장하지 않습니다.**

---

# 방법3

# 문제별 해결방안 정리

---

## 🔴 문제 1: 응답 속도 심각 (TTFT 58초, 생성 0.45 tok/s)

### 원인
- CPU 전용 환경에서 대형 모델 실행
- 입력 토큰 957개 (과다)
- 생성 토큰 1024개 설정 (과다)
- 리랭킹 8.7초 소요

### 해결방안

#### 🔴 즉시 적용 (1일 내)
```python
# 1-1. 입력/출력 토큰 제한
# config.py
MAX_INPUT_TOKENS = 512  # 957 → 512 (TTFT 50% 감소 예상)
MAX_NEW_TOKENS = 256    # 1024 → 256 (전체 시간 75% 감소)
MAX_CONTEXT_LENGTH = 1000  # 2215 → 1000

# 예상 효과: TTFT 58초 → 25-30초, 전체 7분 → 2-3분
```

```python
# 1-2. 빠른 모드 옵션
# config.py
FAST_MODE = True
ENABLE_RERANKING = True  # 리랭킹은 유지 (정확도 위해)
VECTOR_SEARCH_TOP_K = 10  # 20 → 10

if FAST_MODE:
    MAX_INPUT_TOKENS = 512
    MAX_NEW_TOKENS = 256
    VECTOR_SEARCH_TOP_K = 10
```

```python
# 1-3. 생성 파라미터 최적화
# llm.py
generation_kwargs = {
    # ... 기존 파라미터
    "num_beams": 1,      # beam search 비활성화
    "use_cache": True,   # KV 캐시 활성화
    "early_stopping": True,  # 조기 종료
}
```

#### 🟡 단기 적용 (1주 내)
```python
# 1-4. GGUF 양자화 모델 전환
# models.py
from llama_cpp import Llama

model = Llama(
    model_path="llama-3.2-8b-instruct-q4_k_m.gguf",  # 4bit 양자화
    n_ctx=2048,
    n_threads=8,        # CPU 코어 수 맞춤
    n_batch=512,
    n_gpu_layers=0      # CPU 전용
)

# 예상 효과: 속도 3-5배 향상 → 0.45 tok/s → 1.5-2.5 tok/s
```

#### 🟢 중기 적용 (GPU 도입 후)
```python
# 1-5. GPU 활용
model = Llama(
    model_path="model.gguf",
    n_gpu_layers=32,    # GPU 레이어 수
    # ...
)

# 예상 효과: 20-100 tok/s
```

---

## 🔴 문제 2: RAG 검색 실패율 75% (8건 중 6건 실패)

### 원인
- 한영 용어 불일치: "기화기 결빙" ≠ "Carburetor Icing"
- 약어 미인식: "CG" 검색 불가
- 섹션 분산: 관련 내용이 여러 절에 분산

### 해결방안

#### 🔴 즉시 적용 (1-2일)
```python
# 2-1. 쿼리 전처리 - 약어 확장
# rag.py
ABBREVIATION_MAP = {
    'CG': ['CG', 'Center of Gravity', '무게중심', 'Weight and Balance'],
    'RPM': ['RPM', 'Revolutions Per Minute', '회전수', '회전속도'],
    'HYD': ['HYD', 'Hydraulic', '유압'],
    'CB': ['CB', 'Circuit Breaker', '차단기'],
    # ... 더 추가
}

def expand_query(query: str) -> str:
    """쿼리 확장"""
    expanded_terms = []
    
    for abbr, expansions in ABBREVIATION_MAP.items():
        if abbr in query.upper():
            expanded_terms.extend(expansions)
    
    return f"{query} {' '.join(expanded_terms)}"

# 사용
def rag_search_with_rerank(query: str, top_k: int = None, rerank_top_k: int = None):
    # 쿼리 확장
    enhanced_query = expand_query(query)
    logger.info(f"쿼리 확장: {query} → {enhanced_query}")
    
    # 검색
    search_results = models.collection.similarity_search(enhanced_query, k=top_k * 2)
    # ...

# 예상 효과: 약어 관련 검색 성공률 0% → 70%+
```

```python
# 2-2. 한영 동시 검색
# rag.py
TERM_TRANSLATION = {
    '기화기 결빙': 'Carburetor Icing',
    '유압 고장': 'Hydraulic Failure',
    '가버너': 'Governor',
    '연료 제어': 'Fuel Control',
    # ...
}

def add_english_terms(query: str) -> str:
    """한글 쿼리에 영문 추가"""
    for ko, en in TERM_TRANSLATION.items():
        if ko in query:
            query = f"{query} {en}"
    return query

# 예상 효과: 한글 전문용어 검색 0% → 60%+
```

#### 🟡 단기 적용 (1주 내)
```python
# 2-3. 메타데이터 키워드 필터링
# indexing.py - 문서 인덱싱 시
def create_documents_with_metadata(chunks):
    documents = []
    for chunk in chunks:
        # 키워드 자동 추출
        keywords_ko = extract_korean_keywords(chunk['text'])
        keywords_en = extract_english_keywords(chunk['text'])
        
        doc = Document(
            page_content=chunk['text'],
            metadata={
                'page': chunk['page'],
                'section': chunk['section'],
                'title_ko': chunk.get('title_ko', ''),
                'title_en': chunk.get('title_en', ''),
                'keywords_ko': keywords_ko,  # ['무게중심', '계산', '한계']
                'keywords_en': keywords_en,  # ['CG', 'Center of Gravity', 'Limits']
                'procedure_type': detect_procedure_type(chunk['text'])  # 'emergency', 'normal', 'theory'
            }
        )
        documents.append(doc)
    return documents

# rag.py - 검색 시 메타데이터 활용
def rag_search_with_metadata_filter(query: str, top_k: int = 20):
    # 쿼리에서 키워드 추출
    query_keywords = extract_keywords(query)
    
    # 메타데이터 필터 구성
    if query_keywords:
        search_results = models.collection.similarity_search(
            query,
            k=top_k,
            filter={
                "$or": [
                    {"keywords_ko": {"$in": query_keywords['ko']}},
                    {"keywords_en": {"$in": query_keywords['en']}}
                ]
            }
        )
    else:
        search_results = models.collection.similarity_search(query, k=top_k)
    
    return search_results

# 예상 효과: 전체 검색 성공률 25% → 60-70%
```

```python
# 2-4. 관련 섹션 자동 확장
# rag.py
RELATED_SECTIONS = {
    '3.7.3': ['3.7.4', '3.7.5'],  # 유압 고장 → 가버너 고장, 엔진 고장
    '3.7.4': ['3.7.3', '4.2.1'],  # 가버너 고장 → 유압, 엔진 이론
    # ...
}

def expand_related_sections(initial_results):
    """검색된 섹션의 관련 섹션도 가져오기"""
    sections = [doc.metadata.get('section') for doc in initial_results]
    related_sections = set()
    
    for section in sections:
        if section in RELATED_SECTIONS:
            related_sections.update(RELATED_SECTIONS[section])
    
    # 관련 섹션 추가 검색
    additional_docs = []
    for section in related_sections:
        docs = models.collection.get(where={"section": section})
        additional_docs.extend(docs)
    
    return initial_results + additional_docs

# 예상 효과: 다문서 통합 성공률 0% → 50%+
```

---

## 🔴 문제 3: 정확 매칭에서도 오답 (검색 성공했으나 답변 왜곡)

### 원인
- 핵심 개념 추출 실패: "차등 컬렉티브" → "RPM 차이"로 왜곡
- 절차 우선순위 무시: "활주착륙" 빠짐
- 안전 절차 폴리시 부재

### 해결방안

#### 🔴 즉시 적용 (1-2일)
```python
# 3-1. 프롬프트 강화 - 절차형 질문 처리
# config.py
RAG_INSTRUCTION_TEMPLATE = """
다음은 헬리콥터 비행 교범에서 검색된 자료입니다:

{rag_context}

위 자료를 바탕으로 다음 질문에 답변하세요:
{prompt}

**중요 지침:**
1. 안전 절차가 포함된 경우, 반드시 순서대로 명시하세요
2. 핵심 용어를 정확하게 사용하세요 (예: "차등 컬렉티브", "활주착륙")
3. 수치나 범위가 있다면 반드시 포함하세요
4. 자료에 없는 내용은 추측하지 마세요
5. 긴급 절차의 경우 우선순위를 명확히 하세요
"""

# 예상 효과: 절차 정확도 33% → 60%+
```

```python
# 3-2. 핵심 개념 추출 검증
# llm.py - 생성 후 검증
CRITICAL_TERMS = {
    'coaxial rotor': ['차등 컬렉티브', 'differential collective', '토크 불균형'],
    'hydraulic failure': ['유압 스위치', 'CB', '활주착륙', 'rolling landing'],
    'governor failure': ['스로틀', 'throttle', '오토로테이션', 'autorotation'],
    # ...
}

def verify_answer_quality(query: str, answer: str, rag_context: str):
    """답변 품질 검증"""
    # 쿼리 타입 감지
    query_type = detect_query_type(query)
    
    if query_type in CRITICAL_TERMS:
        required_terms = CRITICAL_TERMS[query_type]
        
        # 필수 용어가 답변에 있는지 확인
        missing_terms = []
        for term in required_terms:
            if term.lower() not in answer.lower():
                # RAG 컨텍스트에는 있는지 확인
                if term.lower() in rag_context.lower():
                    missing_terms.append(term)
        
        if missing_terms:
            logger.warning(f"핵심 용어 누락: {missing_terms}")
            # 경고 메시지 추가
            return answer + f"\n\n⚠️ 참고: 교범에는 '{', '.join(missing_terms)}'도 언급되어 있습니다."
    
    return answer

# 예상 효과: 핵심 용어 누락 방지
```

#### 🟡 단기 적용 (1주 내)
```python
# 3-3. 절차 우선순위 파싱
# utils.py
def extract_procedure_steps(text: str):
    """절차 단계 추출"""
    steps = []
    
    # 패턴 매칭
    patterns = [
        r'(\d+)\.\s*([^\n]+)',  # 1. 첫 번째 단계
        r'([가-힣]+)\s*→\s*([가-힣]+)',  # 스위치 확인 → 착륙
        r'First.*?then.*?',  # First check, then land
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            steps.extend(matches)
    
    return steps

# llm.py - 절차형 답변 강제
def format_procedure_answer(steps: list):
    """절차를 명확한 형식으로 출력"""
    formatted = "**절차:**\n"
    for i, step in enumerate(steps, 1):
        formatted += f"{i}. {step}\n"
    return formatted

# 예상 효과: 절차 순서 명확화
```

---

## 🔴 문제 4: 다문서 통합 완전 실패 (검색 0건)

### 원인
- 복잡한 질문을 단일 쿼리로 검색 시도
- 여러 섹션에 분산된 내용을 통합 못함

### 해결방안

#### 🔴 즉시 적용 (1-2일)
```python
# 4-1. 쿼리 분해 + 다중 검색
# rag.py
def decompose_complex_query(query: str):
    """복잡한 질문을 하위 질문으로 분해"""
    
    # 패턴 감지
    if '조건' in query and '징후' in query and '예방' in query:
        # "조건·징후·예방" 패턴
        return [
            query.replace('조건·징후·예방', '조건'),
            query.replace('조건·징후·예방', '징후'),
            query.replace('조건·징후·예방', '예방')
        ]
    
    if '회전속도' in query and '유지 방법' in query and '판단 기준' in query:
        # "방법·기준" 패턴
        return [
            query.split('과')[0],  # 회전속도 유지 방법
            query.split('과')[1]   # 판단 기준
        ]
    
    return [query]  # 분해 불가능하면 원본 반환


def multi_query_search(query: str, top_k: int = 20):
    """다중 쿼리 검색"""
    sub_queries = decompose_complex_query(query)
    
    all_results = []
    for sub_q in sub_queries:
        logger.info(f"하위 검색: {sub_q}")
        results = rag_search_with_rerank(sub_q, top_k=top_k // len(sub_queries))
        all_results.extend(results)
    
    # 중복 제거 + 재정렬
    unique_results = deduplicate_results(all_results)
    return unique_results

# 예상 효과: 다문서 통합 0% → 40-50%
```

#### 🟡 단기 적용 (1주 내)
```python
# 4-2. 명확화 대화 시스템
# llm.py
def detect_complex_query(query: str):
    """복잡한 질문 감지"""
    complexity_indicators = [
        len(query.split('와')) > 2,  # "A와 B와 C"
        '·' in query,  # "조건·징후·예방"
        query.count('?') > 1,  # 여러 질문
        len(query) > 100,  # 긴 질문
    ]
    return sum(complexity_indicators) >= 2


def generate_clarification_for_complex_query(query: str):
    """복잡한 질문에 대한 명확화"""
    return {
        'type': 'clarification',
        'message': '질문이 여러 부분으로 나뉘어 있습니다. 어느 부분을 우선 알고 싶으신가요?',
        'options': decompose_complex_query(query)
    }

# 사용
def llm_streamer_with_rag_and_tts(...):
    # ...
    if detect_complex_query(prompt) and len(rag_results) == 0:
        clarification = generate_clarification_for_complex_query(prompt)
        yield f"data: {json.dumps(clarification, ensure_ascii=False)}\n\n"
        return

# 예상 효과: 사용자가 명확화 → 검색 성공률 상승
```

```python
# 4-3. 섹션 간 관계 그래프
# indexing.py
SECTION_GRAPH = {
    'carburetor_icing': {
        'related_sections': ['3.5.2', '4.3.1'],
        'related_topics': ['engine_management', 'cold_weather'],
        'keywords': ['기화기', 'carburetor', '결빙', 'icing', '습도']
    },
    'hydraulic_failure': {
        'related_sections': ['3.7.3', '3.7.4'],
        'related_topics': ['emergency_procedures', 'landing'],
        'keywords': ['유압', 'hydraulic', '스위치', 'CB', '활주']
    },
    # ...
}

def search_with_graph_expansion(query: str):
    """그래프 기반 검색 확장"""
    # 1차 검색
    initial_results = rag_search_with_rerank(query)
    
    # 관련 토픽 감지
    detected_topics = []
    for topic, info in SECTION_GRAPH.items():
        if any(kw in query for kw in info['keywords']):
            detected_topics.append(topic)
    
    # 관련 섹션 추가 검색
    for topic in detected_topics:
        related_sections = SECTION_GRAPH[topic]['related_sections']
        for section in related_sections:
            additional = models.collection.get(where={"section": section})
            initial_results.extend(additional)
    
    return initial_results

# 예상 효과: 다문서 통합 0% → 60%+
```

---

## 🔴 문제 5: 한국어 품질 저하 ("포유륜류..." 오타)

### 원인
- CPU 저성능 + 느린 생성 속도
- 모델의 한국어 토크나이저 품질

### 해결방안

#### 🔴 즉시 적용 (1일 내)
```python
# 5-1. 생성 파라미터 조정
# config.py
TEMPERATURE = 0.4  # 0.6 → 0.4 (더 보수적 생성)
REPETITION_PENALTY = 1.3  # 1.1 → 1.3 (반복 강하게 억제)

# 예상 효과: 오타 발생률 감소
```

```python
# 5-2. 후처리 맞춤법 검사
# utils.py
from hanspell import spell_checker

def postprocess_korean_text(text: str):
    """한국어 텍스트 후처리"""
    # 명백한 오타 수정
    corrections = {
        '포유륜류': '포유류',
        '헬리콥타': '헬리콥터',
        # 자주 발생하는 오타 추가
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # 한글 맞춤법 검사 (선택적)
    # spelled = spell_checker.check(text)
    # text = spelled.checked
    
    return text

# llm.py - 스트리밍 중 적용
for new_text in streamer:
    if new_text:
        # 문장 완성 시 후처리
        if utils.is_complete_sentence(sentence_buffer):
            sentence_buffer = postprocess_korean_text(sentence_buffer)
        # ...

# 예상 효과: 명백한 오타 즉시 수정
```

#### 🟢 중기 적용 (GPU 도입 후)
```python
# 5-3. 한국어 특화 모델 평가 (GPU 후)
KOREAN_MODELS = [
    'Qwen/Qwen2.5-7B-Instruct',  # 다국어 강점
    'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct',  # 한국어 특화
    # ...
]

# A/B 테스트로 품질 비교
# 현재는 리스크가 커서 GPU 도입 후 검토
```

---

## 📊 문제별 우선순위 및 예상 효과

| 문제 | 우선순위 | 즉시 적용 효과 | 단기 적용 효과 | 총 개선율 |
|------|---------|--------------|--------------|----------|
| **1. 속도** | 🔴🔴🔴 | TTFT 50%↓, 전체 75%↓ | 속도 3-5배↑ | **85-90%↓** |
| **2. 검색 실패** | 🔴🔴🔴 | 약어 70%↑, 한영 60%↑ | 전체 60-70%↑ | **280%↑** |
| **3. 답변 왜곡** | 🔴🔴 | 절차 60%↑ | 핵심 용어 보존 | **180%↑** |
| **4. 다문서 통합** | 🔴🔴 | 40-50%↑ | 60%↑ | **신규 기능** |
| **5. 한국어 품질** | 🟡 | 오타 50%↓ | - | **50%↓** |

---

## 🎯 적용 로드맵

### Day 1 (오늘)
- ✅ 토큰 제한 (문제 1)
- ✅ 약어 확장 (문제 2)
- ✅ 프롬프트 강화 (문제 3)
- ✅ 쿼리 분해 (문제 4)
- ✅ 오타 후처리 (문제 5)

### Week 1
- ✅ GGUF 모델 전환 (문제 1)
- ✅ 메타데이터 키워드 (문제 2)
- ✅ 절차 파싱 (문제 3)
- ✅ 명확화 대화 (문제 4)

### Month 1 (GPU 도입 후)
- ✅ GPU 활용 (문제 1)
- ✅ 그래프 검색 (문제 4)
- ✅ 모델 평가 (문제 5)

**가장 먼저 적용할 코드를 작성해드릴까요?**

---

# 방법4