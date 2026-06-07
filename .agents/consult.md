# 상담 기록

- 사용자 질문과 의도를 요약하고, AI 응답 요약을 함께 기록하세요.
- 중요한 결정이나 합의는 날짜와 함께 남깁니다.

---

## 2025-10-28: 프로젝트 전면 리팩토링

### 사용자 요청
프로젝트를 리팩토링하고자 함. 특히 config 설정을 Python 클래스에서 JSON 파일로 분리하고, 전체 모듈 구조를 개선하고 싶음.

### AI 제안 및 합의
1. **Config 분리**:
   - 기존의 하드코딩된 config.py를 6개의 JSON 파일로 분리
   - models.json, rag.json, server.json, audio.json, prompts.json, logging.json
   - 환경변수 우선순위 유지 (USE_EMBEDDING_GPU, USE_RERANK_GPU 등)

2. **모듈 구조 개편**:
   - `app/core/`: 핵심 비즈니스 로직 (models.py, llm.py)
   - `app/services/`: 서비스 레이어 (rag_service.py, audio_service.py, stt_service.py)
   - `app/api/`: API 라우팅 (main.py, routes/)
   - `app/utils/`: 유틸리티 함수 (text_utils.py)

3. **라우트 분리**:
   - 600줄 이상의 main.py를 기능별로 분리
   - pages.py: HTML 페이지 라우트
   - chat.py: 텍스트 채팅 API
   - voice.py: 음성 채팅 API

4. **Scope 확인**:
   - 파일/폴더 구조 재편성만 수행
   - 소스코드 로직 자체는 수정하지 않음
   - Import 경로만 수정

### 추가 작업
- 중복된 ssl_certificates 폴더 확인 및 삭제 (app/ssl_certificates → 프로젝트 루트만 사용)
- 사용하지 않는 파일 정리:
  - 백업 파일 (_old_files_backup/, main.py.bak)
  - 캐시 파일 (__pycache__, *.pyc, *.log)
  - IDE/시스템 파일 (.DS_Store, .idea/, .ipynb_checkpoints/, .claude/)
  - 사용하지 않는 모델 (my_whisper_base)
- Whisper 모델 경로 수정 (app/my_whisper_small → app/services/my_whisper_small)
- .gitignore 파일 생성

### 결과
- ✅ 리팩토링 완료
- ✅ Import 테스트 성공
- ✅ 깔끔한 프로젝트 구조 확립
- ✅ JSON 기반 설정 관리 도입

### 추가 규칙 설정
- `.claude` 폴더는 "지금까지 내역 정리" 명령 시 자동으로 삭제
- `agents_meta.md`의 "Session Summary Command" 섹션에 임시 파일 정리 규칙 추가

---

## 2025-10-29: Jupyter 노트북 Import 경로 업데이트

### 사용자 요청
리팩토링된 프로젝트 구조에 맞춰 Jupyter 노트북들의 import 경로를 수정해야 함. 처음에 llm_tester-Copy1.ipynb를 중복 파일로 판단하여 삭제했으나, 사용자가 필요한 파일임을 알림.

### AI 작업 내용
1. **파일 복원**:
   - 백업 폴더(`AITutor_dy_v2_bak251028_first_gitlab`)에서 `llm_tester-Copy1.ipynb` 복사

2. **Import 경로 수정 (3개 노트북)**:
   ```python
   # 변경 전
   import config
   import models
   import llm
   import utils
   import rag
   from app.models import CustomEmbeddings

   # 변경 후
   from app.config import config
   from app.core import models
   from app.core import llm
   from app.utils import text_utils as utils
   from app.services import rag_service as rag
   from app.core.models import CustomEmbeddings
   ```

3. **Config 참조 패턴 수정**:
   - `config.config.LLM_MODEL` → `config.LLM_MODEL`
   - `config.config.USE_EMBEDDING_GPU` → `config.USE_EMBEDDING_GPU`
   - 기타 모든 `config.config.XXX` 패턴 수정

### 결과
- ✅ 3개 노트북 모두 리팩토링된 구조와 호환
- ✅ Import 테스트 성공
- ✅ 개발/테스트 환경 정상화

---

## 2025-10-29: Level 1, 2 리팩토링 (코드 품질 개선)

### 사용자 요청
Level 1, 2 리팩토링을 진행하되, 주석은 유지. 완료 후 메타 업데이트 및 Level 3 플랜 작성 요청.

### AI 작업 내용

#### Level 1: 빠른 정리
1. **매직 넘버 상수화**
   - 문제: llm.py와 rag_service.py에 하드코딩된 숫자(1500, 800, 200)
   - 해결: config/rag.json에 상수 추가
     ```json
     "rag_context_max_length": 1500,
     "rag_context_sentence_boundary_min": 800,
     "context_min_remaining_length": 200
     ```
   - app/config.py에서 로딩, llm.py와 rag_service.py에서 사용

2. **RAG 검색 로직 중복 제거**
   - 문제: chat.py와 voice.py에 동일한 RAG 처리 로직 반복 (40+ 줄)
   - 해결: rag_service.py에 헬퍼 함수 추가
     - `format_rag_results()`: RAG 결과를 포맷팅하여 문서 요약 및 컨텍스트 생성
     - `generate_rag_sse_events()`: RAG 결과를 SSE 이벤트로 변환
   - chat.py, voice.py에서 5줄로 축약

#### Level 2: 모듈 책임 분리
**문제**: app/core/models.py (254줄)가 너무 많은 책임을 가짐
- LLM 로딩
- 임베딩 로딩
- 리랭크 로딩
- 벡터DB 관리
- CustomEmbeddings 클래스
- 임베딩 생성 함수

**해결**: 단일 책임 원칙(SRP) 적용하여 3개 모듈로 분리

1. `app/core/model_loader.py` (124줄)
   - get_device(): 디바이스 선택
   - load_llm_model(): LLM 로딩
   - load_rerank_model(): 리랭크 모델 로딩
   - cleanup_models(): 리소스 해제

2. `app/core/embeddings.py` (101줄)
   - CustomEmbeddings 클래스
   - get_embeddings(): 임베딩 생성
   - load_embedding_model(): 임베딩 모델 로딩

3. `app/core/vector_store.py` (65줄)
   - load_vector_db(): 벡터 DB 로딩/생성

4. `app/core/models.py` (83줄, 리팩토링 후)
   - 전역 변수 관리
   - 각 모듈에서 import한 함수들을 호출하여 전역 변수에 할당
   - 기존 인터페이스 유지 (하위 호환성)

### 결과
- ✅ 매직 넘버 제거로 유지보수성 향상
- ✅ RAG 로직 중복 제거 (40+ 줄 → 5줄)
- ✅ 모듈 분리로 단일 책임 원칙 준수
- ✅ 코드 가독성 대폭 향상
- ✅ Import 테스트 성공
- ✅ 주석 코드 유지 (사용자 요청 준수)

### Level 3 플랜 작성
- 의존성 주입 패턴 도입 방안 제시
- 장단점 분석 및 권장 시점 명시
- 현재 프로젝트 규모에는 과도할 수 있음을 명시
- 실제 필요성이 생길 때 진행하도록 권장
