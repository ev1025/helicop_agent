# 진행 계획

> 모든 작업 항목은 `[대주제]` 형식으로 시작합니다.

## 진행 완료 (Completed)

### [리팩토링] 프로젝트 구조 개편 (2025-10-28)
1. ✅ Config를 JSON 파일로 분리 (config/ 폴더 생성)
   - models.json, rag.json, server.json, audio.json, prompts.json, logging.json
   - app/config.py를 JSON 로더로 변경
2. ✅ 모듈 구조를 계층화 (core/, services/, api/, utils/)
   - core/: models.py, llm.py
   - services/: rag_service.py, audio_service.py, stt_service.py
   - api/: main.py, routes/(chat.py, voice.py, pages.py)
   - utils/: text_utils.py
3. ✅ 모든 파일의 import 경로 수정
4. ✅ 새로운 진입점 생성 (main.py)
5. ✅ 불필요한 파일 정리
   - 백업 파일 삭제 (_old_files_backup/, main.py.bak)
   - 캐시 파일 삭제 (__pycache__, *.pyc)
   - 시스템 파일 삭제 (.DS_Store, .idea/, .ipynb_checkpoints/)
   - 중복 폴더 삭제 (app/ssl_certificates/)
   - 사용하지 않는 모델 삭제 (app/my_whisper_base/)
6. ✅ Whisper 모델 경로 수정 (app/my_whisper_small → app/services/my_whisper_small)
7. ✅ .gitignore 생성
8. ✅ 리팩토링 테스트 및 동작 확인

**성과**:
- 설정을 코드에서 분리하여 유지보수성 향상
- 명확한 계층 구조로 코드 가독성 개선
- 깔끔한 프로젝트 구조

### [리팩토링] Jupyter 노트북 Import 경로 업데이트 (2025-10-29)
1. ✅ llm_tester-Copy1.ipynb 복원
   - 백업 폴더(AITutor_dy_v2_bak251028_first_gitlab)에서 복사
2. ✅ 3개 노트북 Import 경로 수정
   - llm_tester.ipynb
   - llm_쿼리 개선.ipynb
   - llm_tester-Copy1.ipynb
3. ✅ Import 패턴 업데이트
   - `import config, models, llm, utils` → `from app.config import config` 등
   - `config.config.XXX` → `config.XXX` 패턴 수정
   - `import rag` → `from app.services import rag_service as rag`
   - `from app.models import CustomEmbeddings` → `from app.core.models import CustomEmbeddings`
4. ✅ Import 테스트 성공

**성과**:
- 리팩토링된 프로젝트 구조와 노트북 호환성 확보
- 개발/테스트 도구 정상 작동 보장

### [리팩토링] Level 1, 2 코드 품질 개선 (2025-10-29)
1. ✅ 매직 넘버 상수화
   - config/rag.json에 상수 추가: rag_context_max_length(1500), rag_context_sentence_boundary_min(800), context_min_remaining_length(200)
   - llm.py, rag_service.py에서 config 상수 사용
2. ✅ RAG 검색 로직 중복 제거
   - rag_service.py에 헬퍼 함수 추가:
     - format_rag_results(): RAG 결과 포맷팅
     - generate_rag_sse_events(): SSE 이벤트 생성
   - chat.py, voice.py에서 중복 코드 제거
3. ✅ models.py 모듈 분리 (단일 책임 원칙 적용)
   - app/core/model_loader.py: LLM, 임베딩, 리랭크 모델 로딩
   - app/core/embeddings.py: CustomEmbeddings 클래스 + 임베딩 로직
   - app/core/vector_store.py: 벡터DB 관리
   - app/core/models.py: 전역 변수 관리 및 통합 인터페이스
4. ✅ Import 테스트 성공

**성과**:
- 코드 가독성 및 유지보수성 향상
- 단일 책임 원칙(SRP) 준수
- 중복 코드 제거로 일관성 확보
- 모듈 간 명확한 경계 설정

## 진행 중 (In Progress)

## 진행 예정 (To Do)

### [리팩토링] Level 3 - 의존성 주입 패턴 도입 (선택 사항)
**목적**: 전역 변수 제거, 테스트 가능성 향상, 명시적 의존성 관리

**작업 내용**:
1. 서비스 클래스 도입
   - LLMService 클래스 생성 (모델, 토크나이저 의존성 주입)
   - RAGService 클래스 생성 (임베딩, 리랭크, 벡터DB 의존성 주입)
   - AudioService 클래스 재구성

2. FastAPI 의존성 주입
   ```python
   async def get_llm_service() -> LLMService:
       return LLMService(models.model, models.tokenizer)

   @router.post("/chat")
   async def chat(llm: LLMService = Depends(get_llm_service)):
       ...
   ```

3. 단위 테스트 작성
   - Mock 객체를 주입하여 각 서비스 테스트
   - RAG 파이프라인 테스트
   - LLM 응답 생성 테스트

**예상 소요 시간**: 5-8시간

**장점**:
- 테스트 가능한 코드
- 명시적 의존성 관리
- 확장성 증가 (멀티테넌트, A/B 테스팅 등)

**단점**:
- 현재 프로젝트 규모에는 과도할 수 있음
- 코드 복잡도 증가
- 초기 학습 곡선

**권장 시점**: 다음 중 하나에 해당할 때
- 단위 테스트가 필요할 때
- 여러 모델을 동시에 사용해야 할 때
- 멀티테넌트 기능이 필요할 때
- 팀 규모가 커질 때
