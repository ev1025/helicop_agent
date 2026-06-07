# Project AI Instructions

This file contains project-specific instructions for AI assistants working on this project.

## Base Instructions

This project follows the global AI metadata defined in `.agents/agents_meta.md`.

## Project-Specific Instructions

- 모든 AI 응답은 한국어로 제공하십시오.

## 🚨 현재 프로젝트 상태 (2025-11-18)

**Phase 3 진행 중 - 그룹 1 (Step 1-2) 완료, 테스트 대기**

### 빠른 참조
- 상세 진행 상황: `PHASE3_PROGRESS.md`
- 빠른 상태: `.agents/PHASE3_STATUS.md`
- Phase 2 분석: `results/phase2_final_analysis.md`

### 다음 작업
1. 그룹 1 테스트 실행: `jupyter_notebooks/phase3_step1-2_prompt_and_query.ipynb`
2. 결과 검증 후 그룹 2 (Multi-Query) 진행

### 최근 커밋
```
59f1ffe - Phase 3 Step 1-2: Prompt and Query improvements
c619b40 - Add Phase 2 multi-config benchmark results
3611f8a - Minor notebook metadata update
```

### 프로젝트 개요
- FastAPI 기반 음성·텍스트 챗봇 백엔드 (RAG 시스템)
- 헬리콥터 교재 PDF 기반 항공역학 전문 AI 튜터

### 아키텍처 (2025-10-28 리팩토링 완료)

#### 진입점
- `main.py` - 애플리케이션 실행 진입점

#### 설정 관리 (JSON 기반)
- `config/` - JSON 설정 파일
  - `models.json` - LLM, 임베딩, 리랭크 모델 설정
  - `rag.json` - RAG 파라미터 (청크 크기, 검색 설정 등)
  - `server.json` - 서버 설정 (호스트, 포트, SSL)
  - `audio.json` - 오디오 처리 설정
  - `prompts.json` - 시스템 프롬프트 및 템플릿
  - `logging.json` - 로깅 설정
- `app/config.py` - JSON 로더

#### 핵심 로직 (app/core/)
- `models.py` - 전역 모델 인스턴스 관리 및 통합 인터페이스
- `model_loader.py` - LLM, 임베딩, 리랭크 모델 로딩
- `embeddings.py` - CustomEmbeddings 클래스, 임베딩 생성
- `vector_store.py` - Chroma 벡터DB 생성/로딩
- `llm.py` - LLM 텍스트 생성 및 스트리밍 (프롬프트 구성, 토큰화, 스트리밍)

#### 서비스 레이어 (app/services/)
- `rag_service.py` - 검색→리랭크→컨텍스트 조합 RAG 파이프라인
- `audio_service.py` - 음성 입력 처리, VAD (Voice Activity Detection)
- `stt_service.py` - Whisper STT (Speech-To-Text)
- `my_whisper_small/` - Whisper 모델 파일

#### API 라우팅 (app/api/)
- `main.py` - FastAPI 앱 생성 및 라우터 등록
- `routes/chat.py` - 텍스트 채팅 엔드포인트
- `routes/voice.py` - 음성 채팅 엔드포인트
- `routes/pages.py` - HTML 페이지 라우트

#### 유틸리티 (app/utils/)
- `text_utils.py` - 텍스트 처리, TTS 전처리, 파일 검증

#### UI
- `app/templates/` - HTML 템플릿
- `app/static/`, `static/` - 정적 파일 (CSS, JS, 이미지)

## Auto-managed Files

The following files are automatically read and updated based on this instruction file:
- `.agents/agents_meta.md` - Project metadata and architecture
- `.agents/plan.md` - Task planning and progress tracking
- `.agents/consult.md` - Consultation history and decisions
