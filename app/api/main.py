import os
import sys
import time
import torch
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

# .env 로드 (있으면) — LANGFUSE_*, CHAT_V2_BACKEND 등을 환경변수로 주입.
# 다른 import 보다 먼저 실행해야 config / 하위 모듈이 값을 읽을 수 있음.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"  # team_gitlab/.env
    load_dotenv(_env_path if _env_path.exists() else None)
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from app.config import config
from app.core import models
from app.api.routes import pages, chat_v2, voice

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 로깅 초기화
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),  # UTF-8 명시
        logging.StreamHandler(sys.stdout)  # 수정된 stdout 사용
    ]
)
logger = logging.getLogger(__name__)

if config.DEBUG_VERBOSE:
    logger.setLevel(logging.DEBUG)

# ==================== FastAPI 앱 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 수명주기 관리.

    - 임베딩 + Chroma 벡터 DB 를 lifespan 에 미리 로드 (RAG 용).
    - agent_v2 의 GGUF LLM 은 chat_v2.get_chat_model() 첫 요청 시 lazy load (~2초, 1회).
    - STT (Whisper) 는 stt_service 모듈 import 시 자동 로드.
    """
    logger.info("[lifespan] 임베딩 + Chroma 로드 시작")
    models.load_embedding_model()
    models.load_vector_db()
    logger.info("[lifespan] RAG 인프라 준비 완료. agent_v2 LLM 은 첫 요청 시 lazy load.")
    yield
    logger.info("[lifespan] 종료 정리")
    models.cleanup_models()


# FastAPI 애플리케이션 생성
app = FastAPI(
    lifespan=lifespan,              # 수명주기 관리자 등록
    title="Voice Chat RAG System",  # API 문서 제목
    version="1.0.0"                 # 버전
)

if not os.path.exists(config.TEMPLATES_DIR):
    os.makedirs(config.TEMPLATES_DIR, exist_ok=True)
    logger.info(f"디렉토리 생성: {config.TEMPLATES_DIR}")

# ==================== 라우터 등록 ====================
app.include_router(pages.router, tags=["Pages"])
app.include_router(chat_v2.router, tags=["ChatV2 (agent_v2 LangGraph + GGUF)"])
app.include_router(voice.router, tags=["Voice (STT → agent_v2)"])

# ==================== 헬스체크 ====================

@app.get("/health")
async def health_check():
    """
    서버 상태 확인 API

    Returns:
        dict: 서버 및 모델 상태 정보

    Note:
        로드밸런서나 모니터링 시스템에서 서버 상태 확인용
    """
    return {
        "status": "healthy",                            # 전체 상태
        "llm_loaded": models.model is not None,         # LLM 모델 로딩 상태
        "rag_loaded": models.collection is not None,    # RAG 시스템 상태
        "device": "cuda" if torch.cuda.is_available() else "cpu",   # 사용 중인 디바이스
        "timestamp": time.time()                        # 응답 시간
    }


# ==================== 전역 예외 처리 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    전역 예외 처리기

    Args:
        request: HTTP 요청 객체
        exc: 발생한 예외

    Returns:
        HTMLResponse: 오류 페이지

    Note:
        처리되지 않은 모든 예외를 캐치하여 사용자에게 친화적인 오류 메시지 표시
    """
    logger.error(f"전역 예외 발생: {exc}")
    logger.error(traceback.format_exc())
    return HTMLResponse(
        content=f"<h1>서버 오류</h1><p>예상치 못한 오류가 발생했습니다: {str(exc)}</p>",
        status_code=500
    )


# ==================== 서버 실행 ====================

if __name__ == "__main__":
    import uvicorn

    # Windows 콘솔 UTF-8 설정
    if sys.platform == 'win32':
        try:
            import io
            if not isinstance(sys.stdout, io.TextIOWrapper):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            if not isinstance(sys.stderr, io.TextIOWrapper):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not set UTF-8 encoding: {e}")

    # SSL 인증서 확인
    ssl_key = Path(config.SSL_KEYFILE)
    ssl_cert = Path(config.SSL_CERTFILE)
    use_ssl = ssl_key.exists() and ssl_cert.exists()

    # 서버 설정
    server_config = {
        "app": "app.api.main:app",  # 모듈 경로 형식으로 지정
        "host": config.HOST,
        "port": config.PORT,
        "reload": False,
        "log_level": "info"
    }

    if use_ssl:
        server_config["ssl_keyfile"] = str(ssl_key)
        server_config["ssl_certfile"] = str(ssl_cert)
        protocol = "https"
        logger.info(f"[HTTPS] 보안 서버 시작")
    else:
        protocol = "http"
        logger.warning("[HTTP] SSL 인증서를 찾을 수 없습니다")
        logger.warning("       음성 기능은 HTTPS에서만 작동합니다")

    logger.info(f"접속 주소: {protocol}://{config.HOST}:{config.PORT}")
    logger.info(f"로컬 접속: {protocol}://localhost:{config.PORT}")

    # 서버 시작
    uvicorn.run(**server_config)
