import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.config import config

logger = logging.getLogger(__name__)

# APIRouter 생성
router = APIRouter()

# 템플릿 엔진 설정
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

# ==================== HTML 페이지 라우트 ====================

@router.get("/", response_class=HTMLResponse)
async def select_mode(request: Request):
    """
    모드 선택 페이지 (홈페이지)

    Returns:
        HTMLResponse: 모드 선택 HTML 페이지

    Note:
        사용자가 텍스트 채팅 또는 음성 채팅을 선택할 수 있는 메인 페이지
    """
    try:
        return templates.TemplateResponse(request, "select_mode.html")
    except Exception as e:
        logger.error(f"select_mode 템플릿 오류: {e}")
        raise HTTPException(status_code=500, detail="페이지 로딩 실패")


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """
    텍스트 채팅 페이지

    Returns:
        HTMLResponse: 텍스트 채팅 인터페이스

    Note:
        키보드로 텍스트 입력하여 채팅하는 페이지
    """
    try:
        return templates.TemplateResponse(request, "chat.html")
    except Exception as e:
        logger.error(f"chat_page 템플릿 오류: {e}")
        raise HTTPException(status_code=500, detail="페이지 로딩 실패")


@router.get("/speak", response_class=HTMLResponse)
async def speak_page(request: Request):
    """
    음성 채팅 페이지

    Returns:
        HTMLResponse: 음성 채팅 인터페이스

    Note:
        마이크로 음성 입력하여 채팅하는 페이지
    """
    try:
        return templates.TemplateResponse(request, "speak.html")
    except Exception as e:
        logger.error(f"speak_page 템플릿 오류: {e}")
        raise HTTPException(status_code=500, detail="페이지 로딩 실패")


@router.get("/mp3", response_class=HTMLResponse)
async def mp3_page(request: Request):
    """
    MP3 페이지 (추가 기능)

    Returns:
        HTMLResponse: MP3 관련 페이지
    """
    try:
        return templates.TemplateResponse(request, "mp3.html")
    except Exception as e:
        logger.error(f"mp3_page 템플릿 오류: {e}")
        raise HTTPException(status_code=500, detail="페이지 로딩 실패")
