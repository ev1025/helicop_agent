import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core import models
from app.services import rag_service
from app.core import llm

logger = logging.getLogger(__name__)

# APIRouter 생성
router = APIRouter()

# ==================== 텍스트 채팅 라우트 ====================

@router.get("/chat/stream")
async def chat_stream(message: str, client_tts: bool = True):
    """
    텍스트 채팅 실시간 스트리밍 API

    Args:
        message: 사용자 메시지
        client_tts: 클라이언트 TTS 사용 여부

    Returns:
        StreamingResponse: Server-Sent Events 스트림

    Note:
        텍스트 입력 -> RAG 검색 -> LLM 답변 생성 -> TTS 데이터 전송
    """
    # 모델 로딩 확인
    if not models.model or not models.tokenizer:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    def event_stream():
        """Server-Sent Events 스트림 생성기"""
        try:
            # [1] RAG 검색 시작 알림
            yield f"data: {json.dumps({'type': 'info', 'content': '📚 관련 문서를 검색하고 있습니다...'}, ensure_ascii=False)}\n\n"

            # [2] RAG 검색
            rag_results = rag_service.rag_search_with_rerank(message)

            # [3] RAG 결과 포맷팅
            formatted = rag_service.format_rag_results(rag_results)

            # [4] RAG 결과 SSE 이벤트 생성
            yield from rag_service.generate_rag_sse_events(rag_results)

            # [5] LLM 답변 생성
            yield f"data: {json.dumps({'type': 'info', 'content': '🤖 답변을 생성하고 있습니다...'}, ensure_ascii=False)}\n\n"

            # [6] LLM 스트리밍 및 TTS
            yield from llm.llm_streamer_with_rag_and_tts(
                prompt=message,
                use_rag=formatted['use_rag'],
                rag_context=formatted['rag_context'],
                client_tts=True     # 클라이언트 TTS 활성화
            )

        except Exception as e:
            logger.error(f"event_stream 오류: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'스트리밍 처리 중 오류: {str(e)}'}, ensure_ascii=False)}\n\n"

    # Server-Sent Events 응답 반환
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",     # SSE MIME 타입
        headers={
            "Cache-Control": "no-cache",    # 캐싱 비활성화
            "Connection": "keep-alive",     # 연결 유지
            "Access-Control-Allow-Origin": "*", # CORS 허용
        }
    )
