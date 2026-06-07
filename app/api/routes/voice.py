"""
음성 라우터 — STT → agent_v2 LangGraph → SSE 스트리밍 (+ 클라이언트 TTS).

기존 legacy LLM 의존(`app.core.llm`) 을 제거하고 chat_v2 의 `stream_answer` 를
재사용한다. TTS 는 브라우저 SpeechSynthesis (client_tts) 에 위임 — 별도 모델 불필요.

엔드포인트:
  - POST /speak/stream    : 음성 파일 업로드 → STT → SSE 답변 스트림
  - POST /voice_chat/     : 음성 파일 업로드 → STT → 답변 JSON (스트리밍 없는 버전)
  - POST /reset_tts/      : (client TTS 모드에서는 no-op)
"""

import os
import json
import tempfile
import traceback
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.config import config
from app.services import audio_service, stt_service
from app.utils import text_utils
from app.api.routes.chat_v2 import stream_answer

logger = logging.getLogger(__name__)
router = APIRouter()


class VoiceChatResponse(BaseModel):
    """음성 채팅 응답 (JSON 모드)."""
    user_text: str
    ai_response: str
    status: str = "success"


def _audio_to_text(upload: UploadFile, file_content: bytes) -> str:
    """업로드 음성 → WAV 변환 → VAD 추출 → Whisper STT 텍스트 반환.

    실패 시 HTTPException. 임시 파일은 호출자가 책임지지 않고 내부에서 정리.
    """
    filename, _ = text_utils.validate_file_info(upload)
    if not filename:
        raise HTTPException(status_code=400, detail="파일이 업로드되지 않았습니다.")

    suffix = text_utils.extract_file_suffix(filename)
    fd, temp_path = tempfile.mkstemp(suffix=f".{suffix}")
    os.close(fd)
    with open(temp_path, "wb") as f:
        f.write(file_content)

    processed_wav = None
    try:
        if suffix == "wav":
            wav_path = temp_path
        else:
            wav_path = audio_service.audio_processor.webm_to_wav(temp_path)

        processed_wav = audio_service.audio_processor.extract_voice(wav_path)
        if not processed_wav:
            raise HTTPException(status_code=400, detail="음성이 감지되지 않았습니다.")

        user_text = stt_service.transcribe_audio(processed_wav)
        if not user_text or not user_text.strip():
            raise HTTPException(status_code=400, detail="음성 인식 결과가 비어있습니다.")
        return user_text

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"음성 처리 실패: {e}")
    finally:
        for p in (temp_path, processed_wav):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as cleanup_error:
                    logger.warning(f"임시 파일 정리 실패 {p}: {cleanup_error}")


@router.post("/speak/stream")
async def speak_stream(audio_file: UploadFile = File(...)):
    """음성 파일 업로드 → STT → agent_v2 LangGraph SSE 스트림.

    TTS 는 클라이언트(SpeechSynthesis API) 에서 처리.
    """
    try:
        file_content = await audio_file.read()
        if len(file_content) > config.MAX_AUDIO_FILE_SIZE:
            raise HTTPException(status_code=400, detail="파일 크기가 너무 큽니다 (최대 10MB)")

        user_text = _audio_to_text(audio_file, file_content)
        logger.info(f"[voice] STT 결과: {user_text[:60]!r}")

        async def event_stream():
            # 1) 사용자 발화 메타 이벤트 (UI 가 transcript 표시)
            yield (
                f"data: "
                f"{json.dumps({'type': 'user_speech', 'content': f'🎤 인식된 음성: {user_text}'}, ensure_ascii=False)}"
                f"\n\n"
            )
            # 2) chat_v2 SSE 스트림에 위임
            async for ev in stream_answer(user_text):
                yield ev

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"speak_stream 오류: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice_chat/")
async def voice_chat(audio_file: UploadFile = File(...)):
    """음성 파일 → STT → 답변 JSON (스트리밍 없는 버전, 모바일/단순 클라이언트용)."""
    try:
        file_content = await audio_file.read()
        if len(file_content) > config.MAX_AUDIO_FILE_SIZE:
            raise HTTPException(status_code=400, detail="파일 크기가 너무 큽니다 (최대 10MB)")

        user_text = _audio_to_text(audio_file, file_content)

        ai_response_parts = []
        async for ev in stream_answer(user_text):
            line = ev.strip()
            if not line.startswith("data:"):
                continue
            try:
                payload = json.loads(line[len("data:"):].strip())
            except Exception:
                continue
            if payload.get("type") == "text":
                ai_response_parts.append(payload.get("content", ""))
            elif payload.get("type") == "error":
                raise HTTPException(status_code=500, detail=payload.get("content", "답변 생성 실패"))

        ai_response = "".join(ai_response_parts).strip() or "답변을 생성할 수 없습니다."

        return VoiceChatResponse(user_text=user_text, ai_response=ai_response, status="success")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"voice_chat 처리 실패: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")


@router.post("/reset_tts/")
async def reset_tts():
    """클라이언트 TTS 모드에서는 리셋할 서버 상태가 없으므로 no-op."""
    return JSONResponse({"ok": True, "message": "클라이언트 TTS 모드에서는 리셋이 불필요합니다."})
