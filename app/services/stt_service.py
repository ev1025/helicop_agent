import os
import torch
import logging
from transformers import pipeline

# 로깅 설정
logger = logging.getLogger(__name__)

# Whisper 모델 경로 및 환경 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "my_whisper_small")
use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"

whisper_pipe = None

def initialize_whisper():
    """Whisper 파이프라인 초기화"""
    global whisper_pipe
    try:
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_DIR,
            device=device,
            return_timestamps=True
        )
        logger.info(f"Whisper 모델 로드 성공 (디바이스: {device})")
    except Exception as init_error:
        logger.error(f"Whisper 모델 또는 파이프라인 로드 오류: {init_error}")
        whisper_pipe = None

def transcribe_audio(audio_path: str) -> str:
    """
    오디오 파일 경로를 받아 Whisper 파이프라인으로 텍스트 변환.
    """
    global whisper_pipe
    
    # 파이프라인이 없으면 초기화 시도
    if whisper_pipe is None:
        initialize_whisper()
    
    # 여전히 None이면 오류 발생
    if whisper_pipe is None:
        raise RuntimeError("Whisper 파이프라인이 로드되지 않았습니다.")
    
    try:
        # 파일 존재 확인
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("오디오 파일이 비어있습니다.")
        
        logger.info(f"STT 처리 시작: {audio_path} (크기: {file_size} bytes)")
        
        # Whisper 실행
        result = whisper_pipe(audio_path)
        
        # 결과 검증
        if not result or "text" not in result:
            raise ValueError("Whisper 결과가 올바르지 않습니다.")
        
        text = result["text"].strip()
        
        if not text:
            raise ValueError("음성 인식 결과가 비어있습니다.")
        
        logger.info(f"STT 처리 완료: {text[:50]}...")
        return text
        
    except Exception as transcribe_error:
        logger.error(f"음성 인식 오류: {transcribe_error}")
        raise

# 초기화 실행
initialize_whisper()
