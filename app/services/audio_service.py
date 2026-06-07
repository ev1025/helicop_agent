import os
import tempfile
import subprocess
import wave
import logging
import numpy as np
import webrtcvad
from app.config import config

logger = logging.getLogger(__name__)

class AudioProcessor:
    """오디오 파일 처리 클래스"""
    

    def __init__(self):
        """오디오 처리 매개변수 초기화"""
        self.sample_rate = config.SAMPLE_RATE               # 샘플링 주파수 (16kHz - 음성 인식 표준)
        self.frame_duration = config.FRAME_DURATION_MS      # 프레임 지속 시간 (30ms)
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000) # 프레임 크기 계산
        self.silence_sec = config.SILENCE_THRESHOLD_SEC     # 무음 감지 기준 시간 (1초)
        self.max_silence_frames = int(self.silence_sec * 1000 / self.frame_duration)    # 최대 무음 프레임 수
        # Voice Activity Detection 초기화 (0-3: 민감도, 3이 가장 엄격)
        self.vad = webrtcvad.Vad(3)     # webrtcvad 캡슐화 
        
    

    def webm_to_wav(self, src_path: str) -> str:
        """
        WebM 파일을 WAV 파일로 변환
        
        Args:
            src_path: 원본 WebM 파일 경로
        
        Returns:
            str: 변환된 WAV 파일 경로
        
        Raises:
            ValueError: 파일이 없거나 변환 실패시
        
        Note:
            FFmpeg를 사용하여 16kHz 모노 WAV로 변환
        """

        # 입력 검증
        if not src_path or not os.path.exists(src_path):
            raise ValueError(f"파일이 존재하지 않습니다: {src_path}")
        
        # 임시 WAV 파일 생성
        fd, dst_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)    # 파일 디스크립터 즉시 닫기 (FFmpeg가 파일에 쓸 수 있도록)
        
        try:
            # FFmpeg 명령 실행
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",    # FFmpeg 설정
                 "-i", src_path,                                    # 입력 파일
                 "-ar", str(self.sample_rate),                      # 샘플링 레이트 16kHz
                 "-ac", "1",                                        # 모노 채널
                 "-f", "wav",                                       # WAV 형식
                 "-y", dst_path],                                   # 출력 파일 (덮어쓰기)
                check=True,  # 오류 시, 예외 발생
                timeout=30   # 30초 타임아웃
            )
            
            if not os.path.exists(dst_path):    # 변환 결과 검증
                raise ValueError("WAV 변환 실패")
            return dst_path
            
        except subprocess.TimeoutExpired:
            # 타임아웃 발생 시, 정리 후 예외
            if os.path.exists(dst_path):
                os.remove(dst_path)
            raise ValueError("오디오 변환 시간 초과")
        
        except Exception as e:
            # 기타 오류 발생 시, 정리 후 예외
            if os.path.exists(dst_path):
                os.remove(dst_path)
            raise ValueError(f"오디오 변환 실패: {str(e)}")
    
    def extract_voice(self, wav_path: str) -> str:
        """
        WAV 파일에서 VAD로 음성 구간만 추출
        
        Args:
            wav_path: WAV 파일 경로
        
        Returns:
            Optional[str]: 음성 구간만 추출된 WAV 파일 경로 (실패시 None)
        
        Note:
            webrtcvad를 사용하여 음성 구간 검출 후 무음 제거
        """

        # 입력 검증
        if not wav_path or not os.path.exists(wav_path):
            logger.error(f"WAV 파일이 존재하지 않습니다: {wav_path}")
            return None
        
        try:
            # WAV 파일 읽기 및 검증
            with wave.open(wav_path, "rb") as wf:
                if wf.getframerate() != self.sample_rate or wf.getnchannels() != 1:
                    raise ValueError("wav must be 16 kHz mono")     # 16kHz 모노 채널이 아니면 오류
                pcm_data = wf.readframes(wf.getnframes())           # 모든 프레임 읽기
            
            # PCM 데이터를 numpy 배열로 변환
            pcm = np.frombuffer(pcm_data, dtype=np.int16)
            
            # 프레임 크기에 맞도록 패딩 (VAD 처리를 위해)
            if len(pcm) % self.frame_size:
                pad = self.frame_size - (len(pcm) % self.frame_size)
                pcm = np.pad(pcm, (0, pad), 'constant')
            
            # 음성 구간 추출
            voiced = []     # 음성이 감지된 프레임들
            silence = 0     # 연속 무음 프레임 카운터
            
            # 프레임 단위로 음성 활동 검출
            for i in range(0, len(pcm), self.frame_size):
                frame = pcm[i:i+self.frame_size]
                
                # webrtcvad로 음성 여부 판단
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    voiced.extend(frame)        # 음성 프레임 추가
                    silence = 0                 # 무음 카운터 리셋
                else:
                    silence += 1    # 짧은 무음 보존 (자연스러운 발음을 위해)
                    if silence <= 2:
                        voiced.extend(frame)
                
                # 긴 무음이 지속되면 현재까지의 음성을 사용하고 루프를 종료
                if silence >= self.max_silence_frames:
                    # 음성 길이가 충분하지 않다면 계속 탐색
                    if len(voiced) < self.sample_rate * 0.2:    # 0.2초 미만이면 스킵 후 계속
                        voiced = []
                        silence = 0
                        continue
                    break
            
            # 음성이 전혀 감지되지 않은 경우
            if not voiced:
                logger.warning("음성이 감지되지 않았습니다")
                return None
            
            # 추출된 음성을 새 WAV 파일로 저장
            fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)      # 모노 채널
                wf.setsampwidth(2)      # 16비트
                wf.setframerate(self.sample_rate)   # 16kHz
                wf.writeframes(np.array(voiced, dtype=np.int16).tobytes())    # PCM 데이터 쓰기
            
            return out_path
            
        except Exception as e:
            logger.error(f"음성 추출 오류: {e}")
            return None

# 오디오 프로세서 인스턴스 생성
audio_processor = AudioProcessor()