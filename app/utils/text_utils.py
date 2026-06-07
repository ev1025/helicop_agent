import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ==================== 안전 체크 함수 ====================
def safe_string_check(value, substring):
    """
    None-safe 문자열 검사 함수
    
    Args:
        value: 검사할 값 (None일 수 있음)
        substring: 찾을 부분 문자열
    
    Returns:
        bool: substring이 value에 포함되어 있으면 True, 아니면 False
    
    Note:
        value가 None이거나 문자열이 아니면 안전하게 False 반환
    """
    if value is None or not isinstance(value, str):
        return False
    return substring in value


def safe_getattr(obj, attr_name, default_value=""):
    """
    None-safe getattr 함수 - 객체 속성 안전하게 접근
    
    Args:
        obj: 속성을 가져올 객체
        attr_name: 속성명
        default_value: 기본값 (속성이 없거나 None일 때)
    
    Returns:
        객체의 속성값 또는 기본값
    
    Note:
        예외가 발생하거나 속성이 None이면 기본값 반환
    """
    try:
        result = getattr(obj, attr_name, None)
        return result if result is not None else default_value
    except Exception:
        return default_value


def validate_file_info(file_obj):
    """
    파일 객체 정보 검증 및 안전한 추출
    
    Args:
        file_obj: FastAPI UploadFile 객체
    
    Returns:
        tuple: (파일명, 콘텐츠 타입)
    
    Note:
        파일 정보 추출 실패시 기본값 반환하여 안정성 확보
    """
    try:
        filename = safe_getattr(file_obj, 'filename', 'unknown_file.webm')
        content_type = safe_getattr(file_obj, 'content_type', '')
        
        if not isinstance(filename, str):
            filename = 'unknown_file.webm'
        if not isinstance(content_type, str):
            content_type = ''
        
        return filename, content_type
    except Exception as e:
        logger.error(f"파일 정보 추출 오류: {e}")
        return 'unknown_file.webm', ''


def extract_file_suffix(filename):
    """
    안전한 파일 확장자 추출
    
    Args:
        filename: 파일명
    
    Returns:
        str: 파일 확장자 (소문자)
    
    Note:
        확장자 추출 실패시 기본적으로 'webm' 반환
    """
    try:
        if not filename or not isinstance(filename, str):
            return 'webm'
        
        if '.' in filename:
            suffix = filename.split('.')[-1].lower()        # 마지막 점 이후 문자열을 소문자로
            return suffix if suffix else 'webm'
        else:
            return 'webm'       # 확장자가 없으면 기본값
    except Exception:
        return 'webm'


# ==================== TTS 전처리 ====================
def preprocess_text_for_tts(text: str) -> str:
    """
    TTS(Text-To-Speech)를 위한 텍스트 전처리
    
    Args:
        text: 원본 텍스트
    
    Returns:
        str: TTS에 적합하게 정리된 텍스트
    
    Note:
        마크다운, 특수문자, URL 등을 제거하여 TTS가 자연스럽게 읽을 수 있도록 처리
    """
    if not text or not text.strip():
        return ""
    
    # 유니코드 정규화
    text = unicodedata.normalize('NFC', text)
    
    # 개행, 탭 등을 공백으로
    text = re.sub(r'[\n\r\t\f\v]', ' ', text)
    
    # 마크다운 굵은체/이탤릭 제거: ***text***, **text**, *text*
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    
    # 마크다운 제목 제거: # Header, ## Header 등
    text = re.sub(r'#{1,6}\s*([^#\n]+)', r'\1', text)
    
    # 인라인 코드 제거: `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # HTML 태그 제거: <tag>content</tag>
    text = re.sub(r'<[^>]+>', '', text)
    
    # 마크다운 특수문자 제거
    text = re.sub(r'[*#@|~^{}\\`]', '', text)
    text = re.sub(r'[\[\]()]', '', text)
    
    # 이모티콘 제거 (TTS가 읽기 어려움)
    text = re.sub(r'[📚📖🎤📄🎯🗣️🤖💡✅❌🧠]', '', text)
    
    # 리스트 번호 제거: 1. item, 2) item
    text = re.sub(r'^\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
    
    # 괄호 안 내용 제거: (부연설명)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 연속 문장부호 정리: !!, ?? 등
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r':{2,}', ':', text)
    
    # URL 및 이메일 제거
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    
    # 여러 공백을 하나의 공백으로
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 너무 짧은 텍스트는 빈 문자열 반환
    if len(text) < 2:
        return ""
    
    # 문장 종결 처리 (마침표 추가)
    if text and not text.endswith(('.', '!', '?', '。', '!', '?')):
        if any(text.endswith(ending) for ending in ['다', '요', '야', '지', '네', '어', '아']):
            text += '.'
    
    return text


def clean_tts_text(text: str) -> str:
    """
    TTS 텍스트 최종 정리
    
    Args:
        text: 전처리된 텍스트
    
    Returns:
        str: TTS 최종 준비 완료 텍스트
    
    Note:
        특수 기호를 한글로 치환하고 TTS에 방해되는 문자 제거
    """
    clean_text = preprocess_text_for_tts(text)
    if not clean_text:
        return ""
    
    # 특수 기호를 한글로 치환 (TTS로 읽기 위함)
    replacements = {
        '&amp;': '그리고', '@': '골뱅이', '#': '샵', '%': '퍼센트', '$': '달러',
        '€': '유로', '£': '파운드', '¥': '엔', '₩': '원',
        '+': '플러스', '-': '마이너스', '=': '같다', '×': '곱하기', '÷': '나누기',
        '°': '도', '™': '상표', '®': '등록상표', '©': '저작권',
    }
    
    for symbol, replacement in replacements.items():
        clean_text = clean_text.replace(symbol, replacement)
    
    # TTS에 방해되는 문자 제거
    problematic_chars = ['*', '#', '@', '|', '~', '^', '{', '}', '[', ']', '`', '\\']
    for char in problematic_chars:
        clean_text = clean_text.replace(char, '')
    
    # 공백 정리
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # 의미있는 텍스트가 남아있는지 확인
    if not clean_text or clean_text.isspace() or len(clean_text.strip()) < 2:
        return ""
    
    # 한글/영문/숫자가 포함된 단어만 유지
    words = clean_text.split()
    cleaned_words = []
    for word in words:
        if re.search(r'[가-힣a-zA-Z0-9]', word):
            cleaned_words.append(word)
    
    clean_text = ' '.join(cleaned_words)
    return clean_text


def is_complete_sentence(text: str) -> bool:
    """
    문장 완성 여부 확인 (TTS 처리 타이밍 결정용)
    
    Args:
        text: 검사할 텍스트
    
    Returns:
        bool: 완전한 문장이면 True
    
    Note:
        명시적 문장부호, 한국어 어미, 길이를 종합 고려
    """
    text = text.strip()
    if not text:
        return False
    
    # 명시적 문장 종결 문장부호
    explicit_endings = ['.', '!', '?', '。', '!', '?']
    if any(text.endswith(punct) for punct in explicit_endings):
        return True
    
    # 한국어 문장 종결 어미 패턴
    korean_sentence_endings = [
        r'습니다\.?$', r'했습니다\.?$', r'됩니다\.?$', r'입니다\.?$',
        r'었습니다\.?$', r'있습니다\.?$', r'없습니다\.?$', r'해요\.?$', r'해\.?$',
        r'야\.?$', r'지\.?$', r'네\.?$', r'요\.?$', r'어\.?$', r'아\.?$',
        r'죠\.?$', r'구나\.?$', r'군요\.?$', r'다\.?$', r'라\.?$', r'자\.?$',
        r'까\.?$', r'나\.?$', r'가\.?$', r'겠다\.?$', r'한다\.?$', r'된다\.?$'
    ]
    
    for pattern in korean_sentence_endings:
        if re.search(pattern, text):
            return True
    
    # 텍스트가 너무 길면 강제로 문장 완료로 간주
    if len(text) > 150:
        return True
    
    return False

def is_tts_worthy_text(text: str) -> bool:
    """
    TTS 처리 가치가 있는 텍스트인지 판단
    
    Args:
        text: 검사할 텍스트
    
    Returns:
        bool: TTS 처리할 가치가 있으면 True
    
    Note:
        너무 짧거나 의미없는 텍스트는 TTS 처리하지 않음
    """
    if not text or not text.strip():
        return False
    
    # TTS 정리 후 길이 확인
    clean_text = clean_tts_text(text)
    if len(clean_text.strip()) < 3:
        return False
    
    # 한글이나 영문이 포함되어야 함
    if not re.search(r'[가-힣a-zA-Z]', clean_text):
        return False
    
    # 의미있는 문자 비율 확인
    meaningful_chars = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', clean_text)
    if len(meaningful_chars.strip()) < 2:
        return False
    
    return True