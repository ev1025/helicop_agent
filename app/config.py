# config.py - JSON 기반 설정 로더

import os
import json
from pathlib import Path

class Config:
    """JSON 파일 기반 설정 관리 클래스"""

    def __init__(self):
        """설정 초기화 및 JSON 파일 로드"""
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self._load_configs()

    def _load_json(self, filename: str) -> dict:
        """
        JSON 파일을 로드합니다.

        Args:
            filename: 로드할 JSON 파일명 (config/ 폴더 기준)

        Returns:
            dict: JSON 데이터
        """
        config_path = self.BASE_DIR / "config" / filename
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_configs(self):
        """모든 설정 파일을 로드하여 속성으로 설정"""

        # ==================== 모델 설정 ====================
        models_cfg = self._load_json('models.json')
        self.LLM_MODEL = models_cfg['llm_model']
        self.EMBEDDING_MODEL = models_cfg['embedding_model']
        self.RERANKER_MODEL = models_cfg['reranker_model']

        # GPU 사용 여부 (환경변수 우선, JSON 기본값)
        self.USE_EMBEDDING_GPU = os.getenv('USE_EMBEDDING_GPU',
                                           str(models_cfg['use_embedding_gpu'])).lower() in ('true', '1', 'yes')
        self.USE_RERANKER_GPU = os.getenv('USE_RERANKER_GPU',
                                        str(models_cfg['use_reranker_gpu'])).lower() in ('true', '1', 'yes')

        # ==================== RAG 설정 ====================
        rag_cfg = self._load_json('rag.json')

        # Parent-Child Document Retrieval 모드
        self.PARENT_DOCUMENT_MODE = rag_cfg.get('parent_document_mode', False)

        # Child chunk 설정 (Parent-Child 모드용)
        self.CHILD_CHUNK_SIZE = rag_cfg.get('child_chunk_size', 400)
        self.CHILD_CHUNK_OVERLAP = rag_cfg.get('child_chunk_overlap', 50)

        # Parent chunk 설정 (Parent-Child 모드용)
        self.PARENT_CHUNK_SIZE = rag_cfg.get('parent_chunk_size', 1500)
        self.PARENT_CHUNK_OVERLAP = rag_cfg.get('parent_chunk_overlap', 150)

        # 일반 chunk 설정 (일반 모드용)
        self.CHUNK_SIZE = rag_cfg['chunk_size']
        self.CHUNK_OVERLAP = rag_cfg['chunk_overlap']

        self.VECTOR_SEARCH_TOP_K = rag_cfg['vector_search_top_k']

        # Reranker 설정
        self.USE_RERANKER = rag_cfg.get('use_reranker', False)
        self.RERANKER_TOP_K = rag_cfg['reranker_top_k']
        self.RERANKER_SCORE_THRESHOLD = rag_cfg['reranker_score_threshold']

        # Hybrid Retrieval 설정 (BM25 + Vector Search)
        self.USE_HYBRID_SEARCH = rag_cfg.get('use_hybrid_search', False)
        self.BM25_WEIGHT = rag_cfg.get('bm25_weight', 0.3)
        self.VECTOR_WEIGHT = rag_cfg.get('vector_weight', 0.7)

        self.MAX_CONTEXT_LENGTH = rag_cfg['max_context_length']

        # RAG 컨텍스트 처리 상수
        self.RAG_CONTEXT_MAX_LENGTH = rag_cfg['rag_context_max_length']
        self.RAG_CONTEXT_SENTENCE_BOUNDARY_MIN = rag_cfg['rag_context_sentence_boundary_min']
        self.CONTEXT_MIN_REMAINING_LENGTH = rag_cfg['context_min_remaining_length']

        # 경로 설정 (BASE_DIR 기준 절대 경로로 변환)
        self.VECTOR_DB_PATH = str((self.BASE_DIR / rag_cfg['vector_db_path']).resolve())
        self.COLLECTION_NAME = rag_cfg['collection_name']
        self.PDF_PATH = str((self.BASE_DIR / rag_cfg['pdf_path']).resolve())

        # ==================== 서버 설정 ====================
        server_cfg = self._load_json('server.json')
        self.HOST = server_cfg['host']
        self.PORT = server_cfg['port']
        self.STATIC_DIR = str((self.BASE_DIR / server_cfg['static_dir']).resolve())
        self.TEMPLATES_DIR = str((self.BASE_DIR / server_cfg['templates_dir']).resolve())
        self.SSL_KEYFILE = str((self.BASE_DIR / server_cfg['ssl_keyfile']).resolve())
        self.SSL_CERTFILE = str((self.BASE_DIR / server_cfg['ssl_certfile']).resolve())

        # ==================== 오디오 설정 ====================
        audio_cfg = self._load_json('audio.json')
        self.SAMPLE_RATE = audio_cfg['sample_rate']
        self.FRAME_DURATION_MS = audio_cfg['frame_duration_ms']
        self.SILENCE_THRESHOLD_SEC = audio_cfg['silence_threshold_sec']
        self.MAX_AUDIO_FILE_SIZE = audio_cfg['max_audio_file_size']

        # ==================== 프롬프트 설정 ====================
        prompts_cfg = self._load_json('prompts.json')
        self.SYSTEM_PROMPT = prompts_cfg['system_prompt']
        self.RAG_INSTRUCTION_TEMPLATE = prompts_cfg['rag_instruction_template']
        self.GENERAL_INSTRUCTION_TEMPLATE = prompts_cfg['general_instruction_template']

        # ==================== 로깅 설정 ====================
        logging_cfg = self._load_json('logging.json')
        self.LOG_LEVEL = logging_cfg['log_level']
        self.LOG_FILE = logging_cfg['log_file']
        self.LOG_FORMAT = logging_cfg['log_format']
        self.DEBUG_VERBOSE = os.getenv('DEBUG_VERBOSE',
                                       str(logging_cfg['debug_verbose'])).lower() in ('true', '1', 'yes')

        # ==================== LLM 생성 파라미터 ====================
        # 이 부분은 JSON으로 분리하지 않았으므로 하드코딩 유지
        self.MAX_NEW_TOKENS = 1024
        self.MIN_NEW_TOKENS = 5
        self.TEMPERATURE = 0.6
        self.TOP_P = 0.85
        self.REPETITION_PENALTY = 1.1
        self.MAX_INPUT_TOKENS = 20000  # Phase 4: Parent chunk 2500 지원 위해 확장 (6000 → 20000)

# 싱글톤 인스턴스
config = Config()
