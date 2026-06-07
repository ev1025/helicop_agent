"""
모델 관리 모듈

이 모듈은 전역 모델 인스턴스를 관리하고,
각 모델 로딩 함수를 외부 모듈에서 import하여 사용합니다.
"""

import os
import logging

# 서브모듈에서 함수들 import
from app.core.model_loader import get_device, load_llm_model as _load_llm, load_reranker_model as _load_reranker, cleanup_models
from app.core.embeddings import CustomEmbeddings, load_embedding_model as _load_embedding
from app.core.vector_store import (
    load_vector_db as _load_vector_db,
    create_parent_document_vectordb as _create_parent_doc_db
)
from app.config import config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "0")

logger = logging.getLogger(__name__)

# ==================== 전역 변수 ====================
# 전역 모델 인스턴스들
model = None
tokenizer = None
embedding_model = None
embedding_tokenizer = None
reranker_model = None
reranker_tokenizer = None
collection = None
parent_store = None  # Parent Document Retriever용 부모 문서 저장소


# ==================== 모델 로딩 함수 ====================

def load_llm_model():
    """LLaMA-3 모델 로딩 (전역 변수에 할당)"""
    global model, tokenizer
    model, tokenizer = _load_llm(config.LLM_MODEL)


def load_embedding_model():
    """임베딩 모델 로딩 (전역 변수에 할당)"""
    global embedding_model, embedding_tokenizer
    device = get_device(config.USE_EMBEDDING_GPU)
    embedding_model, embedding_tokenizer = _load_embedding(config.EMBEDDING_MODEL, device)


def load_reranker_model():
    """리랭커 모델 로딩 (전역 변수에 할당)"""
    global reranker_model, reranker_tokenizer
    if config.RERANKER_MODEL is None:
        reranker_model = None
        reranker_tokenizer = None
        logger.info("Reranker model is None, skipping load")
        return
    device = get_device(config.USE_RERANKER_GPU)
    reranker_model, reranker_tokenizer = _load_reranker(config.RERANKER_MODEL, device)


def load_vector_db():
    """벡터 DB 로딩/생성 (전역 변수에 할당)"""
    global collection, parent_store

    # CustomEmbeddings 인스턴스 생성
    custom_emb = CustomEmbeddings(
        embedding_model,
        embedding_tokenizer,
        get_device(config.USE_EMBEDDING_GPU),
        model_name=config.EMBEDDING_MODEL
    )

    # Parent-Child 모드 확인
    parent_document_mode = getattr(config, 'PARENT_DOCUMENT_MODE', False)

    if parent_document_mode:
        logger.info("Parent-Child Document Retrieval 모드로 벡터 DB 로딩")

        # Parent-Child 벡터 DB 생성/로딩
        collection, parent_store = _create_parent_doc_db(
            embedding_function=custom_emb,
            model_name=config.EMBEDDING_MODEL,
            vector_db_path=config.VECTOR_DB_PATH,
            collection_name=config.COLLECTION_NAME,
            pdf_path=config.PDF_PATH,
            child_chunk_size=getattr(config, 'CHILD_CHUNK_SIZE', 300),
            child_chunk_overlap=getattr(config, 'CHILD_CHUNK_OVERLAP', 30),
            parent_chunk_size=getattr(config, 'PARENT_CHUNK_SIZE', 1500),
            parent_chunk_overlap=getattr(config, 'PARENT_CHUNK_OVERLAP', 150)
        )
    else:
        logger.info("일반 벡터 DB 로딩 모드")

        # 일반 벡터 DB 로딩
        collection = _load_vector_db(
            embedding_function=custom_emb,
            model_name=config.EMBEDDING_MODEL,
            vector_db_path=config.VECTOR_DB_PATH,
            collection_name=config.COLLECTION_NAME,
            pdf_path=config.PDF_PATH,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        parent_store = None


def load_all_models():
    """모든 모델 초기화"""
    load_llm_model()
    load_embedding_model()
    load_reranker_model()
    load_vector_db()
    logger.info("전체 모델 로딩 완료")
