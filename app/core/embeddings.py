"""
임베딩 관련 기능 모듈

이 모듈은 텍스트를 벡터 임베딩으로 변환하는 기능을 제공합니다.
"""

import torch
import logging
import os
from typing import List
from transformers import AutoTokenizer, AutoModel

# 🔥 trust_remote_code 환경변수 설정 (전역)
os.environ['TRANSFORMERS_TRUST_REMOTE_CODE'] = 'true'

logger = logging.getLogger(__name__)

# 임베딩 모델별 최대 시퀀스 길이 (토큰)
MODEL_MAX_LENGTH = {
    # Long Context (8192 토큰)
    'nomic-ai/nomic-embed-text-v1.5': 8192,
    'jinaai/jina-embeddings-v3': 8192,
    'Alibaba-NLP/gte-Qwen2-7B-instruct': 8192,

    # BERT 기반 (512 토큰) - 대부분의 모델
    'intfloat/multilingual-e5-large-instruct': 512,
    'mixedbread-ai/mxbai-embed-large-v1': 512,
    'BAAI/bge-m3': 512,
    'BAAI/bge-large-en-v1.5': 512,
    'BAAI/bge-base-en-v1.5': 512,
    'BAAI/bge-small-en-v1.5': 512,
    'sentence-transformers/all-mpnet-base-v2': 512,
    'sentence-transformers/all-MiniLM-L6-v2': 512,
    'thenlper/gte-base': 512,
    'intfloat/multilingual-e5-large': 512,
    'Alibaba-NLP/gte-large-en-v1.5': 512,
    'dunzhang/stella_en_1.5B_v5': 512,
}

def get_model_max_length(model_name: str) -> int:
    """
    모델별 최대 시퀀스 길이 반환

    Args:
        model_name: 임베딩 모델 이름

    Returns:
        int: 최대 시퀀스 길이 (토큰)
    """
    return MODEL_MAX_LENGTH.get(model_name, 512)  # 기본값 512


class CustomEmbeddings:
    """
    LangChain Chroma와 호환되는 커스텀 임베딩 클래스

    Note:
        LangChain의 임베딩 인터페이스를 구현하여 Chroma DB와 연동
    """
    def __init__(self, model, tokenizer, device, model_name: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name

    def embed_documents(self, texts):
        """문서 임베딩"""
        logger.info("---문서 임베딩---")
        return get_embeddings(texts, self.model, self.tokenizer, self.device,
                            model_name=self.model_name, is_query=False)

    def embed_query(self, text):
        """쿼리 임베딩"""
        logger.info("---쿼리 임베딩---")
        return get_embeddings([text], self.model, self.tokenizer, self.device,
                            model_name=self.model_name, is_query=True)[0]


def get_embeddings(texts: List[str], model, tokenizer, device,
                   model_name: str = None, is_query: bool = False) -> List[List[float]]:
    """
    텍스트를 벡터 임베딩으로 변환

    Args:
        texts: 임베딩할 텍스트 리스트
        model: 임베딩 모델
        tokenizer: 토크나이저
        device: 연산 디바이스 (CPU/GPU)
        model_name: 임베딩 모델 이름 (max_length 결정용)
        is_query: 쿼리인지 여부 (쿼리와 문서는 다른 prefix 사용)

    Returns:
        List[List[float]]: 정규화된 임베딩 벡터 리스트

    Note:
        쿼리와 문서를 구분하여 더 나은 검색 성능 달성
    """
    logger.info("---텍스트 임베딩 시작---")
    # 쿼리인 경우 'query:' prefix 추가 (검색 성능 향상)
    if is_query:
        texts = [f"query: {text}" for text in texts]

    # 모델별 최대 시퀀스 길이 결정
    max_length = get_model_max_length(model_name) if model_name else 512
    logger.info(f"---모델: {model_name}, max_length: {max_length}---")

    # 토큰화
    tokens = tokenizer(
        texts,
        padding=True,   # 배치 내에서 길이 통일
        truncation=True,    # 최대 길이 초과시 자르기
        return_tensors='pt',    # PyTorch 텐서로 반환
        max_length=max_length
    ).to(device)

    # 임베딩 생성 (그래디언트 계산 비활성화로 메모리 절약)
    with torch.no_grad():
        embeddings = model(**tokens)[0][:, 0]

    # L2 정규화 (코사인 유사도 계산을 위해)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    logger.info("---임베딩 변환 완료---")
    return embeddings.cpu().tolist()


def load_embedding_model(model_name: str, device: torch.device):
    """
    임베딩 모델 로딩

    Args:
        model_name: HuggingFace 모델 이름
        device: 로딩할 디바이스

    Returns:
        tuple: (모델, 토크나이저)
    """
    logger.info(f"임베딩 모델 로딩 시작... (device: {device})")

    embedding_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 일부 모델은 add_pooling_layer 파라미터를 지원하지 않음
    try:
        embedding_model = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            trust_remote_code=True
        )
    except TypeError:
        # add_pooling_layer를 지원하지 않는 모델의 경우 (예: BAAI/bge-m3)
        embedding_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    embedding_model.to(device).eval()  # 평가 모드로 설정

    logger.info("임베딩 모델 로딩 완료")
    return embedding_model, embedding_tokenizer
