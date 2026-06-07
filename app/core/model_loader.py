"""
모델 로딩 관련 기능 모듈

이 모듈은 LLM, 임베딩, 리랭크 모델을 로딩하는 기능을 제공합니다.
"""

import torch
import logging
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


def get_device(use_gpu: bool) -> torch.device:
    """
    GPU 사용 여부에 따른 디바이스 결정

    Args:
        use_gpu: GPU 사용 여부

    Returns:
        torch.device: 사용할 디바이스 (cuda:0, mps, 또는 cpu)

    Note:
        GPU를 사용하려 해도 CUDA/MPS가 사용불가하면 CPU로 fallback
        CUDA 사용 시 단일 GPU(cuda:0)만 사용하도록 명시적으로 지정
    """
    logger.info("---디바이스 선택---")
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda:0')
    elif use_gpu and torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_llm_model(model_name: str):
    """
    LLaMA-3 모델 로딩

    Args:
        model_name: HuggingFace 모델 이름

    Returns:
        tuple: (모델, 토크나이저)
    """
    logger.info("LLaMA-3 모델 로딩 시작...")

    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CUDA 가용 여부에 따라 로딩 전략 분기
    if torch.cuda.is_available():
        # GPU에서 4비트 양자화 설정
        # qcfg = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     quantization_config=qcfg,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        # 단일 GPU(cuda:0)만 사용하도록 명시적으로 지정
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        torch.set_grad_enabled(False)
    elif torch.mps.is_available():
        # MPS (Apple Silicon) - float16 사용으로 메모리 50% 절감
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": "mps"},
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        torch.set_grad_enabled(False)
    else:
        # CPU 폴백
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        torch.set_grad_enabled(False)

    model.eval()
    logger.info("LLaMA-3 모델 로딩 완료")
    return model, tokenizer


def load_reranker_model(model_name: str, device: torch.device):
    """
    리랭커 모델 로딩

    Args:
        model_name: HuggingFace 모델 이름
        device: 로딩할 디바이스

    Returns:
        tuple: (모델, 토크나이저)
    """
    logger.info(f"Reranker 모델 로딩 시작... (device: {device})")

    reranker_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    reranker_model.to(device).eval()  # 평가 모드로 설정

    logger.info("Reranker 모델 로딩 완료")
    return reranker_model, reranker_tokenizer


def cleanup_models():
    """
    모델 정리 및 리소스 해제

    Note:
        프로그램 종료시 GPU 메모리 정리
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPU 메모리 캐시 정리
    elif torch.mps.is_available():
        torch.mps.empty_cache()
    logger.info("모델 정리 완료")
