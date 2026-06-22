import torch
import json
import time
import logging
import threading
from typing import Optional
from transformers import TextIteratorStreamer
from app.config import config
from app.core import models
from app.utils import text_utils as utils

logger = logging.getLogger(__name__)

# 이전 함수명: llm_streamer_with_rag_and_tts_v3
# def llm_streamer_with_rag_and_tts(
#     prompt: str, 
#     use_rag: bool = False, 
#     rag_context: str = "",
#     client_tts: bool = True
# ):
#     """
#     LLM 스트리밍 + RAG + TTS 통합 처리 메인 함수
    
#     Args:
#         prompt: 사용자 질문
#         use_rag: RAG 사용 여부
#         rag_context: RAG로 검색된 문서 컨텍스트
#         tts_worker: TTS 작업자 (현재 비활성화)
#         client_tts: 클라이언트 TTS 사용 여부
    
#     Yields:
#         str: Server-Sent Events 형식의 JSON 데이터 스트림
    
#     Note:
#         실시간 텍스트 생성과 동시에 TTS 데이터를 클라이언트로 전송
#         문장 단위로 TTS 처리하여 자연스러운 음성 출력
#     """
#     logger.info(' --- LLM 스트리밍 함수 동작 ---')
#     function_start = time.perf_counter()

#     try:
#         # [0] 모델 초기화
#         if not models.model or not models.tokenizer:
#             raise RuntimeError("모델이 초기화되지 않았습니다.")
        

#         # [1] 프롬프트 구성
#         prompt_start = time.perf_counter()
#         if use_rag and rag_context:    # RAG 기반 답변
#             if len(rag_context) > 1500:     # RAG 컨텍스트 길이 제한 (토큰 수 관리)
#                 cut = rag_context[:1500]
#                 last_period = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
                
#                 # 문장 경계로 안전 절단 시도
#                 if last_period > 800:
#                     cut = cut[:last_period+1]
#                 rag_context = cut + "..."
            
#             instruction = config.RAG_INSTRUCTION_TEMPLATE.format(
#                 rag_context=rag_context,
#                 prompt=prompt
#             )
#             logger.info("RAG 기반 프롬프트 구성 완료")
#         else:   # 일반 지식 기반
#             instruction = config.GENERAL_INSTRUCTION_TEMPLATE.format(prompt=prompt)
#             logger.info("일반 지식 기반 프롬프트 구성 완료")
        
#         messages = [
#             {"role": "system", "content": config.SYSTEM_PROMPT},
#             {"role": "user", "content": instruction}
#         ]
        
#         prompt_elapsed = (time.perf_counter() - prompt_start) * 1000
#         logger.info(f"프롬프트 구성 수행시간: {prompt_elapsed:.1f}ms")


#         # [2] 토큰화
#         tokenize_start = time.perf_counter()
#         try:
#             input_ids = models.tokenizer.apply_chat_template(
#                 messages, 
#                 add_generation_prompt=True,     # 어시스턴트 응답을 위한 프롬프트 추가
#                 return_tensors="pt"
#             ).to(models.model.device)
#             logger.info("채팅 템플릿 적용 성공")
#         except Exception as e:       # 채팅 템플릿 실패시 폴백 처리
#             logger.warning(f"채팅 템플릿 실패, 폴백 처리: {e}")
#             fallback_text = f"{config.SYSTEM_PROMPT}\n\n{instruction}"
#             input_ids = models.tokenizer.encode(fallback_text, return_tensors="pt").to(models.model.device)

#         input_length = input_ids.shape[1]
#         tokenize_elapsed = (time.perf_counter() - tokenize_start) * 1000
#         logger.info(f"--> 토큰화 수행시간: {tokenize_elapsed:.1f}ms, 입력 토큰 길이: {input_length}")
        
        
#         # 입력 길이 제한 (메모리 및 성능 고려)
#         if input_length > config.MAX_INPUT_TOKENS:
#             logger.warning(f"입력이 너무 깁니다 ({input_length} 토큰). RAG 없이 답변합니다.")
#             simple_instruction = f"질문: {prompt}\n\n간단하고 명확하게 답변해주세요."
#             input_ids = models.tokenizer.encode(simple_instruction, return_tensors="pt").to(models.model.device)
        
#         attention_mask = torch.ones_like(input_ids)     # Attention mask 생성
#         pad_token_id = models.tokenizer.pad_token_id if models.tokenizer.pad_token_id else models.tokenizer.eos_token_id    # 패딩 토큰 설정
        
#         # 종료 토큰 설정 (모델별 상이)
#         terminators = []
#         if models.tokenizer.eos_token_id:
#             terminators.append(models.tokenizer.eos_token_id)
        
#         try:    # LLaMA 모델의 특수 종료 토큰 처리
#             eot_id = models.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#             if eot_id != models.tokenizer.unk_token_id and eot_id not in terminators:
#                 terminators.append(eot_id)
#         except Exception:
#             pass
        
#         # 종료 토큰이 없으면 기본 EOS 토큰 사용
#         if not terminators:
#             terminators = [models.tokenizer.eos_token_id]
        
#         eos_token_id = terminators[0] if len(terminators) == 1 else terminators
#         logger.info(f"종료 토큰 아이디: {eos_token_id}")
        
        
#         # [3] 스트리밍 생성
#         streamer = TextIteratorStreamer(
#             models.tokenizer, 
#             timeout=60.0,       # 타임아웃 증가
#             skip_prompt=True,   # 입력 프롬프트 제외하고 생성된 텍스트만
#             skip_special_tokens=True    # 특수 토큰 제외
#         )
#         logger.info("스트리머 초기화 완료")

#         # 텍스트 생성 파라미터 설정
#         generation_kwargs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "pad_token_id": pad_token_id,
#             "max_new_tokens": config.MAX_NEW_TOKENS,
#             "min_new_tokens": config.MIN_NEW_TOKENS,
#             "eos_token_id": eos_token_id,           # 종료 토큰
#             "temperature": config.TEMPERATURE,      # 창의성 (낮을수록 일관됨)
#             "top_p": config.TOP_P,                  # Nucleus sampling 축소
#             "do_sample": True,                      # 샘플링 활성화
#             "repetition_penalty": config.REPETITION_PENALTY,    # 반복 억제
#             "streamer": streamer,                   # 실시간 스트리밍
#         }

#         logger.info(f"생성 파라미터: max_tokens={config.MAX_NEW_TOKENS}, temp={config.TEMPERATURE}, top_p={config.TOP_P}")


#         # [4] 별도 스레드에서 생성 (메인 스레드 블로킹 방지)
#         generation_start = time.perf_counter()
#         generation_thread = threading.Thread(target=models.model.generate, kwargs=generation_kwargs)
#         generation_thread.daemon = True     # 메인 프로그램 종료시 함께 종료
#         generation_thread.start()
#         logger.info("생성 스레드 시작")

#         # 스트리밍 처리 변수
#         sentence_buffer = ""    # 문장 구성용 버퍼 
#         generated_text = ""     # 전체 생성된 텍스트
#         token_count = 0         # 생성된 토큰 수
#         tts_queue = []          # TTS 처리할 문장들
#         tts_count = 0           # TTS 생성 카운트
#         first_token_time = None # 첫 토큰 생성 시간 (TTFT)        
        

#         # [5] 실시간 토큰 수신 및 전송
#         stream_start = time.perf_counter()
#         try:
#             for new_text in streamer:
#                 if new_text:

#                     # 첫 토큰 생성 시간 측정 (Time To First Token)
#                     if first_token_time is None:
#                         first_token_time = time.perf_counter()
#                         ttft = (first_token_time - generation_start) * 1000
#                         logger.info(f"--> 첫 토큰 생성 시간(TTFT): {ttft:.1f}ms")

#                     # 생성된 텍스트 누적
#                     generated_text += new_text
#                     token_count += 1
#                     sentence_buffer += new_text
                    
#                     # 클라이언트에 텍스트 즉시 전송
#                     yield f"data: {json.dumps({'type': 'text', 'content': new_text, 'lang': 'ko'}, ensure_ascii=False)}\n\n"
                    
#                     # 완전한 문장으로 완성되면 TTS 처리
#                     if utils.is_complete_sentence(sentence_buffer):
#                         if utils.is_tts_worthy_text(sentence_buffer):
#                             clean_sentence = utils.clean_tts_text(sentence_buffer)
#                             if clean_sentence and len(clean_sentence.strip()) > 3:
#                                 tts_queue.append(clean_sentence)
#                                 tts_count += 1
                                
#                                 # 클라이언트에 TTS 데이터 전송
#                                 tts_data = {
#                                     'type': 'tts', 
#                                     'content': clean_sentence,
#                                     'lang': 'ko-KR',    # 한국어
#                                     'rate': 1.0,        # 말하기 속도
#                                     'pitch': 1.0,       # 음 높이
#                                     'volume': 0.8       # 볼륨
#                                 }
#                                 yield f"data: {json.dumps(tts_data, ensure_ascii=False)}\n\n"
#                                 logger.debug(f"TTS 전송 #{tts_count}: {len(clean_sentence)}자")
#                         sentence_buffer = ""      # 버퍼 초기화
#                     time.sleep(0.005)     # 과부하 방지를 위한 짧은 지연
        
#         except StopIteration:
#             logger.info("스트리밍 정상 완료")
#         except Exception as e:
#             logger.error(f"스트리밍 중 오류 발생: {e}")
#             raise

#         stream_elapsed = (time.perf_counter() - stream_start) * 1000
#         logger.info(f"--> 스트리밍 수행시간: {stream_elapsed:.1f}ms")
        

#         # 생성 스레드 종료 대기 (최대 30초)
#         generation_thread.join(timeout=30)

#         if generation_thread.is_alive():
#             logger.warning("생성 스레드가 30초 내에 종료되지 않음")
#         else:
#             generation_elapsed = (time.perf_counter() - generation_start) * 1000
#             logger.info(f"텍스트 생성 총 시간: {generation_elapsed:.1f}ms")

#         # 마지막 남은 텍스트 TTS 처리
#         if sentence_buffer.strip():
#             logger.info(f"마지막 버퍼 처리: {len(sentence_buffer)}자")
#             if utils.is_tts_worthy_text(sentence_buffer):
#                 clean_sentence = utils.clean_tts_text(sentence_buffer)
#                 if clean_sentence:
#                     tts_count += 1
#                     tts_data = {
#                         'type': 'tts', 
#                         'content': clean_sentence, 
#                         'lang': 'ko-KR', 
#                         'rate': 1.0, 
#                         'pitch': 1.0, 
#                         'volume': 0.8
#                     }
#                     yield f"data: {json.dumps(tts_data, ensure_ascii=False)}\n\n"
#                     logger.debug(f"TTS 전송 #{tts_count} (마지막): {len(clean_sentence)}자")
        
#         # 스트리밍 완료 신호
#         yield f"data: {json.dumps({'type': 'tts_complete', 'total_items': len(tts_queue)}, ensure_ascii=False)}\n\n"
#         yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        
#         function_elapsed = (time.perf_counter() - function_start) * 1000
        
#         logger.info(f"=== LLM 스트리밍 완료 ===")
#         logger.info(f"--> 생성 텍스트 길이: {len(generated_text)}자, 토큰 수: {token_count}")
#         logger.info(f"--> TTS 문장 수: {tts_count}개")
#         logger.info(f"전체 함수 수행시간: {function_elapsed:.1f}ms")

#         if token_count > 0 and first_token_time:
#             tokens_per_sec = token_count / ((time.perf_counter() - first_token_time))
#             logger.info(f"--> 평균 생성 속도: {tokens_per_sec:.2f} 토큰/초")
        
#     except Exception as e:
#         logger.error(f"LLM 스트리밍 오류: {e}", exc_info=True)
#         yield f"data: {json.dumps({'type': 'error', 'content': f'답변 생성 실패: {str(e)}'}, ensure_ascii=False)}\n\n"

# 기능별로 쪼갬 (프롬프트+토큰화+생성기+스트리밍+로그)

# [1] 프롬프트 생성
def build_prompt(prompt: str, rag_context: str, use_rag: bool) -> str:
    """
    사용자 입력을 기반으로 LLM 프롬프트 문자열을 생성합니다.

    RAG 문서 컨텍스트가 주어지고 사용 설정이 True인 경우,
    해당 컨텍스트를 포함한 프롬프트를 구성합니다.
    그렇지 않으면 일반 지식 기반 질문으로 처리됩니다.

    Args:
        prompt (str): 사용자 질문
        rag_context (str): 벡터 검색을 통해 얻은 문서 컨텍스트
        use_rag (bool): RAG 사용 여부

    Returns:
        str: 모델 입력용 완성된 프롬프트 텍스트
    """
    if use_rag and rag_context:
        cut = rag_context[:config.RAG_CONTEXT_MAX_LENGTH]
        last_period = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
        if last_period > config.RAG_CONTEXT_SENTENCE_BOUNDARY_MIN:
            cut = cut[:last_period+1]
        rag_context = cut + "..."
        return config.RAG_INSTRUCTION_TEMPLATE.format(rag_context=rag_context, prompt=prompt)
    return config.GENERAL_INSTRUCTION_TEMPLATE.format(prompt=prompt)

# [2] 토큰화
def tokenize_input(messages: list[dict], tokenizer, model, fallback_text=None):
    """
    대화 메시지를 모델 입력에 맞게 토큰화합니다.

    채팅 템플릿 적용을 시도하고, 실패할 경우 일반 텍스트로 폴백합니다.
    출력 텐서는 자동으로 모델 디바이스로 이동됩니다.

    Args:
        messages (list[dict]): 시스템/사용자 역할 기반 메시지 리스트
        tokenizer: 모델에 연결된 토크나이저 객체
        model: LLM 모델 객체 (디바이스 확인용)
        fallback_text (str, optional): 템플릿 실패 시 사용할 기본 프롬프트

    Returns:
        torch.Tensor: 모델 입력용 토큰 시퀀스
    """
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    except Exception:
        fallback = fallback_text or messages[-1]["content"]
        return tokenizer.encode(fallback, return_tensors="pt").to(model.device)

# [3] 생성기 설정
def prepare_generation(input_ids, tokenizer, model):
    """
    LLM 생성 함수에 전달할 매개변수를 구성합니다.

    스트리밍 출력을 위한 TextIteratorStreamer와 함께 생성 파라미터를 설정합니다.

    Args:
        input_ids (torch.Tensor): 토크나이즈된 입력 시퀀스
        tokenizer: 모델의 토크나이저
        model: LLM 모델 객체

    Returns:
        tuple: (생성 파라미터 딕셔너리, 스트리머 객체)
    """
    attention_mask = torch.ones_like(input_ids)
    eos_token = tokenizer.eos_token_id or tokenizer.pad_token_id
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_args = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "min_new_tokens": config.MIN_NEW_TOKENS,
        "eos_token_id": eos_token,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "do_sample": True,
        "repetition_penalty": config.REPETITION_PENALTY,
        "streamer": streamer,
    }
    return gen_args, streamer

# [4] 스레드 시작
def start_generation_thread(model, gen_args):
    """
    텍스트 생성을 별도 스레드에서 비동기 실행합니다.

    FastAPI 환경이나 기타 비동기 환경에서 메인 스레드 블로킹을 방지합니다.

    Args:
        model: LLM 모델 객체
        gen_args (dict): 모델.generate() 호출 시 사용할 파라미터 딕셔너리

    Returns:
        threading.Thread: 실행 중인 생성 스레드 객체
    """
    thread = threading.Thread(target=model.generate, kwargs=gen_args)
    thread.daemon = True
    thread.start()
    return thread

# [5] 스트리밍 출력
def stream_output(streamer, client_tts=True):
    """
    생성된 텍스트를 실시간으로 스트리밍합니다.

    텍스트가 문장 단위로 완성될 경우, TTS 처리 대상 문장으로 큐잉하며
    필요 시 클라이언트에 TTS용 JSON 데이터도 함께 전송합니다.

    Args:
        streamer (TextIteratorStreamer): 텍스트 스트리밍을 위한 객체
        client_tts (bool): TTS 활성화 여부 (기본값: True)

    Yields:
        str: Server-Sent Events(SSE) 형식의 데이터 스트링 (JSON 직렬화된 텍스트)
    """
    buffer = ""
    tts_count = 0
    tts_queue = []
    generated_text = ""
    token_count = 0
    first_token_time = None

    for new_text in streamer:
        if new_text:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated_text += new_text
            token_count += 1
            buffer += new_text

            yield f"data: {json.dumps({'type': 'text', 'content': new_text, 'lang': 'ko'}, ensure_ascii=False)}\n\n"

            if utils.is_complete_sentence(buffer):
                if utils.is_tts_worthy_text(buffer):
                    clean = utils.clean_tts_text(buffer)
                    if clean and len(clean.strip()) > 3:
                        tts_queue.append(clean)
                        tts_data = {
                            'type': 'tts',
                            'content': clean,
                            'lang': 'ko-KR',
                            'rate': 1.0,
                            'pitch': 1.0,
                            'volume': 0.8
                        }
                        yield f"data: {json.dumps(tts_data, ensure_ascii=False)}\n\n"
                buffer = ""
            time.sleep(0.005)

    return generated_text, token_count, tts_queue, tts_count, first_token_time

# 최종
def llm_streamer_with_rag_and_tts(prompt, use_rag=False, rag_context="", client_tts=True):
    """
    LLM 스트리밍 응답을 생성하는 메인 컨트롤 함수

    사용자 질문에 대해 RAG 문서를 포함한 프롬프트를 생성하고,
    실시간 토큰 생성을 스레드로 실행하여 스트리밍합니다.
    문장 단위로 TTS 응답을 포함하며, 클라이언트에 데이터 전송을 담당합니다.

    Args:
        prompt (str): 사용자 질문
        use_rag (bool): RAG 사용 여부
        rag_context (str): 벡터 검색을 통해 얻은 문서 컨텍스트
        client_tts (bool): 클라이언트 TTS 데이터 전송 여부

    Yields:
        str: Server-Sent Events(SSE) 형식의 응답 JSON 문자열

    Note:
        - 오류 발생 시 error 이벤트가 전송됩니다.
        - 스트리밍 완료 후 'done' 이벤트가 전송됩니다.
        - 실시간 음성 합성을 위한 TTS 전송이 포함될 수 있습니다.
    """
    logger.info(' --- LLM 스트리밍 함수 동작 ---')
    function_start = time.perf_counter()

    try:
        # [0] 모델 초기화
        if not models.model or not models.tokenizer:
            raise RuntimeError("모델 미초기화")

        # [1] 프롬프트 구성
        instruction = build_prompt(prompt, rag_context, use_rag)
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": instruction}]
        
        # [2] 토큰화
        tokenize_start = time.perf_counter()
        input_ids = tokenize_input(messages, models.tokenizer, models.model, fallback_text=instruction)
        tokenize_elapsed = (time.perf_counter() - tokenize_start) * 1000
        logger.info(f"--> 토큰화 수행시간: {tokenize_elapsed:.1f}ms")
        
        # [3] 스트리밍 생성
        gen_args, streamer = prepare_generation(input_ids, models.tokenizer, models.model)
        
        # [4] 별도 스레드에서 생성 (메인 스레드 블로킹 방지)
        stream_start = time.perf_counter()
        gen_thread = start_generation_thread(models.model, gen_args)
        logger.info("스트리밍 시작")

        for chunk in stream_output(streamer, client_tts):
            yield chunk

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        gen_thread.join(timeout=30)
        if gen_thread.is_alive():
            logger.warning("생성 스레드 미종료")
        stream_elapsed = (time.perf_counter() - stream_start) * 1000
        logger.info(f"--> 스트리밍 수행시간: {stream_elapsed:.1f}ms")

    except Exception as e:
        logger.error(f"에러: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': f'오류 발생: {str(e)}'}, ensure_ascii=False)}\n\n"


# async 내부는 await? 비동기로
def generate_chat_style_response(prompt: str, use_rag: bool, rag_context: str) -> str:
    """
    스트리밍이 안 되는 환경에서 LLM 응답을 재사용하려고 만든 단일 호출용 fallback 버전
    
    Args:
        prompt: 사용자 질문
        use_rag: RAG 사용 여부
        rag_context: RAG 검색 결과 컨텍스트
    
    Returns:
        str: 완성된 응답 텍스트
    
    Note:
        음성 채팅에서 JSON 응답을 위해 사용
        스트리밍과 동일한 로직이지만 전체 텍스트를 한 번에 반환
    """
    logger.info(' --- 제너럴 챗 응답 함수 동작 ---')
    try:
        if not models.model or not models.tokenizer:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        # [1] 프롬프트 구성
        instruction = build_prompt(prompt, rag_context, use_rag)
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ]

        # [2] 입력 토큰화
        input_ids = tokenize_input(messages, models.tokenizer, models.model, fallback_text=instruction)
        input_len = input_ids.shape[1]

        # [3] 생성 파라미터 구성 (스트리머 제외)
        gen_args, _ = prepare_generation(input_ids, models.tokenizer, models.model)
        gen_args.pop("streamer", None)  # 스트리머 제거

        # [4] 텍스트 생성
        with torch.no_grad():
            outputs = models.model.generate(**gen_args)

        # [5] 응답 후처리
        response_tokens = outputs[0][input_len:]
        response = models.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response.strip()

    except Exception as e:
        logger.error(f"[generate_chat_response_once] 응답 생성 실패: {e}", exc_info=True)
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"


# ==================== Phase 2: 로컬 LLM + 도구 통합 ====================

def generate_with_tools_local(
    user_message: str,
    tools: list[dict],
    conversation_history: Optional[str] = None,
    max_new_tokens: int = 1024
) -> tuple[Optional[str], list]:
    """
    로컬 LLM을 도구 스키마와 함께 호출하여 응답을 생성합니다.

    Phase 2에서 추가된 함수로, 로컬 LLM이 상황에 따라 도구를 선택할 수 있습니다.
    프롬프트 엔지니어링을 통해 LLM이 JSON 형식으로 도구 호출을 응답하도록 유도합니다.

    Args:
        user_message (str): 사용자 질문
        tools (list[dict]): 도구 스키마 리스트 (ToolExecutor.get_all_schemas() 형식)
        conversation_history (str, optional): 이전 대화 히스토리 (있으면 포함)
        max_new_tokens (int): 최대 생성 토큰 수 (기본값: 1024)

    Returns:
        (text_content, tool_calls) 튜플
        - text_content: LLM이 생성한 텍스트 (도구 호출 JSON 제외)
        - tool_calls: ToolCall 객체 리스트 (없으면 빈 리스트)

    Example:
        >>> from app.core.tool_executor import ToolExecutor
        >>> from app.core.tools import FinalAnswerTool, RAGSearchTool
        >>>
        >>> executor = ToolExecutor()
        >>> executor.register_tool(FinalAnswerTool())
        >>> executor.register_tool(RAGSearchTool())
        >>>
        >>> tools = executor.get_all_schemas()
        >>> text, tool_calls = generate_with_tools_local("양력이란?", tools)
        >>>
        >>> if tool_calls:
        >>>     # LLM이 도구를 선택함
        >>>     for tc in tool_calls:
        >>>         result = executor.execute(tc.name, **tc.arguments)
        >>> else:
        >>>     # LLM이 직접 답변함
        >>>     print(text)

    Note:
        - 로컬 Hugging Face 모델 사용
        - 도구 스키마를 프롬프트에 포함
        - LLM이 JSON 형식으로 도구 호출 응답
        - ```json {...} ``` 블록을 파싱하여 ToolCall 객체 생성
    """
    from app.core.tool_prompt import (
        build_tool_prompt,
        parse_tool_call_from_response,
        extract_text_without_tool_call
    )
    from app.core.conversation import ToolCall

    logger.info(f" --- 로컬 LLM + 도구 호출 함수 동작 ---")

    try:
        # [0] 모델 초기화 확인
        if not models.model or not models.tokenizer:
            raise RuntimeError("모델이 초기화되지 않았습니다.")

        # [1] 도구 프롬프트 생성
        tool_prompt = build_tool_prompt(tools) if tools else ""

        # [2] 전체 프롬프트 구성
        if conversation_history:
            # 이전 대화가 있으면 포함
            full_prompt = f"""{conversation_history}

{tool_prompt}

사용자: {user_message}

어시스턴트:"""
        else:
            # 첫 번째 질문
            full_prompt = f"""{tool_prompt}

사용자: {user_message}

어시스턴트:"""

        # [3] 메시지 구성
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt}
        ]

        logger.info(f"프롬프트 길이: {len(full_prompt)}자")

        # [4] 토큰화
        input_ids = tokenize_input(messages, models.tokenizer, models.model, fallback_text=full_prompt)

        # [5] 생성 파라미터 구성 (스트리머 제외)
        gen_args, _ = prepare_generation(input_ids, models.tokenizer, models.model)
        gen_args.pop("streamer", None)  # 스트리머 제거
        gen_args["max_new_tokens"] = max_new_tokens  # 파라미터 오버라이드

        # [6] 텍스트 생성
        logger.info("로컬 LLM 생성 시작")
        with torch.no_grad():
            outputs = models.model.generate(**gen_args)

        # [7] 응답 후처리
        input_len = input_ids.shape[1]
        response_tokens = outputs[0][input_len:]
        response = models.tokenizer.decode(response_tokens, skip_special_tokens=True)

        logger.info(f"LLM 응답 길이: {len(response)}자")

        # [8] 도구 호출 파싱
        tool_name, arguments = parse_tool_call_from_response(response)

        if tool_name:
            # 도구 호출이 있음
            tool_call = ToolCall(name=tool_name, arguments=arguments)
            tool_calls = [tool_call]
            logger.info(f"도구 호출 감지: {tool_name}({arguments})")
        else:
            # 도구 호출 없음 (직접 답변)
            tool_calls = []
            logger.info("도구 호출 없음 (직접 답변)")

        # [9] 텍스트 추출 (JSON 블록 제외)
        text_content = extract_text_without_tool_call(response)

        return text_content, tool_calls

    except Exception as e:
        logger.error(f"로컬 LLM + 도구 호출 실패: {e}", exc_info=True)
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}", []

