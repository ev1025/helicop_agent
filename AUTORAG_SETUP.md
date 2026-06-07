# AutoRAG 본 평가 설정 정리 (2026-05-19)

## 프로젝트
- **수리온 헬리콥터 매뉴얼 RAG LLM 평가**
- AutoRAG 0.3.22
- 서버: 203.247.46.194, RTX 5090 32GB × 4 (Blackwell sm_120)
- 디렉토리: `/home/jinwoolee/surion/`

## 데이터 (AutoRAG 표준 준수)

### corpus.parquet (495 chunks)
| 컬럼 | 타입 | 비고 |
|---|---|---|
| doc_id | str | 고유 ID (c-*, p-*) |
| contents | str | 텍스트 본문 |
| metadata | dict | source/page/doc_type/parent_doc_id/**last_modified_datetime** |

### qa.parquet (79개 = single 55 + multi-hop 24)
| 컬럼 | 타입 | 비고 |
|---|---|---|
| qid | str | UUID |
| query | str | 질문 |
| retrieval_gt | 2D list | `[[c1]]` single, `[[c1],[c2]]` multi-hop (AND/OR) |
| generation_gt | list | 정답 |
| retrieval_gt_contents | 2D list | retrieval_gt의 contents 매핑 |

### QA 생성
- 생성기: **Qwen2.5-72B-Instruct-AWQ** (vLLM TP=2)
- AutoRAG 정식 pipeline:
  - `random_single_hop` 또는 `random_two_hop_same_source` (custom) sampling
  - `factoid_query_gen` / `two_hop_incremental_patched` (한국어 prompt 패치)
  - `make_basic_gen_gt` (answer 생성)
  - `dontknow_filter_rule_based` (모르는 답변 필터)
  - `passage_dependency_filter_llama_index` (passage 의존성 필터)
- multi-hop AutoRAG ko prompt가 영어 example이라서 → `KO_MULTIHOP_MESSAGES` 한국어 example 직접 작성

## 환경 분리 (envA, envB)

| env | venv | transformers | vLLM | 대상 모델 | GPU |
|---|---|---|---|---|---|
| **envA** | `.venv` | 4.57.6 | 0.20.1 | Qwen/Llama/Mistral/Phi/Bllossom/VARCO/Yi/gemma 작은 (22개) | vLLM=GPU2, eval=GPU3 |
| **envB** | `.venv_tf5` | 5.8.1 | 0.21.0 | Gemma 3/3n/27b + Gemma 4 시리즈 (6개) | vLLM=GPU2, eval=GPU3 (envA 끝나고) |

## 평가 모델 (27개, cyankiwi-31B 제외)

### envA (22)
Qwen3-32B-AWQ, Qwen2.5-32B-Instruct-AWQ, QwQ-32B-AWQ, Qwen2.5-14B-AWQ, Qwen3-14B-AWQ, Qwen3-8B, Qwen2.5-7B-AWQ, Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct, MLP-KTLim/Bllossom-8B, NCSOFT/VARCO-8B, Bllossom-3B, microsoft/Phi-3.5-mini-instruct (max_len 16384), mistralai/Mistral-7B-v0.3, mistralai/Mistral-Nemo-Instruct-2407 (top_k 3), HuggingFaceH4/zephyr-7b-beta, 01-ai/Yi-1.5-9B-Chat (top_k 3), 01-ai/Yi-1.5-6B-Chat (top_k 3), google/gemma-3-1b, gemma-3-4b, gemma-3-270m, RedHatAI/gemma-3-27b-it-quantized.w4a16

### envB (6)
google/gemma-3-12b, google/gemma-3n-E2B, google/gemma-4-E2B, cyankiwi/gemma-4-E4B-AWQ-INT4, cyankiwi/gemma-4-26B-A4B-AWQ-4bit, RedHatAI/gemma-3-27b (재평가)

### 제외
- **cyankiwi/gemma-4-31B-AWQ-4bit** — RTX 5090 sm_120 + vLLM 0.21 NCCL P2P 미지원 + TP=2 시 c10::Error sendBytes
  - 관련 이슈: [vLLM #21491](https://github.com/vllm-project/vllm/issues/21491), [#14628](https://github.com/vllm-project/vllm/issues/14628), [#33041](https://github.com/vllm-project/vllm/issues/33041)
  - 우회: Docker NCCL 2.26.5+ 필요 (현재 환경엔 미적용)

## AutoRAG 본 평가 config (config.server.api.prod.yaml)

```yaml
vectordb:
  - name: chroma_arctic_ko
    embedding_model: dragonkue/snowflake-arctic-embed-l-v2.0-ko
  - name: chroma_bge_m3
    embedding_model: BAAI/bge-m3
  - name: chroma_koe5
    embedding_model: nlpai-lab/KoE5
  - name: chroma_kure_v1
    embedding_model: nlpai-lab/KURE-v1

node_lines:
  retrieve_line:
    - lexical_retrieval: bm25 (ko_kiwi)
    - semantic_retrieval: 4 vectordb (best 자동 선정)
    - hybrid_retrieval: hybrid_rrf + hybrid_cc (normalize [mm, tmm], weight_range [[0.0, 1.0]], test_weight_size 21)
    - passage_reranker (top_k 3):
        - flag_embedding_reranker: BAAI/bge-reranker-v2-m3
        - flag_embedding_reranker: dragonkue/bge-reranker-v2-m3-ko
        - sentence_transformer_reranker: bongsoo/kpf-cross-encoder-v1
  prompt_line:
    - prompt_maker: fstring (한국어 질문/답변 템플릿)
  generate_line:
    - generator: llama_index_llm + openailike
        - api_base: http://127.0.0.1:8000/v1 (vLLM API server)
        - api_key: EMPTY
        - max_tokens: 256, temperature: 0.1, top_p: 0.95
```

## Generator 모듈 선택 분석

AutoRAG의 4가지 generator module:
| Module | 방식 | GPU 배치 | 결과 |
|---|---|---|---|
| `vllm` | Python `vllm.LLM` 직접 | autorag + vLLM 같은 GPU | 메모리 경합 → fail |
| `vllm_api` | vLLM API server HTTP | 분리 가능 | 검증 미진행 |
| **`llama_index_llm` + openailike** | LlamaIndex가 HTTP 호출 | **분리 (vLLM=GPU2, eval=GPU3)** | **채택** ✅ |
| `openai_llm` | OpenAI 직접 | - | 미사용 |

**선택 이유**: GPU 분리로 메모리 경합 회피 + 어제 PASS 검증.

## 핵심 trial-and-error 발견

### GPU 메모리 한계
- nvidia-smi free 32 GB ≠ vLLM 측정 free
- **vLLM startup 가용 = ~25.99 GB** (CUDA driver + PyTorch context로 ~6 GB overhead)
- gpu_util 안전 한계 = **0.70~0.80**

### 모델별 특이사항
- **Yi-1.5-9B/6B, Mistral-Nemo**: native context 4096 → `top_k 3`으로 prompt 단축 필요
- **Phi-3.5-mini**: native 128K → `max_model_len 16384` 명시
- **gemma-3-12b, gemma-3n, gemma-4 시리즈**: envA의 vLLM 0.20.1 미지원 → envB(0.21)로 평가
- **cyankiwi-31B**: 32GB GPU 단독 부족 + TP=2 불가 → 제외

### ChromaDB 제약
- collection name: 3-512자, `[a-zA-Z0-9._-]`, 시작/끝 알파벳/숫자
- `__CHROMA_SUFFIX__` 같은 placeholder가 wrapper sed로 교체 안 되면 검증 실패 → 제거함

### vLLM 호환
- vLLM 0.21 + Gemma 4: AutoRAG `vllm` module의 silent crash (수동 curl은 OK, AutoRAG의 호출 패턴이 trigger)
- → `llama_index_llm` (HTTP API)로 회피

### AutoRAG 미설치 env 처리
- `.venv_tf5`에 AutoRAG + deps 추가 설치 (`pip install AutoRAG[ko] llama-index-embeddings-huggingface bert_score`)

## 자동화 (auto_prod.sh, nohup 백그라운드)

```
STEP 1: envA 22 모델 평가 (run_server_multi.sh, vLLM=GPU2, eval=GPU3, .venv)
STEP 2: vLLM 정리 (pkill + nvidia-smi kill, sleep 15)
STEP 3: envB 6 모델 평가 (.venv_tf5)
STEP 4: vLLM 정리
STEP 5: Qwen2.5-72B-AWQ vLLM 시작 (TP=2, GPU 2+3, gpu_util 0.92, max_len 16384)
STEP 6: 72B ready 대기 (max 10분)
STEP 7: judge_all.py 실행 (모든 모델 결과 모아서 72B로 평가)
STEP 8: 모든 사용자 vLLM/wrapper/run_autorag KILL (다른 사용자 GPU 사용 가능)
STEP 9: 최종 GPU 상태 출력
```

**컴퓨터 꺼도 진행** (nohup, 서버 자체 백그라운드).

## 주요 wrapper 파일

| 파일 | 역할 |
|---|---|
| `/home/jinwoolee/surion/auto_prod.sh` | 본 평가 + judge + KILL 통합 자동화 |
| `/home/jinwoolee/surion/autorag/run_server_multi.sh` | 모델별 vLLM API server 띄우고 AutoRAG 호출 |
| `/home/jinwoolee/surion/run_autorag_vllm.sh` | `vllm` module 방식 (사용 X, 메모리 경합) |
| `/home/jinwoolee/surion/autorag/judge_all.py` | 72B-as-Judge 스크립트 |

## 결과 위치 (내일 SSH 접속 확인)

| 파일 | 내용 |
|---|---|
| `auto_prod.log` | 전체 흐름 (STEP 1-9) |
| `run_summary_prodApi_A.log` | envA 모델별 OK/FAIL/SKIP |
| `run_summary_prodApi_B.log` | envB 모델별 OK/FAIL/SKIP |
| `autorag/PROD_RESULT_FINAL.md` | 모델별 점수 표 (AutoRAG 자동 생성) |
| `autorag/judge_scores.csv` | judge raw (모든 QA × 모델 × faithfulness/relevance/naturalness) |
| **`autorag/judge_avg.csv`** | **모델별 평균 점수 (최종 순위) — 가장 중요** |

## 현재 진행 (2026-05-19 16:42)
- ✅ auto_prod.sh v3 실행 중
- ✅ envA 2/22 PASS (Qwen3-32B-AWQ 488s, Qwen2.5-32B-AWQ 498s)
- ⏳ envA 진행 중 (3/22 QwQ-32B-AWQ ...)
- 예상 총: envA 3-4h + envB 1h + 72B judge 1-2h = **약 6-7시간 후 완료**
