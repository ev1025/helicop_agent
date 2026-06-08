# AutoRAG Multi-Model 평가 결과

평가 모델 수: **27**

## 1. 모델별 best 조합 + 메트릭 + 시간

| 모델명 | 조합 | 토크나이저 | reranker | NDCG | MRR | retrieval_map | sem_score | g_eval | faithfulness | gen latency(s/q) |
|---|---|---|---|---|---|---|---|---|---|---|
| 01-ai/Yi-1.5-6B-Chat | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.844 | - | - | 0.37 |
| 01-ai/Yi-1.5-9B-Chat | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.869 | - | - | 0.55 |
| Bllossom/llama-3.2-Korean-Bllossom-3B | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.863 | - | - | 0.22 |
| HuggingFaceH4/zephyr-7b-beta | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.861 | - | - | 0.42 |
| MLP-KTLim/llama-3-Korean-Bllossom-8B | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.873 | - | - | 0.32 |
| NCSOFT/Llama-VARCO-8B-Instruct | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.882 | - | - | 0.32 |
| Qwen/QwQ-32B-AWQ | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.857 | - | - | 1.25 |
| Qwen/Qwen2.5-1.5B-Instruct | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.825 | - | - | 0.17 |
| Qwen/Qwen2.5-14B-Instruct-AWQ | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.897 | - | - | 0.38 |
| Qwen/Qwen2.5-32B-Instruct-AWQ | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.896 | - | - | 1.23 |
| Qwen/Qwen2.5-3B-Instruct | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.902 | - | - | 0.24 |
| Qwen/Qwen2.5-7B-Instruct-AWQ | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.893 | - | - | 0.23 |
| Qwen/Qwen3-14B-AWQ | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.796 | - | - | 0.37 |
| Qwen/Qwen3-32B-AWQ | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.789 | - | - | 1.29 |
| Qwen/Qwen3-8B | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.861 | - | - | 0.35 |
| RedHatAI/gemma-3-12b-it-quantized.w4a16 | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.913 | - | - | 0.95 |
| RedHatAI/gemma-3-27b-it-quantized.w4a16 | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.919 | - | - | 1.05 |
| casperhansen/mistral-nemo-instruct-2407-awq | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.897 | - | - | 0.49 |
| cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.819 | - | - | 0.93 |
| cyankiwi/gemma-4-E4B-it-AWQ-INT4 | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.756 | - | - | 0.76 |
| google/gemma-3-1b-it | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.824 | - | - | 0.35 |
| google/gemma-3-270m-it | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.768 | - | - | 0.25 |
| google/gemma-3-4b-it | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.901 | - | - | 0.37 |
| google/gemma-3n-E2B-it | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.918 | - | - | 0.70 |
| google/gemma-4-E2B-it | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.721 | - | - | 0.60 |
| microsoft/Phi-3.5-mini-instruct | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.813 | - | - | 0.51 |
| mistralai/Mistral-7B-Instruct-v0.3 | Hybrid | ko_kiwi | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.884 | - | - | 0.36 |

## 2. 종합 점수 최고 순위 5위

종합점수 = sem_score / g_eval / faithfulness 평균 (있는 것만)

| 순위 | 모델 | 조합 | reranker | sem_score | g_eval | faithfulness | 종합 |
|---|---|---|---|---|---|---|---|
| 1 | RedHatAI/gemma-3-27b-it-quantized.w4a16 | Hybrid | FlagEmbeddingReranker | 0.919 | - | - | **0.919** |
| 2 | google/gemma-3n-E2B-it | Hybrid | FlagEmbeddingReranker | 0.918 | - | - | **0.918** |
| 3 | RedHatAI/gemma-3-12b-it-quantized.w4a16 | Hybrid | FlagEmbeddingReranker | 0.913 | - | - | **0.913** |
| 4 | Qwen/Qwen2.5-3B-Instruct | Hybrid | FlagEmbeddingReranker | 0.902 | - | - | **0.902** |
| 5 | google/gemma-3-4b-it | Hybrid | FlagEmbeddingReranker | 0.901 | - | - | **0.901** |

## 3. 노드별 상세 (모든 모델 평균)

retrieval/reranker 점수는 LLM 무관하게 동일한 corpus/QA 위에서 측정됨.

| 노드 | best 모듈 (1차 모델 기준) | NDCG | MRR | retrieval_map | exec_time(s) |
|---|---|---|---|---|---|
| lexical (BM25) | BM25 | 0.605 | 0.534 | 0.534 | 2.889 |
| semantic (vector) | VectorDB | 0.769 | 0.687 | 0.687 | 0.121 |
| hybrid | HybridCC | 0.791 | 0.703 | 0.703 | 0.086 |
| reranker | FlagEmbeddingReranker | 0.835 | 0.781 | 0.781 | 0.198 |

## 4. Raw 결과 위치

- `autorag/benchmark_01-ai_Yi-1.5-6B-Chat/0/`
- `autorag/benchmark_01-ai_Yi-1.5-9B-Chat/0/`
- `autorag/benchmark_Bllossom_llama-3.2-Korean-Bllossom-3B/0/`
- `autorag/benchmark_HuggingFaceH4_zephyr-7b-beta/0/`
- `autorag/benchmark_MLP-KTLim_llama-3-Korean-Bllossom-8B/0/`
- `autorag/benchmark_NCSOFT_Llama-VARCO-8B-Instruct/0/`
- `autorag/benchmark_Qwen_QwQ-32B-AWQ/0/`
- `autorag/benchmark_Qwen_Qwen2.5-1.5B-Instruct/0/`
- `autorag/benchmark_Qwen_Qwen2.5-14B-Instruct-AWQ/0/`
- `autorag/benchmark_Qwen_Qwen2.5-32B-Instruct-AWQ/0/`
- `autorag/benchmark_Qwen_Qwen2.5-3B-Instruct/0/`
- `autorag/benchmark_Qwen_Qwen2.5-7B-Instruct-AWQ/0/`
- `autorag/benchmark_Qwen_Qwen3-14B-AWQ/0/`
- `autorag/benchmark_Qwen_Qwen3-32B-AWQ/0/`
- `autorag/benchmark_Qwen_Qwen3-8B/0/`
- `autorag/benchmark_RedHatAI_gemma-3-12b-it-quantized.w4a16/0/`
- `autorag/benchmark_RedHatAI_gemma-3-27b-it-quantized.w4a16/0/`
- `autorag/benchmark_casperhansen_mistral-nemo-instruct-2407-awq/0/`
- `autorag/benchmark_cyankiwi_gemma-4-26B-A4B-it-AWQ-4bit/0/`
- `autorag/benchmark_cyankiwi_gemma-4-E4B-it-AWQ-INT4/0/`
- `autorag/benchmark_google_gemma-3-1b-it/0/`
- `autorag/benchmark_google_gemma-3-270m-it/0/`
- `autorag/benchmark_google_gemma-3-4b-it/0/`
- `autorag/benchmark_google_gemma-3n-E2B-it/0/`
- `autorag/benchmark_google_gemma-4-E2B-it/0/`
- `autorag/benchmark_microsoft_Phi-3.5-mini-instruct/0/`
- `autorag/benchmark_mistralai_Mistral-7B-Instruct-v0.3/0/`
