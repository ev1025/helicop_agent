# autoragtest — AutoRAG 27-모델 평가 산출물 (helicop_agent 하위)

이 폴더는 [`helicop_agent`](../) 프로젝트의 **공개용 평가 산출물 정리본**입니다. 운영 코드는 상위 [`../autorag/`](../autorag/)에 별도로 있고, 이 폴더는 외부 공유/보고 목적의 결과 + 스크립트 + 문서 모음입니다.

---

## 폴더 정체성

| 구분 | 이 폴더 (`autoragtest/`) | 상위 (`../autorag/`) |
|---|---|---|
| 역할 | 평가 산출물 공개 정리본 | LangGraph 에이전트(`agent_v2`) 운영 모듈 |
| 핵심 파일 | `results/*.csv`, `docs/*.md` | `adapter.py`, `run_autorag.py` |
| GitHub | https://github.com/ev1025/autoragtest | (비공개) |
| 데이터 포함 | ❌ 라이선스/보안 제외 | ✅ 운영 데이터 |

상위 `adapter.py`는 AutoRAG가 찾은 best_config(KURE-v1 + HybridCC + dragonkue-ko reranker)를 그대로 LangGraph `rag_search` 도구로 swap-in합니다 — 즉 이 폴더의 평가 결과가 곧장 실서비스 RAG 파이프라인이 됩니다.

---

## 평가 개요

- **AutoRAG** 0.3.22, 한국어 RAG 시나리오
- **27개 LLM 평가** (1개 제외 — TP=2 NCCL 미지원 GPU 호환성 이슈)
- **데이터**: 헬리콥터 정비 매뉴얼 (도메인 데이터), 495 chunks, 79 QA (single 55 + multi-hop 24)
- **임베딩 4종**: KURE-v1, arctic-embed-l-v2.0-ko, bge-m3, KoE5
- **Reranker 3종**: BAAI/bge-reranker-v2-m3, dragonkue/bge-reranker-v2-m3-ko, bongsoo/kpf-cross-encoder-v1
- **Hybrid**: BM25 + Vector (RRF / CC 자동 선정)
- **Judge**: `Qwen/Qwen2.5-72B-Instruct-AWQ` (faithfulness · relevance · naturalness, 0–10)

## 결과 Top 5

| Rank | 모델 | Overall (Judge) |
|---|---|---:|
| 🥇 | Qwen2.5-32B-Instruct-AWQ | **8.27** |
| 🥈 | Qwen2.5-14B-Instruct-AWQ | **8.25** |
| 🥉 | NCSOFT Llama-VARCO-8B-Instruct | **8.17** |
| 4 | Qwen QwQ-32B-AWQ | **7.87** |
| 5 | MLP-KTLim Bllossom-8B | **7.85** |

## 1위 파이프라인 (AutoRAG 자동 선정 best chain)

```
BM25 (ko_kiwi)                                    NDCG 0.605
  → KURE-v1 (semantic best, 4종 중 1위)           NDCG 0.769
  → HybridCC (normalize=mm, weight=0.70)          NDCG 0.791
  → dragonkue/bge-reranker-v2-m3-ko (3종 중 1위)  NDCG 0.835
  → Fstring 한국어 prompt
  → Qwen2.5-32B-Instruct-AWQ                      Judge 8.27
```

---

## 평가 환경

- GPU: RTX 5090 32GB × 4 (Blackwell sm_120)
- GPU 분리: vLLM과 evaluator를 별도 GPU에 할당 (메모리 경합 방지)
- 두 vLLM 환경 병행:
  - **envA**: vLLM 0.20.1, transformers 4.57.6 — 일반 LLM 22종
  - **envB**: vLLM 0.21.0, transformers 5.8.1 — Gemma 3 / 3n / 4 시리즈 (신규 아키텍처)
- Judge: TP=2, `max_model_len 16384`, `gpu_util 0.92`

---

## 작업 시 주의사항

### 1. 데이터 공개 금지
다음 파일은 도메인 매뉴얼 본문/일부를 포함하므로 **GitHub 등 외부에 절대 push 금지**:
- `corpus.parquet` — 매뉴얼 본문 그대로
- `qa.parquet` (모든 variant) — `retrieval_gt_contents`에 매뉴얼 일부 + `generation_gt`에 요약
- `benchmark_*/0/...` raw output — 모델 답변이 매뉴얼 기반

`.gitignore`에 위 패턴 차단되어 있음. 신규 파일 추가 시 데이터 노출 여부 확인.

### 2. 출처 표기 금지 (할루시네이션 방지)
README/문서에 다음 단어 명시 금지:
- 군용 헬기 사업명 (보안)
- "국토교통부 / 조종사 표준교재 / 공공누리 / KOGL" 등 출처 단정 (할루시네이션, 실제 출처 불확실)
- 표기 시 "헬리콥터 정비 매뉴얼 (도메인 데이터)"로 일반화

### 3. Git push 정책
- **GitHub** (`github.com`) push는 사용자 명시 요청 시 가능 — 이 폴더는 `ev1025/autoragtest`로 공개됨
- **사내 GitLab** push 절대 금지

---

## 핵심 파일 위치

| 파일 | 역할 |
|---|---|
| `README.md` | 산출물 중심 안내 (외부용) |
| `LICENSE` | MIT (코드) + 도메인 데이터 자체 라이선스 주석 |
| `docs/ANALYSIS_REPORT.md` | 5가지 분석 + 부록 A (4 임베딩 비교) |
| `docs/SETUP.md` | 환경/실행 가이드 |
| `docs/PROD_RESULT_FINAL.md` | AutoRAG 자동 생성 보고서 |
| `results/judge_avg.csv` | 27 모델 Judge 평균 (rank, F, R, N, overall) |
| `results/judge_scores.csv` | 모델 × qid raw (2,133행, 답변 텍스트 없음) |
| `results/metrics_combined.csv` | 27 × 9 metric 통합 표 (분석 핵심 입력) |
| `results/retrieval_embedding_comparison.csv` | 4 임베딩 NDCG/MRR/MAP |
| `scripts/auto_prod.sh` | 본 평가 자동화 (envA→envB→72B Judge) |
| `scripts/auto_retry.sh` | 실패 모델 재평가 (Yi 4K context, Nemo AWQ 교체 등) |
| `scripts/auto_retry_g12b.sh` | RedHatAI/gemma-3-12b-w4a16 개별 평가 |
| `scripts/run_server_multi.sh` | 모델별 vLLM 기동 + AutoRAG 호출 wrapper |
| `scripts/judge_all.py` | 72B Judge 스크립트 |
| `scripts/gen_qa_official.py` | single-hop QA 생성 |
| `scripts/gen_qa_multihop.py` | multi-hop QA 생성 (KO_MULTIHOP_MESSAGES 한국어 prompt 패치) |
| `configs/config.server.api.prod.yaml` | 본 평가 config (4 임베딩 + 3 reranker + Hybrid) |
| `configs/config.server.api.prod.shortgen.yaml` | max_tokens 200 (Yi-1.5 4K context 모델용) |
| `configs/valid_prod*.txt` | 모델 ID + 양자화/메모리 설정 (`model|quant|dtype|util|mml`) |

## 평가 자동화 흐름

```
auto_prod.sh (본 평가)
├── STEP 1: envA 22개 (vLLM 0.20.1)
├── STEP 2: vLLM cleanup (본인 process만)
├── STEP 3: envB 6개 (vLLM 0.21.0)
├── STEP 4: vLLM cleanup
├── STEP 5: 72B vLLM 시작 (TP=2)
├── STEP 6: Judge ready 대기 (~10분)
├── STEP 7: judge_all.py (전체 일관 score)
├── STEP 8: 본인 process 전체 KILL
└── STEP 9: 최종 GPU 상태

auto_retry.sh (실패 재평가)
└── envA retry 3 (max_tokens 200) + envB 5 + Judge

auto_retry_g12b.sh
└── RedHatAI/gemma-3-12b-w4a16 단일 + Judge
```

---

## 참고

- 평가 진행 일자: 2026-05-19 ~ 2026-05-26
- 운영 적용 코드: `../autorag/adapter.py` → `agent_v2.tools.rag_search`
- 상위 helicop_agent: LangGraph 멀티에이전트 + FastAPI + transformers/llama-cpp 백엔드
