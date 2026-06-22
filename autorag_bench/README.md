# AutoRAG 27-모델 한국어 RAG 평가 — 산출물 모음

[AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) 0.3.22 기반으로 한국어 RAG 시나리오에서 **27개 LLM × 4 임베딩 × 3 reranker × Hybrid retrieval**을 비교 평가했습니다. 본 저장소는 평가 산출물(점수 CSV + 분석 보고서)을 중심으로 정리되어 있습니다.

- 데이터: 헬리콥터 정비 매뉴얼 (도메인 데이터) — 495 chunks, 79 QA (single 55 + multi-hop 24)
- Judge: `Qwen/Qwen2.5-72B-Instruct-AWQ` (faithfulness · relevance · naturalness, 0–10)
- 서버: RTX 5090 32GB × 4 + vLLM 0.20.1 / 0.21.0

---

## 📊 산출물 (results/)

### 1. `judge_avg.csv` — 27 모델 최종 순위 ⭐
| 컬럼 | 설명 |
|---|---|
| `model` | 모델 HF ID |
| `faithfulness` `relevance` `naturalness` | 72B Judge 점수 (각 0–10) |
| `overall` | 위 3개 평균 |

```bash
# 활용 예: overall 내림차순 Top 10
sort -t, -k5 -nr results/judge_avg.csv | head
```

### 2. `judge_scores.csv` — Judge raw (2,133행) 🔹 답변 텍스트 없음(안전)
| 컬럼 | 설명 |
|---|---|
| `model` | 모델 HF ID |
| `qid` | QA UUID |
| `faithfulness` `relevance` `naturalness` | 단일 QA 점수 (0–10) |

→ 모델별 분산 측정, 어려운 QA 식별, 표준편차 산출 등에 사용.

### 3. `metrics_combined.csv` — **분석 핵심 입력** (27 × 9 통합)
| 컬럼 그룹 | 컬럼 | 의미 |
|---|---|---|
| 메타 | `rank`, `model` | Judge overall 순위 |
| **어휘 metric** | `bleu`, `rouge`, `meteor`, `bert_score` | GT vs 답변 n-gram·BERT 유사도 |
| **의미 metric** | `sem_score` | bge-m3 임베딩 cosine similarity |
| **LLM Judge** | `faithfulness`, `relevance`, `naturalness`, `overall` | 72B 정성평가 |

→ 어휘 vs 의미 vs Judge 상관 분석, 패밀리별 평균, 모델 크기 회귀 등 모든 분석의 단일 진실의 원천(SSoT).

### 4. `retrieval_embedding_comparison.csv` — 4 임베딩 비교
| 컬럼 | 설명 |
|---|---|
| `rank`, `embedding_model` | NDCG 순위 |
| `ndcg` `mrr` `map` `f1` `recall` `precision` | retrieval metric (top_k=10) |
| `exec_time_sec` | 평균 실행 시간 |
| `is_best` | AutoRAG 자동 선정 best 여부 (True/False) |

→ 임베딩 선택 의사결정의 객관적 근거.

---

## 📑 문서 (docs/)

| 파일 | 내용 |
|---|---|
| **`ANALYSIS_REPORT.md`** | 5가지 분석 (Pearson 상관·패밀리별 평균·Qwen2.5 vs Qwen3·크기 vs Judge·어휘 vs Judge 격차) + **부록 A: 4 임베딩 비교** |
| **`SETUP.md`** | 환경 셋업 (envA/envB venv, vLLM 두 버전, GPU 분리, QA 생성 파이프라인) + 실패 패턴 (Yi-1.5 4K context, Mistral-Nemo BF16 OOM, gemma-3-12b OOM 등) |
| **`PROD_RESULT_FINAL.md`** | AutoRAG 자동 생성 보고서 (모델별 best 조합 + sem_score + 노드별 평균) |

---

## 🏆 결과 하이라이트

### Judge Overall Top 5
| Rank | 모델 | F | R | N | **Overall** |
|---|---|---:|---:|---:|---:|
| 🥇 | Qwen2.5-32B-Instruct-AWQ | 7.58 | 8.42 | 8.81 | **8.27** |
| 🥈 | Qwen2.5-14B-Instruct-AWQ | 7.58 | 8.42 | 8.75 | **8.25** |
| 🥉 | NCSOFT Llama-VARCO-8B-Instruct | 7.49 | 8.24 | 8.79 | **8.17** |
| 4 | Qwen QwQ-32B-AWQ | 7.35 | 8.01 | 8.24 | **7.87** |
| 5 | MLP-KTLim llama-3-Korean-Bllossom-8B | 7.17 | 7.90 | 8.48 | **7.85** |

### 임베딩 1위
| Rank | 임베딩 모델 | NDCG | best |
|---|---|---:|:---:|
| 🥇 | **nlpai-lab/KURE-v1** | **0.7693** | ✅ |
| 🥈 | dragonkue/snowflake-arctic-embed-l-v2.0-ko | 0.7648 | ❌ |
| 🥉 | BAAI/bge-m3 | 0.7557 | ❌ |
| 4 | nlpai-lab/KoE5 | 0.6738 | ❌ |

### AutoRAG 1위 파이프라인 (자동 선정 best chain)
```
BM25 (ko_kiwi)                                    NDCG 0.605
  → KURE-v1 (semantic, 4종 중 1위)                NDCG 0.769
  → HybridCC (normalize=mm, weight=0.70)          NDCG 0.791
  → dragonkue/bge-reranker-v2-m3-ko (3종 중 1위)  NDCG 0.835
  → Fstring 한국어 prompt
  → Qwen2.5-32B-Instruct-AWQ                      Judge 8.27
```

### 핵심 인사이트
- **한국어 특화 8B (VARCO/Bllossom)가 32B 일반 모델 추월** — 한국어 학습 데이터가 모델 크기를 이김
- **Qwen3 시리즈는 `<think>` 토큰 노출로 BLEU/naturalness 하락** → RAG엔 Qwen2.5 권장
- **Gemma 계열은 GT 어휘 "복사" 경향** — 어휘 metric 1위, Judge 중하위 (의미 부족)
- **어휘 metric ↔ Judge 상관 거의 0** — 두 평가 체계가 독립적, 어휘만 보면 잘못된 선택 위험

---

## ⚖️ 라이선스 / 데이터 정책

- 본 저장소의 **코드는 MIT 라이선스**, **결과 점수는 자유롭게 공개**
- 원본 데이터는 **자체 라이선스/보안 정책**이 적용되는 도메인 매뉴얼이라 본 저장소에 미포함:
  - ❌ `corpus.parquet` (매뉴얼 본문 그대로)
  - ❌ `qa.parquet` (`retrieval_gt_contents`에 매뉴얼 일부)
  - ❌ `benchmark_*/0/...` (모델 raw 답변, 매뉴얼 기반)

평가 재현 시 원본 매뉴얼 PDF를 각자 보유한 후 `scripts/gen_qa_*.py`로 corpus·QA를 직접 생성합니다.

---

## 🛠 스크립트 / 설정

| 종류 | 파일 | 역할 |
|---|---|---|
| 자동화 | `scripts/auto_prod.sh` | 본 평가 (envA→envB→72B Judge) |
| 자동화 | `scripts/auto_retry.sh` | 실패 모델 재평가 |
| 자동화 | `scripts/auto_retry_g12b.sh` | gemma-3-12b 개별 평가 |
| Wrapper | `scripts/run_server_multi.sh` | 모델별 vLLM 기동 + AutoRAG 호출 |
| Judge | `scripts/judge_all.py` | 72B LLM Judge |
| QA 생성 | `scripts/gen_qa_official.py` | single-hop |
| QA 생성 | `scripts/gen_qa_multihop.py` | multi-hop (한국어 prompt 패치) |
| Config | `configs/config.server.api.prod.yaml` | 본 평가 (4 임베딩 + 3 reranker) |
| Config | `configs/config.server.api.prod.shortgen.yaml` | max_tokens 200 (4K context용) |
| Valid | `configs/valid_prod*.txt` | 모델 ID + 양자화/메모리 (`model|quant|dtype|util|mml`) |

자세한 실행 절차는 [`docs/SETUP.md`](docs/SETUP.md) 참고.

---

## 📊 평가 한계

- **단일 도메인**: 헬리콥터 정비 매뉴얼 79 QA에 한정 — 일반화 주의
- **단일 Judge**: Qwen2.5-72B 하나로만 평가
- **단일 실행**: 모델당 1회, 분산 미측정
- **제외 모델 1개**: `cyankiwi/gemma-4-31B-AWQ-4bit` — RTX 5090 sm_120 + TP=2 NCCL 미지원

---

## 🔗 참고

- AutoRAG: https://github.com/Marker-Inc-Korea/AutoRAG
- 평가 기간: 2026-05-19 ~ 2026-05-26
- 본 평가 결과는 helicop_agent `../autorag/adapter.py`를 통해 LangGraph 에이전트(`agent_v2`)의 `rag_search` 도구로 직접 swap-in되어 실서비스에 적용됨
