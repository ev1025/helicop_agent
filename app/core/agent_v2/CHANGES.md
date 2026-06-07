# agent_v2 변경 이력

| # | 시도 | 결과 |
|---|---|---|
| **#1** | RAG 결과를 본문 순서 (page → parent_chunk_id) 로 정렬 | 단조 증가 0/7 → **7/7**<br>인접 페이지 거리 75.5 → **27.8** (-63%) |
| **#2** | `bind_tools(tool_choice=...)` 도구 호출 강제 옵션 | 호출 성공률 60% → **100%**<br>BPE 노이즈 11.2자 → **0자**<br>속도 7.6s → **1.4s** (5.5×) |
| **#3** | RAG top_k 튜닝 — `vector_search_top_k` 30→60, `reranker_top_k` 7→6 | 정밀도 0.216 → **0.299** (+39%)<br>속도 842ms → 1430ms (1.7×) |
| **#4** | Sentence-Window — 매칭 Parent ±1 인접 청크 자동 회수 | window=1 무비용 (5ms)<br>평균 +1.4개 인접 추가, 컨텍스트 길이 +830자 |
| **#5** | Qwen 7B 양자화 백엔드 결정 — BNB / GGUF / AutoAWQ 속도 측정 | BNB 304s<br>**GGUF 5.2s (60× 빠름)**<br>AutoAWQ 9.6s (32× 빠름, WSL 에서만 동작)<br>→ **GGUF 채택** |
| **#6** | `QwenLlamaCppChatModel` 신규 — LangChain + GGUF 통합 ChatModel | multi_graph 18~20s/쿼리<br>BNB 6분 대비 **20× 빠름** |
| **#7** | Critic feedback loop — Critic 의 issues 를 worker 시스템 프롬프트에 주입해 retry | retry 후 답변 실제 개선 (베르누이 "잘못 → 정확", score 6→7) |
| **#8** | **fair 3-way 비교** (Qwen 7B GGUF 동일): A=수동 단일 / B=LangGraph 단일 / C=LangGraph 멀티 | A: 2.5s, kw=0.066 (답변 부정확)<br>B: 7.2s, kw=0.125 (3개 무한 루프)<br>**C: 10.9s, kw=0.416** (A 의 6.3×)<br>→ 단일 → 멀티 전환이 큰 이득 |
| **#9** | Token-단위 streaming — `QwenLlamaCppChatModel._stream` (`create_completion` 경로, ChatML 직접 구성) + `researcher.run_streaming` (Critic 은 streaming UX 충돌로 skip) | TTFT 10.9s → **6.5s**, 총 시간 10.9s → 9.7s<br>status 메시지(검색 중/결과 N자)로 대기 체감 완화<br>답변 품질 ↑ (이전 `create_chat_completion` 경로는 RAG 컨텍스트 일부만 사용 + UTF-8 글자 누락 → `create_completion` 경로로 전체 컨텍스트 사용 + 누락 해결)<br>TTFT 의 ~5s 는 8GB GPU 에서 13KB RAG 컨텍스트 prefill 물리 하한 |
| **#10** | Langfuse 통합 — 모든 LLM 호출 + `rag_search` 도구 + RAG 파이프라인 단계를 한 trace 로. `.env` 자동 로드, env vars 없으면 no-op. `event_stream` 을 `async def` 로 + sync 호출을 `iterate_in_threadpool`/`anyio.to_thread.run_sync` 로 감싸 OTEL context 전파 | **요청 1개 = trace 1개** — `chat-v2-multi` ▸ Supervisor(RunnableSequence) ▸ rag_search(TOOL) ▸ {`rag.child_to_parent` n_child→n_parent, `rag.parent_rerank`, `rag.sentence_window`} ▸ 답변 streaming. 각 노드 input/output/latency/token UI 확인<br>핵심: async generator 는 event loop 에서 돌아 OTEL contextvars 가 yield/await 경계 유지 (sync generator + threadpool 은 매 `.next()` 가 새 context copy 라 안 됐음). `rag_service` 단계는 `maybe_span()`/`end_span()` 으로 명시 계측 |
| **#11** | RAG 파라미터를 LLM 손에서 잠금 + system prompt 역할 분리 | `rag_search` 시그니처에서 `top_k`/`reranker_top_k` 제거 → 항상 `config` 단일 소스 (LLM 이 임의로 3 박아 컨텍스트 축소하던 문제 차단)<br>검색 쿼리 추출용 `SEARCH_QUERY_SYSTEM` 신설 (워커 답변용 prompt 와 분리) — Tool-call LLM 은 query 만 결정, 답변 LLM 은 답변에만 집중<br>응답 시간 ~9.9s → ~7.4s (-25%) |
| **#12** | 임베딩 모델 한국어 특화 교체 — `all-MiniLM-L6-v2` (384d, 영어 위주) → **`dragonkue/snowflake-arctic-embed-l-v2.0-ko`** (1024d, 한국어 특화) | "베르누이 정리/양력" 같은 핵심 한국어 질의에서 에어포일 챕터 회수 성공 (이전엔 목차/시험과목 등 무관 페이지만 회수)<br>예: "양력은 어떻게 발생" — 이전 "정보 없음" → **126~449 토큰 정답** (받음각·압력차·뉴턴 제3법칙 등)<br>인덱싱: 256 parent / 1025 child 청크 (5분), 새 콜렉션으로 분리 (옛 MiniLM 콜렉션 디스크 보존) |
| **#13** | LLM 백엔드 Gemma 4 E4B-it 도입 + 기본 백엔드 변경 | `qwen_llama_cpp_chat_model.py` 에 `chat_format` 파라미터화 + `_stream` 분기 (Gemma 는 GGUF 임베디드 jinja, Qwen 은 chatml-function-calling 우회)<br>`build_gemma_chat_model_gguf()` 신설, `CHAT_V2_BACKEND=gemma` 기본값<br>Gemma 4 E4B 가 Qwen 2.5-7B 대비: 응답 ~38% 빠름 (10.5s→6.7s), instruction-following 강함 (SEARCH_QUERY_SYSTEM "PDF 금지" 등 strict 준수), 모르는 질문은 strict 하게 "정보 없음" (할루시네이션 적음, 안전 도메인 적합)<br>벤치마크상 4.5B effective 인데 MMLU-Pro 69.4 / GPQA Diamond 58.6 (Qwen 7B 의 56.3 / 36 대비 우위, 신세대 아키텍처 PLE 효과) |
| **#14** | Langfuse **Dataset + Experiment 자동화** — 코드/모델 바꿀 때마다 한 줄로 정량 평가, UI 에서 run 간 side-by-side 비교 | `35_langfuse_dataset_init.py` (1회: 7질의 + expected_keywords/route 업로드)<br>`36_langfuse_experiment_run.py` (매번: `dataset.run_experiment(task=SSE_call, evaluators=[kw_recall, route_correct, latency, length])`)<br>측정 결과 (Arctic-ko 임베딩, 7질의):<br>· **Gemma 4 E4B**: route 100%, latency 7.44s, kw 0.548, len 162자<br>· **Qwen 2.5-7B**: route 100%, latency 16.31s (2.2× 느림), kw 0.643 (둘러대기 효과), len 196자<br>→ Gemma 가 속도, Qwen 이 키워드 매치 — 트레이드오프 정량화됨 |

## 결론

✅ **LangGraph 멀티에이전트 도입 효과 명확** (#8)
- 정확도 6.3× ↑, 안정성 ↑ (단일 ReAct 무한 루프 차단)
- Supervisor 자동 라우팅 100% 정확
- Critic + retry feedback loop 답변 자동 개선

⚠️ **트레이드오프**: 응답 시간 2.7× (단일 사용자엔 OK, 다중 사용자는 vLLM 필요)

## 현재 운영 권장 설정

| 항목 | 값 |
|---|---|
| LLM | **Gemma 4 E4B-it GGUF Q4_K_M** (llama.cpp) — `CHAT_V2_BACKEND=gemma` (기본). Qwen 으로 전환은 `=qwen` |
| 임베딩 | **dragonkue/snowflake-arctic-embed-l-v2.0-ko** (한국어 특화, 1024d) |
| 라우터 | `/chat/v2/stream?mode=multi` |
| RAG | Parent-Child + 본문 순서 정렬 (#1) + Sentence-Window 1 (#4) |
| Reranker | OFF |
| 도구 호출 | tool_choice 강제 (#2), `rag_search` 인자는 `query` 만 LLM 결정 (#11) |
| System prompt | 검색 추출용 / 답변 작성용 분리 (#11) |
| 멀티에이전트 | Supervisor + Workers + Critic + feedback loop (#7) |

## 데이터 위치

- raw 측정 JSON: `results/agent_v2_changes/`
- 측정 스크립트: `scripts/agent_v2_poc/`

## 용어

- **단조 증가**: 검색된 청크들이 원본 문서 page 순서대로 정렬된 질의 수 (7질의 기준)
- **인접 페이지 거리**: 검색 결과 청크들 사이의 평균 page 간격 (작을수록 본문이 연결됨)
- **BPE 노이즈**: Qwen 토크나이저의 도구 호출 특수태그(`<tool_call>` 등)가 답변 본문에 섞여 나온 평균 글자 수
- **kw (keyword recall)**: 답변에 정답 키워드가 포함된 비율 (0~1)
