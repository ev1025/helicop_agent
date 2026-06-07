"""
#008 - 양자화 × 컨텍스트/출력 변형 12 가지 속도+정확도 비교.

매트릭스 (모델 3종 × 설정 4종 = 12):
                        max_tokens / ctx_chars
                    ┌─ V0 (1024/8000) │ V1 (256/8000) │ V2 (1024/3000) │ V12 (256/3000)
  bnb 4bit    (B)  │   1               2                3                4
  GGUF Q4_K_M (G)  │   5               6                7                8
  AWQ         (A)  │   9              10               11               12

같은 RAG 검색 결과 → 각 variant 로 동일 prompt → generate → 측정.

질의 (3개 — 시간 절약):
  Q1 베르누이 양력
  Q2 Vortex Ring State
  Q3 동적 롤오버

정확도: keyword_recall (정답 키워드 N 개 중 답변 포함 비율).

예상 총 시간:
  bnb V0  : ~6분/쿼리 (3 = 18분)
  bnb V12 : ~30초/쿼리 (3 = 1.5분)
  GGUF    : ~30~60초/쿼리 평균 (12 = 6~12분)
  AWQ     : ~2분/쿼리 평균 (12 = 24분)
  합      : 약 50~80분
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


# ────────────────────────────────────────────────
# 테스트 질의 + 예상 키워드
# ────────────────────────────────────────────────
QUERIES = [
    {
        "id": "Q1",
        "query": "베르누이 원리로 양력이 어떻게 발생하나요?",
        "expected_keywords": ["베르누이", "압력", "속도", "양력"],
    },
    {
        "id": "Q2",
        "query": "Vortex Ring State 란 무엇인가요?",
        "expected_keywords": ["하강", "공기", "양력", "수직"],
    },
    {
        "id": "Q3",
        "query": "동적 롤오버는 어떤 상황에서 발생하나요?",
        "expected_keywords": ["기울", "회전", "지면", "측"],
    },
]

VARIANTS = [
    {"name": "V0_baseline",   "max_tokens": 1024, "ctx": 8000},
    {"name": "V1_short_out",  "max_tokens": 256,  "ctx": 8000},
    {"name": "V2_short_ctx",  "max_tokens": 1024, "ctx": 3000},
    {"name": "V12_both",      "max_tokens": 256,  "ctx": 3000},
]

PROMPT_TEMPLATE = """당신은 헬리콥터 표준 교재 기반 AI 튜터입니다. 한국어로 간결하게 답변하세요.

참고 문서:
{context}

질문: {query}

답변:"""


def build_prompt(query: str, context: str, max_chars: int) -> str:
    truncated = context[:max_chars] + ("..." if len(context) > max_chars else "")
    return PROMPT_TEMPLATE.format(query=query, context=truncated)


def keyword_recall(answer: str, expected: list) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for k in expected if k in answer)
    return hits / len(expected)


def gather_rag_results():
    """모든 질의에 대해 RAG 검색 한 번. variants 간 동일 입력 보장."""
    from app.core import models
    from app.services.rag_service import rag_search_with_rerank

    models.load_embedding_model()
    models.load_vector_db()

    by_q = {}
    for item in QUERIES:
        results = rag_search_with_rerank(item["query"])
        context = "\n\n".join(
            f"[p.{r.get('page')}] {r.get('content','')}" for r in results
        )
        by_q[item["id"]] = context
    return by_q


# ─────────────────────────────────────────────────────
# 모델별 측정 함수
# ─────────────────────────────────────────────────────
def measure_bnb(rag_contexts: dict) -> list[dict]:
    print("\n" + "=" * 70)
    print("[BNB 4bit / transformers] 1회 로드 후 4 variants × 3 질의")
    print("=" * 70)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
    print(f"  로드: {time.perf_counter()-t0:.1f}초, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    out = []
    for v in VARIANTS:
        for q in QUERIES:
            prompt = build_prompt(q["query"], rag_contexts[q["id"]], v["ctx"])
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            n_input = inputs.input_ids.shape[1]
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=v["max_tokens"],
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            elapsed = time.perf_counter() - t0
            n_out = outputs[0][n_input:].shape[0]
            answer = tok.decode(outputs[0][n_input:], skip_special_tokens=True).strip()
            kr = keyword_recall(answer, q["expected_keywords"])
            tps = n_out / elapsed if elapsed > 0 else 0
            row = {
                "engine": "BNB", "variant": v["name"], "qid": q["id"],
                "n_input_tokens": n_input, "n_output_tokens": n_out,
                "elapsed_sec": round(elapsed, 2), "tok_per_sec": round(tps, 1),
                "keyword_recall": round(kr, 2), "answer_length": len(answer),
                "answer_preview": answer[:120],
            }
            out.append(row)
            print(f"  [BNB {v['name']:14s}] {q['id']}  in={n_input:5d} out={n_out:4d} "
                  f"{elapsed:6.1f}s ({tps:5.1f}t/s)  kw={kr:.2f}")

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return out


def measure_gguf(rag_contexts: dict) -> list[dict]:
    print("\n" + "=" * 70)
    print("[GGUF Q4_K_M / llama.cpp] 1회 로드 후 4 variants × 3 질의")
    print("=" * 70)
    import torch
    # Windows 한글 경로 + CUDA dll loading 우회
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.isdir(torch_lib) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib)
    try:
        from llama_cpp import Llama
    except Exception as e:
        print(f"  llama_cpp import 실패 - 건너뜀: {e}")
        return []

    candidates = list(Path.home().glob(".cache/huggingface/hub/**/*Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    if not candidates:
        print("  GGUF 파일 못 찾음 - 다운로드 안 됐거나 경로 다름")
        return []
    gguf_path = str(candidates[0])
    print(f"  파일: {gguf_path}")

    t0 = time.perf_counter()
    llm = Llama(model_path=gguf_path, n_gpu_layers=-1, n_ctx=8192, verbose=False)
    print(f"  로드: {time.perf_counter()-t0:.1f}초")

    out = []
    for v in VARIANTS:
        for q in QUERIES:
            prompt = build_prompt(q["query"], rag_contexts[q["id"]], v["ctx"])
            chat_messages = [{"role": "user", "content": prompt}]
            t0 = time.perf_counter()
            try:
                resp = llm.create_chat_completion(
                    messages=chat_messages, max_tokens=v["max_tokens"], temperature=0,
                )
                ok = True; err = None
                answer = resp["choices"][0]["message"]["content"].strip()
                usage = resp.get("usage", {})
                n_in = usage.get("prompt_tokens", 0)
                n_out = usage.get("completion_tokens", 0)
            except Exception as e:
                ok = False; err = str(e); answer = ""; n_in = 0; n_out = 0
            elapsed = time.perf_counter() - t0
            tps = n_out / elapsed if elapsed > 0 else 0
            kr = keyword_recall(answer, q["expected_keywords"])
            row = {
                "engine": "GGUF", "variant": v["name"], "qid": q["id"],
                "n_input_tokens": n_in, "n_output_tokens": n_out,
                "elapsed_sec": round(elapsed, 2), "tok_per_sec": round(tps, 1),
                "keyword_recall": round(kr, 2), "answer_length": len(answer),
                "answer_preview": answer[:120], "error": err,
            }
            out.append(row)
            mark = "" if ok else " [ERR]"
            print(f"  [GGUF {v['name']:13s}] {q['id']}  in={n_in:5d} out={n_out:4d} "
                  f"{elapsed:6.1f}s ({tps:5.1f}t/s)  kw={kr:.2f}{mark}")

    del llm; gc.collect(); torch.cuda.empty_cache()
    return out


def measure_awq(rag_contexts: dict) -> list[dict]:
    print("\n" + "=" * 70)
    print("[AWQ / AutoAWQ] 1회 로드 후 4 variants × 3 질의")
    print("=" * 70)
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  AutoAWQ 미설치 - 건너뜀: {e}")
        return []
    import torch

    MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    try:
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_quantized(
            MODEL, fuse_layers=True, trust_remote_code=True, safetensors=True,
        )
        print(f"  로드: {time.perf_counter()-t0:.1f}초, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  로드 실패 - 건너뜀: {e}")
        return []

    out = []
    for v in VARIANTS:
        for q in QUERIES:
            prompt = build_prompt(q["query"], rag_contexts[q["id"]], v["ctx"])
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.model.device)
            n_input = inputs.input_ids.shape[1]
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=v["max_tokens"], do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                ok = True; err = None
                n_out = outputs[0][n_input:].shape[0]
                answer = tok.decode(outputs[0][n_input:], skip_special_tokens=True).strip()
            except Exception as e:
                ok = False; err = str(e); answer = ""; n_out = 0
            elapsed = time.perf_counter() - t0
            tps = n_out / elapsed if elapsed > 0 else 0
            kr = keyword_recall(answer, q["expected_keywords"])
            row = {
                "engine": "AWQ", "variant": v["name"], "qid": q["id"],
                "n_input_tokens": n_input, "n_output_tokens": n_out,
                "elapsed_sec": round(elapsed, 2), "tok_per_sec": round(tps, 1),
                "keyword_recall": round(kr, 2), "answer_length": len(answer),
                "answer_preview": answer[:120], "error": err,
            }
            out.append(row)
            mark = "" if ok else " [ERR]"
            print(f"  [AWQ {v['name']:13s}] {q['id']}  in={n_input:5d} out={n_out:4d} "
                  f"{elapsed:6.1f}s ({tps:5.1f}t/s)  kw={kr:.2f}{mark}")

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    print("=" * 70)
    print("[1/5] RAG 검색 (모든 variant 동일 입력)")
    print("=" * 70)
    rag_contexts = gather_rag_results()
    for qid, ctx in rag_contexts.items():
        print(f"  {qid}: {len(ctx)}자")

    bnb_rows = measure_bnb(rag_contexts)
    gguf_rows = measure_gguf(rag_contexts)
    awq_rows = measure_awq(rag_contexts)

    all_rows = bnb_rows + gguf_rows + awq_rows

    # ── 요약 (12 cell)
    print("\n" + "=" * 70)
    print("[종합] 12 (engine × variant) 평균")
    print("=" * 70)
    cells = {}
    for r in all_rows:
        key = (r["engine"], r["variant"])
        cells.setdefault(key, []).append(r)

    summary = []
    print(f"\n{'engine':6s} | {'variant':14s} | {'avg sec':>8s} | {'avg t/s':>8s} | {'avg kw':>6s} | {'avg len':>7s}")
    print("-" * 70)
    for (eng, var), rows in sorted(cells.items()):
        n = len(rows)
        avg_sec = sum(r["elapsed_sec"] for r in rows) / n
        avg_tps = sum(r["tok_per_sec"] for r in rows) / n
        avg_kw = sum(r["keyword_recall"] for r in rows) / n
        avg_len = sum(r["answer_length"] for r in rows) / n
        summary.append({
            "engine": eng, "variant": var,
            "avg_elapsed_sec": round(avg_sec, 2),
            "avg_tok_per_sec": round(avg_tps, 1),
            "avg_keyword_recall": round(avg_kw, 3),
            "avg_answer_length": round(avg_len, 0),
        })
        print(f"{eng:6s} | {var:14s} | {avg_sec:>8.2f} | {avg_tps:>8.1f} | {avg_kw:>6.2f} | {avg_len:>7.0f}")

    out = ROOT / "results" / "agent_v2_changes" / "08_quantization_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": all_rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
