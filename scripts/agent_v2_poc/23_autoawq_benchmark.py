"""
WSL/Linux: AutoAWQ + transformers 벤치 (vLLM 의 KV cache 미리할당 우회).

8GB VRAM 에 vLLM 은 7B AWQ 못 띄움. 하지만 autoawq + transformers 는
가능 (KV cache 동적 할당, BNB 와 비슷한 메모리 패턴).

같은 4 variants (V0/V1/V2/V12) × 3 질의 측정 → #008 BNB/GGUF 와 비교.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

INPUT_JSON = Path(
    "/mnt/c/Users/eg287/OneDrive/바탕 화면/project/수리온/team_gitlab/"
    "results/agent_v2_changes/awq_test_inputs.json"
)
OUTPUT_JSON = Path(
    "/mnt/c/Users/eg287/OneDrive/바탕 화면/project/수리온/team_gitlab/"
    "results/agent_v2_changes/15_autoawq_benchmark.json"
)

PROMPT_TEMPLATE = """당신은 헬리콥터 표준 교재 기반 AI 튜터입니다. 한국어로 간결하게 답변하세요.

참고 문서:
{context}

질문: {query}

답변:"""

VARIANTS = [
    {"name": "V0_baseline",   "max_tokens": 1024, "ctx": 8000},
    {"name": "V1_short_out",  "max_tokens": 256,  "ctx": 8000},
    {"name": "V2_short_ctx",  "max_tokens": 1024, "ctx": 3000},
    {"name": "V12_both",      "max_tokens": 256,  "ctx": 3000},
]


def keyword_recall(answer: str, expected: list) -> float:
    if not expected:
        return 0.0
    return sum(1 for k in expected if k in answer) / len(expected)


def main():
    print("=" * 70)
    print("[1/4] 입력 데이터 로드")
    print("=" * 70)
    inputs = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    for it in inputs:
        print(f"  {it['id']}: ctx {it['context_length']}자")

    print()
    print("=" * 70)
    print("[2/4] AutoAWQ + transformers 모델 로드")
    print("=" * 70)
    import torch
    from transformers import AutoTokenizer
    # transformers 5.8.0 의 AWQ quantizer 가 gptqmodel 만 인정 → AutoAWQ 직접 사용
    from awq import AutoAWQForCausalLM

    MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    mdl = AutoAWQForCausalLM.from_quantized(MODEL, fuse_layers=True, safetensors=True)
    # AutoAWQ 모델은 .model 안에 실제 nn.Module
    device = mdl.model.device if hasattr(mdl, "model") else "cuda"
    print(f"  로드: {time.perf_counter()-t0:.1f}초, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB, device={device}")

    # ─────────────────────────────────────
    print()
    print("=" * 70)
    print("[3/4] 4 variants × 3 질의 단일 호출 측정")
    print("=" * 70)
    rows = []
    for v in VARIANTS:
        for it in inputs:
            ctx = it["rag_context"][:v["ctx"]]
            prompt = PROMPT_TEMPLATE.format(query=it["query"], context=ctx)
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs_t = tok(text, return_tensors="pt").to(device)
            n_in = inputs_t.input_ids.shape[1]
            t0 = time.perf_counter()
            with torch.no_grad():
                out = mdl.generate(
                    **inputs_t, max_new_tokens=v["max_tokens"],
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            elapsed = time.perf_counter() - t0
            n_out = out[0][n_in:].shape[0]
            answer = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
            kr = keyword_recall(answer, it["expected_keywords"])
            tps = n_out / elapsed if elapsed > 0 else 0
            row = {
                "engine": "AutoAWQ", "variant": v["name"], "qid": it["id"],
                "n_input_tokens": n_in, "n_output_tokens": n_out,
                "elapsed_sec": round(elapsed, 2), "tok_per_sec": round(tps, 1),
                "keyword_recall": round(kr, 2),
                "answer_length": len(answer),
                "answer_preview": answer[:120],
            }
            rows.append(row)
            print(f"  [{v['name']:14s}] {it['id']}  in={n_in:5d} out={n_out:4d} "
                  f"{elapsed:6.2f}s ({tps:5.1f}t/s)  kw={kr:.2f}")

    # ─────────────────────────────────────
    print()
    print("=" * 70)
    print("[4/4] 종합 (variant 평균)")
    print("=" * 70)
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    summary = []
    print(f"\n{'variant':14s} | {'avg sec':>8s} | {'avg t/s':>8s} | {'avg kw':>6s} | {'avg len':>7s}")
    print("-" * 60)
    for v, vrows in sorted(by_variant.items()):
        n = len(vrows)
        avg_sec = sum(r["elapsed_sec"] for r in vrows) / n
        avg_tps = sum(r["tok_per_sec"] for r in vrows) / n
        avg_kw = sum(r["keyword_recall"] for r in vrows) / n
        avg_len = sum(r["answer_length"] for r in vrows) / n
        summary.append({
            "variant": v, "avg_elapsed_sec": round(avg_sec, 2),
            "avg_tok_per_sec": round(avg_tps, 1),
            "avg_keyword_recall": round(avg_kw, 3),
            "avg_answer_length": round(avg_len, 0),
        })
        print(f"{v:14s} | {avg_sec:>8.2f} | {avg_tps:>8.1f} | {avg_kw:>6.2f} | {avg_len:>7.0f}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps({"engine": "AutoAWQ_Qwen2.5-7B", "summary": summary, "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
