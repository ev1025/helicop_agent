"""
GPU 실제 사용 검증 — Qwen 추론 중 GPU-Util 과 VRAM 을 샘플링.

- inference 시작 전, 추론 중 (별도 스레드에서 0.5s 간격), 종료 후를 모두 기록.
- GPU-Util 이 80%+ 면 GPU 가 진짜 일하는 중. 0~20% 인데 CPU 100% 면 fallback.

실행: .venv/Scripts/python.exe scripts/agent_v2_poc/15_gpu_usage_check.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


def sample_nvidia_smi():
    """현재 시점의 GPU 통계 한 번 샘플."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        util, mem_used, mem_total, temp = [x.strip() for x in out.split(",")]
        return {
            "gpu_util": int(util),
            "mem_used_mb": int(mem_used),
            "mem_total_mb": int(mem_total),
            "temp_c": int(temp),
        }
    except Exception as e:
        return {"error": str(e)}


class GPUSampler(threading.Thread):
    """추론 도는 동안 0.5초 간격으로 GPU 통계 샘플링."""
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.samples.append({
                "t": time.perf_counter(),
                **sample_nvidia_smi(),
            })
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2)


def main():
    print("=" * 70)
    print("GPU 사용 진단 — Qwen 추론 중 GPU-Util 모니터링")
    print("=" * 70)

    # ─── 추론 전 baseline
    print()
    print("[Baseline] (모델 로드 전)")
    base = sample_nvidia_smi()
    print(f"  GPU-Util: {base.get('gpu_util')}%, VRAM: {base.get('mem_used_mb')} MB / {base.get('mem_total_mb')} MB")

    # ─── 모델 로드
    print()
    print("[1/3] Qwen 7B 4bit 로드")
    from app.core.agent_v2.llm_qwen import build_qwen_chat_model
    t0 = time.perf_counter()
    chat = build_qwen_chat_model(model_name="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=200)
    load_sec = time.perf_counter() - t0
    after_load = sample_nvidia_smi()
    print(f"  로드 시간: {load_sec:.1f}초")
    print(f"  GPU-Util: {after_load.get('gpu_util')}%, VRAM: {after_load.get('mem_used_mb')} MB")

    # ─── 추론 중 모니터링
    print()
    print("[2/3] 추론 (200 토큰 생성). GPU 모니터링 0.5초 간격.")
    from langchain_core.messages import HumanMessage, SystemMessage

    sampler = GPUSampler(interval=0.5)
    sampler.start()
    inf_t0 = time.perf_counter()
    try:
        ai = chat.invoke([
            SystemMessage(content="간결한 한국어로 답하세요."),
            HumanMessage(content="베르누이 원리를 5문장 이내로 설명해주세요."),
        ])
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = str(e)
        ai = None
    inf_sec = time.perf_counter() - inf_t0
    sampler.stop()

    samples = sampler.samples
    if samples:
        utils = [s.get("gpu_util", 0) for s in samples if "gpu_util" in s]
        mems = [s.get("mem_used_mb", 0) for s in samples if "mem_used_mb" in s]
        avg_util = sum(utils) / len(utils) if utils else 0
        max_util = max(utils) if utils else 0
        min_util = min(utils) if utils else 0
        peak_mem = max(mems) if mems else 0
    else:
        avg_util = max_util = min_util = peak_mem = 0

    print(f"  추론 시간: {inf_sec:.1f}초, ok={ok}")
    if ok:
        content = ai.content if hasattr(ai, "content") else str(ai)
        print(f"  답변 길이: {len(content)}자")
        print(f"  답변 미리보기: {content[:150]!r}")
    print()
    print(f"  GPU-Util  : avg={avg_util:.1f}%, max={max_util}%, min={min_util}%   (n_samples={len(samples)})")
    print(f"  VRAM peak : {peak_mem} MB ({peak_mem/1024:.2f} GB)")

    # ─── 진단
    print()
    print("=" * 70)
    print("[3/3] 진단")
    print("=" * 70)
    if avg_util >= 70:
        verdict = "✅ GPU 정상 사용 중 (GPU-Util 평균 70%+)"
    elif avg_util >= 30:
        verdict = "⚠️  GPU 부분 사용 (CPU 와 섞여 돌고 있을 가능성)"
    else:
        verdict = "❌ GPU 거의 안 씀 (CPU fallback 의심)"
    print(verdict)
    print(f"\n참고:")
    print(f"  - 진짜 GPU 풀가동 → GPU-Util 90~100%")
    print(f"  - bnb 4bit + transformers 는 일반적으로 GPU-Util 60~85% (커널 호출 사이 idle 있음)")
    print(f"  - 만약 0~20% 면 device_map 설정 잘못되었을 가능성")

    # 저장
    out = ROOT / "results" / "agent_v2_changes" / "00_gpu_usage_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "baseline": base,
        "after_model_load": after_load,
        "model_load_sec": round(load_sec, 2),
        "inference_sec": round(inf_sec, 2),
        "inference_ok": ok,
        "inference_error": err,
        "avg_gpu_util": round(avg_util, 1),
        "max_gpu_util": max_util,
        "min_gpu_util": min_util,
        "peak_vram_mb": peak_mem,
        "n_samples": len(samples),
        "samples": samples[:60],  # 첫 30초까지만 저장
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
