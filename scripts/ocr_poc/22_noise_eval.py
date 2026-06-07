"""
OCR POC Phase 4-B-1 — 노이즈 비율 자동 계산 + 전처리 + 정화 텍스트 저장.

각 모델 출력 텍스트에서 RAG 인덱싱에 해로운 노이즈를 탐지·제거:
  1. Markdown 이미지 링크   ![alt](path)
  2. HTML 태그              <tag ...>  </tag>
  3. 절대 경로              C:\\... 또는 /home/...  (Marker / MinerU 가 잘 만듦)
  4. URL                    http://..., https://...
  5. 짧은 반복 토큰 환각    예: "asdf asdf asdf ..." (TrOCR 류)

산출:
  results/ocr_poc/noise_eval.json
    - 모델별 평균 noise_ratio, raw_chars, clean_chars
    - 페이지별 상세
  results/ocr_poc/{model}_clean/page_NNN.txt
    - 노이즈 제거된 텍스트

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/22_noise_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import results_dir, baseline_dir, pages_dir

RESULTS_DIR = results_dir()
BASELINE_DIR = baseline_dir()
SAMPLES_JSON = BASELINE_DIR / "samples.json"
PYPDF_PAGES = pages_dir()

MODELS = ["pypdf", "easyocr", "paddleocr", "tesseract", "surya", "marker", "mineru", "donut"]

# ── 노이즈 패턴 ─────────────────────────────────────────
# 캡처해서 길이를 노이즈로 카운트.
PATTERNS = {
    "md_image":   re.compile(r"!\[[^\]]*\]\([^)]*\)"),
    "html_tag":   re.compile(r"<[^>\n]{1,200}>"),
    "abs_path":   re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|var)/)[^\s)\]'\"]+"),
    "url":        re.compile(r"https?://\S+"),
    "code_fence": re.compile(r"```[a-zA-Z]*\n[\s\S]*?```"),
    # 3자 이하 토큰이 연속 5회 이상 반복 (TrOCR 류 환각)
    "repeat":     re.compile(r"(\b\S{1,4}\b)(?:\s+\1){4,}"),
}


def analyze(text: str) -> dict:
    """텍스트의 노이즈 추출 + 제거된 clean 반환."""
    total = len(text)
    if total == 0:
        return {"chars": 0, "noise_chars": 0, "clean_chars": 0,
                "noise_ratio": 0.0, "clean": "", "matches": {}}

    matches = {}
    noise_chars = 0
    cleaned = text
    for name, pat in PATTERNS.items():
        found = pat.findall(cleaned)
        # findall 이 그룹 캡처면 첫 그룹만 반환 → 전체 매치 길이 측정 위해 finditer 로 재계산
        match_objs = list(pat.finditer(cleaned))
        if match_objs:
            n = sum(len(m.group(0)) for m in match_objs)
            matches[name] = {"count": len(match_objs), "chars": n,
                             "examples": [m.group(0)[:80] for m in match_objs[:3]]}
            noise_chars += n
            cleaned = pat.sub(" ", cleaned)

    # 추가 정화: 다중 공백 → 단일 공백, 시작/끝 공백 제거
    cleaned_pretty = re.sub(r"\s+", " ", cleaned).strip()

    return {
        "chars": total,
        "noise_chars": noise_chars,
        "clean_chars": len(cleaned_pretty),
        "noise_ratio": round(noise_chars / total, 4),
        "clean": cleaned_pretty,
        "matches": matches,
    }


def load_text(model: str, page: int) -> str:
    if model == "pypdf":
        f = PYPDF_PAGES / f"page_{page:03d}.txt"
    else:
        f = RESULTS_DIR / model / f"page_{page:03d}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def main():
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    pages = [s["page"] for s in samples]

    print(f"=== 노이즈 평가: {len(MODELS)} 모델 × {len(pages)} 페이지 ===\n")

    summary = {}

    for m in MODELS:
        clean_dir = RESULTS_DIR / f"{m}_clean"
        clean_dir.mkdir(parents=True, exist_ok=True)

        per_page = []
        total_raw = total_noise = total_clean = 0
        pattern_totals = {k: 0 for k in PATTERNS}

        for p in pages:
            text = load_text(m, p)
            a = analyze(text)
            per_page.append({
                "page": p,
                "chars": a["chars"],
                "noise_chars": a["noise_chars"],
                "clean_chars": a["clean_chars"],
                "noise_ratio": a["noise_ratio"],
                "matches": {k: v["count"] for k, v in a["matches"].items()},
            })
            (clean_dir / f"page_{p:03d}.txt").write_text(a["clean"], encoding="utf-8")
            total_raw += a["chars"]
            total_noise += a["noise_chars"]
            total_clean += a["clean_chars"]
            for k, v in a["matches"].items():
                pattern_totals[k] += v["count"]

        avg_ratio = round(total_noise / max(1, total_raw), 4)
        summary[m] = {
            "raw_total_chars": total_raw,
            "noise_total_chars": total_noise,
            "clean_total_chars": total_clean,
            "overall_noise_ratio": avg_ratio,
            "pattern_counts": pattern_totals,
            "per_page": per_page,
        }
        print(f"  · {m:<10} raw={total_raw:>5}  noise={total_noise:>5}  clean={total_clean:>5}"
              f"  ratio={avg_ratio:>6.1%}  patterns={pattern_totals}")

    out = RESULTS_DIR / "noise_eval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 요약 표 ─────────────────────────────────────
    print("\n=== 모델별 노이즈 요약 (낮을수록 좋음) ===")
    print(f"{'모델':<12} {'raw글자':>8} {'노이즈':>8} {'clean글자':>10} {'노이즈비율':>10} {'주요 패턴':<30}")
    sorted_models = sorted(summary.items(), key=lambda kv: kv[1]["overall_noise_ratio"])
    for m, s in sorted_models:
        top_pat = sorted(s["pattern_counts"].items(), key=lambda kv: -kv[1])
        top = ", ".join(f"{k}={v}" for k, v in top_pat if v > 0)[:30] or "-"
        print(f"  {m:<10} {s['raw_total_chars']:>8} {s['noise_total_chars']:>8} "
              f"{s['clean_total_chars']:>10} {s['overall_noise_ratio']:>9.1%}  {top}")

    print(f"\n✅ 저장:")
    print(f"   {out}")
    print(f"   results/ocr_poc/{{model}}_clean/page_NNN.txt  (정화된 텍스트)")


if __name__ == "__main__":
    main()
