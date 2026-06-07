"""
OCR POC Phase 3 — 모델별 결과 통합 비교.

모든 results/ocr_poc/{model}/page_NNN.txt + timings.json 을 읽어 종합 비교.

산출:
  - 모델 × 페이지 글자수 매트릭스
  - 모델별 평균 처리시간 / 추출량 / 한글비율
  - 카테고리별 (weak / table_suspect / normal) 모델별 회수율
  - results/ocr_poc/comparison.json + comparison.md

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/20_compare.py
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
PAGES_PYPDF = pages_dir()

# 비교 대상 모델 디렉토리 (있는 것만 자동 포함)
# TrOCR 제외: single-line OCR 이라 페이지 입력 시 환각만 — 페이지별 비교 의미 없음
MODEL_DIRS = ["easyocr", "paddleocr", "tesseract", "surya", "marker", "mineru", "donut"]

_HANGUL_RE = re.compile(r"[가-힯㄰-㆏]")


def char_stats(text: str) -> dict:
    n = len(text)
    if n == 0:
        return {"chars": 0, "kor_ratio": 0.0}
    return {
        "chars": n,
        "kor_ratio": round(len(_HANGUL_RE.findall(text)) / n, 4),
    }


def load_model_outputs(model_dir: Path, pages: list[int]) -> dict[int, str]:
    """페이지번호 → 추출 텍스트 dict."""
    out = {}
    for p in pages:
        f = model_dir / f"page_{p:03d}.txt"
        out[p] = f.read_text(encoding="utf-8") if f.exists() else ""
    return out


def main():
    if not SAMPLES_JSON.exists():
        print(f"❌ {SAMPLES_JSON} 없음.")
        sys.exit(1)

    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    pages = [s["page"] for s in samples]
    page_cat = {s["page"]: s["category"] for s in samples}

    # PyPDF baseline 로드 (페이지별 원문)
    pypdf_texts = {}
    for p in pages:
        f = PAGES_PYPDF / f"page_{p:03d}.txt"
        pypdf_texts[p] = f.read_text(encoding="utf-8") if f.exists() else ""

    all_models = {"pypdf": pypdf_texts}
    timings = {}

    for name in MODEL_DIRS:
        d = RESULTS_DIR / name
        if not d.exists():
            continue
        all_models[name] = load_model_outputs(d, pages)
        t_file = d / "timings.json"
        if t_file.exists():
            timings[name] = json.loads(t_file.read_text(encoding="utf-8"))

    print(f"=== 비교 대상 모델: {list(all_models)} ===\n")

    # ── 매트릭스: 페이지 × 모델 글자수 ─────────────────
    matrix_rows = []
    for p in pages:
        row = {"page": p, "category": page_cat[p]}
        for m, texts in all_models.items():
            stats = char_stats(texts.get(p, ""))
            row[f"{m}_chars"] = stats["chars"]
            row[f"{m}_kor"] = stats["kor_ratio"]
        matrix_rows.append(row)

    # ── 모델별 요약 ──────────────────────────────────
    model_summary = {}
    for m, texts in all_models.items():
        total_chars = sum(len(t) for t in texts.values())
        kor = sum(len(_HANGUL_RE.findall(t)) for t in texts.values())
        nonempty = sum(1 for t in texts.values() if t)
        model_summary[m] = {
            "total_chars": total_chars,
            "kor_chars": kor,
            "avg_kor_ratio": round(kor / max(1, total_chars), 4),
            "nonempty_pages": nonempty,
            "total_pages": len(pages),
            "vs_pypdf_chars_ratio": round(
                total_chars / max(1, sum(len(t) for t in pypdf_texts.values())), 3
            ),
        }
        if m in timings:
            model_summary[m]["avg_seconds_per_page"] = timings[m].get("avg_seconds_per_page")
            model_summary[m]["total_seconds"] = timings[m].get("total_seconds")

    # ── 카테고리별 모델별 회수율 ──────────────────────
    cat_summary = {}
    for cat in ("weak", "table_suspect", "normal"):
        cat_pages = [p for p in pages if page_cat[p] == cat]
        cat_summary[cat] = {"n_pages": len(cat_pages)}
        for m, texts in all_models.items():
            chars = sum(len(texts.get(p, "")) for p in cat_pages)
            cat_summary[cat][f"{m}_avg_chars"] = round(chars / max(1, len(cat_pages)), 1)

    out = {
        "models": list(all_models),
        "n_pages": len(pages),
        "categories": cat_summary,
        "model_summary": model_summary,
        "matrix": matrix_rows,
    }
    (RESULTS_DIR / "comparison.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 마크다운 리포트 ──────────────────────────────
    md_lines = ["# OCR 모델 벤치마크 결과\n"]
    md_lines.append(f"- 비교 대상: **{len(all_models)}개 모델** ({', '.join(all_models)})")
    md_lines.append(f"- 샘플: **{len(pages)}장** (약함 / 표·그림 / 정상 혼합)\n")

    md_lines.append("## 모델별 요약\n")
    md_lines.append("| 모델 | 총 글자 | vs PyPDF | 평균 한글% | 비빈 페이지 | avg sec/page |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for m, s in model_summary.items():
        sec = s.get("avg_seconds_per_page", "-")
        md_lines.append(
            f"| {m} | {s['total_chars']:>6} | {s['vs_pypdf_chars_ratio']:.2f}x | "
            f"{s['avg_kor_ratio']:.1%} | {s['nonempty_pages']}/{s['total_pages']} | {sec} |"
        )

    md_lines.append("\n## 카테고리별 평균 추출 글자수\n")
    md_lines.append("| 카테고리 | 페이지수 | " + " | ".join(all_models) + " |")
    md_lines.append("|---|---:|" + "|".join(["---:"] * len(all_models)) + "|")
    for cat, info in cat_summary.items():
        row = f"| {cat} | {info['n_pages']} | "
        row += " | ".join(str(info[f"{m}_avg_chars"]) for m in all_models)
        row += " |"
        md_lines.append(row)

    md_lines.append("\n## 페이지×모델 글자수 매트릭스\n")
    md_lines.append("| page | cat | " + " | ".join(all_models) + " |")
    md_lines.append("|---:|---|" + "|".join(["---:"] * len(all_models)) + "|")
    for row in matrix_rows:
        md_lines.append(
            f"| {row['page']} | {row['category']} | "
            + " | ".join(str(row[f"{m}_chars"]) for m in all_models)
            + " |"
        )

    (RESULTS_DIR / "comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

    # ── 콘솔 출력 ────────────────────────────────────
    print("=== 모델별 요약 (vs PyPDF) ===\n")
    print(f"{'모델':<12} {'총글자':>8} {'vs.PyPDF':>10} {'한글%':>7} {'avg sec':>9}")
    for m, s in model_summary.items():
        sec = s.get("avg_seconds_per_page", "-")
        sec_str = f"{sec:>5}s" if sec != "-" else f"{sec:>6}"
        print(f"  {m:<10} {s['total_chars']:>8} {s['vs_pypdf_chars_ratio']:>9.2f}x "
              f"{s['avg_kor_ratio']:>6.1%} {sec_str:>9}")

    print(f"\n✅ {RESULTS_DIR / 'comparison.json'}")
    print(f"   {RESULTS_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
