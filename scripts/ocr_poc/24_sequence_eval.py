"""
OCR POC Phase 4-B-3 — 글자 순서 (시퀀스 일치도) 자동 평가.

PyPDF text layer 의 단어 시퀀스를 baseline 으로 하여,
각 OCR 모델의 clean 텍스트가 같은 순서로 단어를 잡았는지 측정.

알고리즘: Python difflib.SequenceMatcher.ratio()
  - Ratcliff-Obershelp 패턴 매칭
  - 0~1 범위 (1=완전 동일, 0=완전 다름)
  - 단어 단위 비교 (공백 split)

근거:
  - 글자수가 같아도 단어 순서가 흐트러지면 RAG 청크 의미 깨짐
  - 표/그림 페이지에서 도식 라벨이 본문 사이에 끼어드는 케이스 정량화

산출:
  results/ocr_poc/sequence_eval.json
    - 모델별 평균 시퀀스 일치도 (raw / clean 둘 다)
    - 페이지별 상세

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/24_sequence_eval.py
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import results_dir, baseline_dir, pages_dir

RESULTS_DIR = results_dir()
BASELINE_DIR = baseline_dir()
SAMPLES_JSON = BASELINE_DIR / "samples.json"
PYPDF_PAGES = pages_dir()
OUT_JSON = RESULTS_DIR / "sequence_eval.json"

MODELS = ["pypdf", "easyocr", "paddleocr", "tesseract", "surya", "marker", "mineru", "donut"]


def load_text(model: str, page: int, clean: bool = False) -> str:
    if model == "pypdf":
        f = PYPDF_PAGES / f"page_{page:03d}.txt"
        if clean:
            cf = RESULTS_DIR / "pypdf_clean" / f"page_{page:03d}.txt"
            if cf.exists():
                f = cf
    else:
        if clean:
            f = RESULTS_DIR / f"{model}_clean" / f"page_{page:03d}.txt"
        else:
            f = RESULTS_DIR / model / f"page_{page:03d}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def word_similarity(a: str, b: str) -> float | None:
    """단어 시퀀스 유사도 (0~1, 1=완전 일치)."""
    a_words = a.split()
    b_words = b.split()
    if not a_words or not b_words:
        return None
    return round(SequenceMatcher(None, a_words, b_words).ratio(), 4)


def main():
    if not SAMPLES_JSON.exists():
        print(f"❌ {SAMPLES_JSON} 없음.")
        sys.exit(1)

    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    pages = [s["page"] for s in samples]
    page_cat = {s["page"]: s["category"] for s in samples}

    print(f"=== 시퀀스 일치도 평가 (vs PyPDF baseline, clean text 비교) ===\n")
    print(f"  대상: {len(MODELS)} 모델 × {len(pages)} 페이지\n")

    results = {}

    for m in MODELS:
        per_page = []
        cat_scores = {"weak": [], "table_suspect": [], "normal": []}
        all_scores = []

        for p in pages:
            pypdf_clean = load_text("pypdf", p, clean=True)
            model_clean = load_text(m, p, clean=True)
            cat = page_cat[p]

            sim = word_similarity(pypdf_clean, model_clean)
            per_page.append({
                "page": p, "category": cat, "sequence_similarity": sim,
                "pypdf_words": len(pypdf_clean.split()),
                "model_words": len(model_clean.split()),
            })
            if sim is not None:
                all_scores.append(sim)
                cat_scores[cat].append(sim)

        def avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else None

        results[m] = {
            "avg_sequence_similarity": avg(all_scores),
            "by_category": {c: avg(v) for c, v in cat_scores.items()},
            "n_pages_compared": len(all_scores),
            "per_page": per_page,
        }

        cb = results[m]["by_category"]
        print(f"  · {m:<10}  overall={avg(all_scores):.3f}  "
              f"weak={cb['weak'] if cb['weak'] is not None else '-':.3f}  " if cb['weak'] is not None
              else f"  · {m:<10}  overall={(avg(all_scores) or 0):.3f}  weak=-  ")
        # 한 줄 다시 단순 출력
    print()
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 깔끔한 요약 표
    print(f"{'모델':<12} {'전체평균':>10} {'TYPE_A_저텍스트':>16} {'TYPE_B_표·그림':>16} {'TYPE_C_본문':>14}")
    print("-" * 72)
    sorted_models = sorted(results.items(),
                           key=lambda kv: -(kv[1]["avg_sequence_similarity"] or 0))

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  -  "

    for m, r in sorted_models:
        cb = r["by_category"]
        print(f"  {m:<10} {fmt(r['avg_sequence_similarity']):>9} "
              f"{fmt(cb['weak']):>14} {fmt(cb['table_suspect']):>14} {fmt(cb['normal']):>12}")

    print(f"\n✅ {OUT_JSON}")


if __name__ == "__main__":
    main()
