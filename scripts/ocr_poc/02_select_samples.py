"""
OCR POC Phase 1.5 — 샘플 페이지 선정.

01_pypdf_baseline.py 결과(per_page.json) 기반으로 Phase 2 비교 대상 페이지를 자동 선정한다.

선정 기준:
  - 약한 페이지 8장   : 글자수<100 또는 한글비율<5% (전부)
  - 표/그림 의심 10장 : small bucket(100-499) 중 한글비율 낮은 순 top-10
  - 정상 대조군 3장   : medium bucket(500-1499) 중 한글비율 50%+ 에서 균등 샘플

결과:
  results/ocr_poc/baseline/samples.json
    [{"page": 27, "chars": 0, "kor_ratio": 0.0, "category": "weak"}, ...]

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/02_select_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import current_pdf, baseline_dir, results_dir

BASELINE_DIR = baseline_dir()
PER_PAGE = BASELINE_DIR / "per_page.json"
SAMPLES = BASELINE_DIR / "samples.json"

N_WEAK_LIMIT = None        # None = 전부
N_TABLE_SUSPECT = 10
N_NORMAL_CONTROL = 3


def is_truly_blank(pdf_path: str, page_idx: int, ocr_dirs: list[Path] | None = None) -> bool:
    """PDF + OCR 결과 모두에서 텍스트가 없으면 진짜 백지로 판정.

    1. PyMuPDF: 텍스트 0자 + 이미지 0개 → 백지
    2. OCR 결과 디렉토리들이 존재하면, 모든 OCR 모델이 그 페이지에서 0자 추출
       → 백지 (워터마크 등 시각적 무의미 이미지만 있는 경우)
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        has_text = bool(page.get_text().strip())
        has_image = len(page.get_images()) > 0
        if not has_text and not has_image:
            return True
    finally:
        doc.close()

    # OCR 결과까지 검증 (있을 때만)
    if ocr_dirs:
        for d in ocr_dirs:
            f = d / f"page_{page_idx:03d}.txt"
            if f.exists() and f.read_text(encoding="utf-8").strip():
                return False  # 한 모델이라도 텍스트 뽑았으면 백지 아님
        return True  # 모든 OCR 도 0자 = 백지 (시각만 있고 의미 없음)

    return False


def main():
    if not PER_PAGE.exists():
        print(f"❌ {PER_PAGE} 없음. 먼저 01_pypdf_baseline.py 실행.")
        sys.exit(1)

    per_page = json.loads(PER_PAGE.read_text(encoding="utf-8"))

    # ── 1) 약한 페이지 (전부, 단 진짜 백지는 자동 제외) ─────────
    pdf_path = current_pdf()
    base = results_dir()
    OCR_DIRS = [base / m
                for m in ["easyocr", "paddleocr", "tesseract", "surya", "marker", "mineru", "donut"]]
    OCR_DIRS = [d for d in OCR_DIRS if d.exists()]

    weak_candidates = [p for p in per_page if p["weak"]]
    weak = []
    blank_pages = []
    for p in weak_candidates:
        if is_truly_blank(str(pdf_path), p["page"], OCR_DIRS):
            blank_pages.append(p["page"])
        else:
            weak.append(p)
    if blank_pages:
        print(f"  · 진짜 백지(텍스트·이미지 없음) {len(blank_pages)}장 자동 제외: {blank_pages}")
    if N_WEAK_LIMIT is not None:
        weak = weak[:N_WEAK_LIMIT]

    # ── 2) 표/그림 의심 (small bucket, 한글비율 낮은 순) ────────
    small = [
        p for p in per_page
        if 100 <= p["chars"] < 500 and not p["weak"]
    ]
    small.sort(key=lambda p: (p["kor_ratio"], p["chars"]))  # 한글비율 ↑, 글자수 ↑ 순
    table_suspect = small[:N_TABLE_SUSPECT]

    # ── 3) 정상 대조군 (medium bucket, 균등 샘플) ──────────────
    medium = [
        p for p in per_page
        if 500 <= p["chars"] < 1500 and p["kor_ratio"] >= 0.5
    ]
    # 균등 샘플 — 책 전반에서 골고루
    if len(medium) >= N_NORMAL_CONTROL:
        step = len(medium) // N_NORMAL_CONTROL
        normal = [medium[i * step] for i in range(N_NORMAL_CONTROL)]
    else:
        normal = medium

    # 카테고리 라벨 붙여 통합
    samples = []
    seen = set()
    for cat, lst in [("weak", weak), ("table_suspect", table_suspect), ("normal", normal)]:
        for p in lst:
            if p["page"] in seen:
                continue
            seen.add(p["page"])
            samples.append({
                "page": p["page"],
                "chars": p["chars"],
                "kor_ratio": p["kor_ratio"],
                "category": cat,
            })

    samples.sort(key=lambda p: p["page"])

    # 저장
    SAMPLES.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 콘솔 출력
    print(f"=== 샘플 페이지 선정: {len(samples)}장 ===\n")
    counts = {"weak": 0, "table_suspect": 0, "normal": 0}
    for s in samples:
        counts[s["category"]] += 1
        cat_short = {"weak": "약함  ", "table_suspect": "표/그림", "normal": "정상  "}[s["category"]]
        print(f"  · [{cat_short}] page {s['page']:>3}: chars={s['chars']:>4} kor={s['kor_ratio']:.2%}")

    print(f"\n분포: 약함 {counts['weak']} / 표·그림 {counts['table_suspect']} / 정상 {counts['normal']}")
    print(f"\n✅ {SAMPLES}")


if __name__ == "__main__":
    main()
