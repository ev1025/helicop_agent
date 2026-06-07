"""
OCR POC Phase 1.6 — 샘플 페이지를 PNG 이미지로 추출.

samples.json 의 페이지들을 PyMuPDF(fitz) 로 PNG 변환. OCR 모델들이 이미지 입력을 받으므로.

해상도: 300 DPI (OCR 표준).

결과:
  results/ocr_poc/images/page_NNN.png  (21장)

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/03_extract_images.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import current_pdf, baseline_dir, images_dir

PDF_PATH = current_pdf()
SAMPLES = baseline_dir() / "samples.json"
IMG_DIR = images_dir()

DPI = 300  # OCR 표준


def main():
    if not SAMPLES.exists():
        print(f"❌ {SAMPLES} 없음. 먼저 02_select_samples.py 실행.")
        sys.exit(1)

    import fitz

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    pages = [s["page"] for s in samples]

    print(f"=== Phase 1.6 — 이미지 추출 ===")
    print(f"  PDF: {PDF_PATH.name}")
    print(f"  대상: {len(pages)}장 (page {pages[0]} ~ {pages[-1]})")
    print(f"  해상도: {DPI} DPI\n")

    doc = fitz.open(str(PDF_PATH))
    zoom = DPI / 72  # PDF default = 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    t0 = time.perf_counter()
    for p in pages:
        page = doc[p]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = IMG_DIR / f"page_{p:03d}.png"
        pix.save(str(out_path))
        kb = out_path.stat().st_size / 1024
        print(f"  · page {p:>3} → {out_path.name}  ({pix.width}x{pix.height}, {kb:.0f}KB)")

    doc.close()
    elapsed = time.perf_counter() - t0

    print(f"\n✅ {len(pages)}장 저장 ({elapsed:.1f}초)")
    print(f"   {IMG_DIR}")


if __name__ == "__main__":
    main()
