"""
OCR POC — Marker (PDF → markdown, surya 기반).

페이지별 비교를 위해 샘플 21장을 1페이지짜리 PDF 21개로 분리해서
각각 Marker 에 통과시킨다 (모델 로드는 1번만).

설치: pip install marker-pdf
모델 자동 다운로드 (~2GB).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PDF_PATH = ROOT / "조종사표준교재(비행이론_헬리콥터).pdf"
SAMPLES_JSON = ROOT / "results" / "ocr_poc" / "baseline" / "samples.json"
OUT_DIR = ROOT / "results" / "ocr_poc" / "marker"


def main():
    import fitz

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    pages = [s["page"] for s in samples]

    # ── 페이지별 1장 PDF 21개 생성 ─────────────────────
    tmp_dir = OUT_DIR / "_tmp_pdfs"
    tmp_dir.mkdir(exist_ok=True)
    src = fitz.open(str(PDF_PATH))
    page_pdfs = {}
    for p in pages:
        out = tmp_dir / f"page_{p:03d}.pdf"
        if not out.exists():
            dst = fitz.open()
            dst.insert_pdf(src, from_page=p, to_page=p)
            dst.save(str(out))
            dst.close()
        page_pdfs[p] = out
    src.close()
    print(f"[marker] 페이지별 PDF 21개 생성: {tmp_dir.name}\n")

    # ── Marker 모델 로드 (1번만) ──────────────────────
    print("[marker] 모델 로드 중 (최초엔 ~2GB 다운로드)...")
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        converter = PdfConverter(artifact_dict=create_model_dict())
        new_api = True
    except ImportError:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
        models = load_all_models()
        new_api = False

    # ── 페이지별 변환 ─────────────────────────────────
    timings = []
    total_chars = 0
    total_sec = 0.0
    for p in pages:
        t0 = time.perf_counter()
        try:
            if new_api:
                rendered = converter(str(page_pdfs[p]))
                text, _, _ = text_from_rendered(rendered)
            else:
                text, _, _ = convert_single_pdf(
                    str(page_pdfs[p]), models, langs=["Korean", "English"]
                )
        except Exception as e:
            text = ""
            print(f"  · page {p:>3}: 실패 — {type(e).__name__}: {e}")
        elapsed = round(time.perf_counter() - t0, 2)
        total_sec += elapsed
        n = len(text)
        total_chars += n
        (OUT_DIR / f"page_{p:03d}.txt").write_text(text, encoding="utf-8")
        timings.append({"page": p, "seconds": elapsed, "chars": n})
        print(f"  · page {p:>3}: {elapsed:>5}s  ({n:>5}자)")

    summary = {
        "model": "marker",
        "n_pages": len(pages),
        "n_failed": sum(1 for t in timings if t["chars"] == 0 and pages.index(t["page"]) > 5),
        "total_chars": total_chars,
        "total_seconds": round(total_sec, 2),
        "avg_seconds_per_page": round(total_sec / max(1, len(pages)), 2),
        "timings": timings,
    }
    (OUT_DIR / "timings.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  총 처리: {summary['total_seconds']}s ({summary['avg_seconds_per_page']}s/page)")
    print(f"  총 추출: {total_chars}자")
    print(f"  → {OUT_DIR}")


if __name__ == "__main__":
    main()
