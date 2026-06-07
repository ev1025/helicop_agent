"""
OCR POC — MinerU (magic-pdf).

OCR + layout + 표 + 수식 + 이미지 통합 추출.
설치: pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
모델 자동 다운로드 (~3GB 첫 실행 시).

주의: 첫 실행 시 magic-pdf.json 생성 + 모델 다운로드가 자동으로 진행됨.
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
OUT_DIR = ROOT / "results" / "ocr_poc" / "mineru"


def main():
    import fitz

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    pages = [s["page"] for s in samples]

    # ── 페이지별 1장 PDF 21개 (Marker 와 공유 가능하나 격리) ─
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
    print(f"[mineru] 페이지별 PDF 21개 생성\n")

    # ── magic-pdf 모듈 로드 ────────────────────────────
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
    except ImportError as e:
        print(f"❌ magic-pdf import 실패: {e}")
        sys.exit(1)

    local_image_dir = OUT_DIR / "_images"
    local_image_dir.mkdir(exist_ok=True)
    image_writer = FileBasedDataWriter(str(local_image_dir))

    # ── 페이지별 변환 ──────────────────────────────────
    timings = []
    total_chars = 0
    total_sec = 0.0
    for p in pages:
        t0 = time.perf_counter()
        text = ""
        try:
            pdf_bytes = Path(page_pdfs[p]).read_bytes()
            ds = PymuDocDataset(pdf_bytes)
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)
            text = pipe_result.get_markdown(str(local_image_dir))
        except Exception as e:
            print(f"  · page {p:>3}: 실패 — {type(e).__name__}: {e}")
        elapsed = round(time.perf_counter() - t0, 2)
        total_sec += elapsed
        n = len(text)
        total_chars += n
        (OUT_DIR / f"page_{p:03d}.txt").write_text(text, encoding="utf-8")
        timings.append({"page": p, "seconds": elapsed, "chars": n})
        print(f"  · page {p:>3}: {elapsed:>5}s  ({n:>5}자)")

    summary = {
        "model": "mineru",
        "n_pages": len(pages),
        "n_failed": sum(1 for t in timings if t["chars"] == 0),
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
