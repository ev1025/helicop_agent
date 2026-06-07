"""
OCR POC 공통 헬퍼.

각 OCR 모델 스크립트가 공유:
  - 이미지 경로 일괄 로드 (results/ocr_poc/pdfs/<책>/images/page_NNN.png)
  - 결과 저장: results/ocr_poc/pdfs/<책>/{model}/page_NNN.txt + timings.json

현재 처리 중인 PDF 는 _config.current_stem() 으로 결정.
환경변수 OCR_POC_PDF_STEM 으로 지정하거나, 미지정 시 data/pdfs/ 의 첫 PDF.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, List

from _config import images_dir, results_dir

ROOT = Path(__file__).resolve().parents[2]


def list_images() -> List[Path]:
    """현재 PDF 의 모든 page_*.png 정렬 반환."""
    return sorted(images_dir().glob("page_*.png"))


def page_no_of(p: Path) -> int:
    """page_027.png → 27"""
    return int(p.stem.split("_")[1])


def run_model(
    model_name: str,
    extract_fn: Callable[[Path], str],
    warmup: bool = True,
) -> dict:
    """
    각 모델 공통 실행 루틴.

    Args:
        model_name: 결과 저장 디렉토리 이름. e.g. "easyocr"
        extract_fn: image path 받아서 추출 text 반환하는 함수.
        warmup: True 면 첫 페이지를 한 번 더 돌려서 warmup time 분리.

    Returns:
        summary dict (timings, totals)
    """
    out_dir = results_dir() / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images()
    if not images:
        print(f"❌ {images_dir()} 에 이미지 없음. 먼저 03_extract_images.py 실행.")
        sys.exit(1)

    print(f"=== {model_name} — {len(images)} 페이지 ===\n")

    # Warmup (첫 페이지 1번 더 돌림 — 첫 추론은 모델 로드/JIT 컴파일로 느림)
    warmup_sec = None
    if warmup and images:
        t0 = time.perf_counter()
        try:
            _ = extract_fn(images[0])
            warmup_sec = round(time.perf_counter() - t0, 2)
            print(f"  [warmup] page {page_no_of(images[0])}: {warmup_sec}s\n")
        except Exception as e:
            print(f"  [warmup] 실패: {e}\n")

    timings = []
    failed = []
    total_chars = 0

    for img in images:
        p = page_no_of(img)
        t0 = time.perf_counter()
        try:
            text = extract_fn(img)
            elapsed = round(time.perf_counter() - t0, 2)
            (out_dir / f"page_{p:03d}.txt").write_text(text or "", encoding="utf-8")
            n = len(text or "")
            total_chars += n
            timings.append({"page": p, "seconds": elapsed, "chars": n})
            print(f"  · page {p:>3}: {elapsed:>5}s  ({n:>5}자)")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failed.append({"page": p, "error": err})
            (out_dir / f"page_{p:03d}.txt").write_text("", encoding="utf-8")
            timings.append({"page": p, "seconds": None, "chars": 0, "error": err})
            print(f"  · page {p:>3}: 실패 — {err}")

    total_sec = sum(t["seconds"] for t in timings if t.get("seconds") is not None)
    summary = {
        "model": model_name,
        "n_pages": len(images),
        "n_failed": len(failed),
        "total_chars": total_chars,
        "total_seconds": round(total_sec, 2),
        "avg_seconds_per_page": round(total_sec / max(1, len(images) - len(failed)), 2),
        "warmup_seconds": warmup_sec,
        "timings": timings,
    }
    (out_dir / "timings.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== 요약 ===")
    print(f"  총 처리: {summary['total_seconds']}s ({summary['avg_seconds_per_page']}s/page)")
    print(f"  총 추출: {total_chars}자")
    print(f"  실패: {len(failed)}/{len(images)}")
    print(f"  → {out_dir}\n")

    return summary
