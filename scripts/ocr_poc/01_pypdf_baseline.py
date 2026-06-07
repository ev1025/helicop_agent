"""
OCR POC Phase 1 — PyPDF 베이스라인 진단.

매뉴얼 PDF 의 모든 페이지를 PyPDFLoader 로 추출해서:
  - 페이지별: 글자수 / 한글비율 / 숫자비율 / 공백비율
  - 글자수 분포 (히스토그램용)
  - 약한 페이지 자동 식별 (글자수 100 미만 또는 한글비율 0.05 미만)

결과:
  results/ocr_poc/baseline/per_page.json   — 페이지별 raw 데이터
  results/ocr_poc/baseline/summary.json    — 전체 통계 + 약한 페이지 목록
  results/ocr_poc/baseline/pages/*.txt     — 페이지별 원문 (디버깅용)

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/01_pypdf_baseline.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import current_pdf, baseline_dir, pages_dir

PDF_PATH = current_pdf()
OUT_DIR = baseline_dir()
PAGES_DIR = pages_dir()

# 약한 페이지 임계
MIN_CHARS = 100        # 글자수 100 미만 → 표/이미지 의심
MIN_KOR_RATIO = 0.05   # 한글비율 5% 미만 → 본문 손실 의심


_HANGUL_RE = re.compile(r"[가-힯㄰-㆏]")
_DIGIT_RE = re.compile(r"\d")
_WS_RE = re.compile(r"\s")


def char_stats(text: str) -> dict:
    """페이지 텍스트의 글자 구성 비율."""
    n = len(text)
    if n == 0:
        return {"chars": 0, "kor_ratio": 0.0, "digit_ratio": 0.0, "ws_ratio": 0.0}
    kor = len(_HANGUL_RE.findall(text))
    digit = len(_DIGIT_RE.findall(text))
    ws = len(_WS_RE.findall(text))
    return {
        "chars": n,
        "kor_ratio": round(kor / n, 4),
        "digit_ratio": round(digit / n, 4),
        "ws_ratio": round(ws / n, 4),
    }


def main():
    if not PDF_PATH.exists():
        print(f"❌ PDF 없음: {PDF_PATH}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    from langchain_community.document_loaders import PyPDFLoader

    print(f"=== Phase 1 — PyPDF baseline ===")
    print(f"  PDF: {PDF_PATH.name}")

    t0 = time.perf_counter()
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    elapsed = time.perf_counter() - t0
    print(f"  → {len(docs)} 페이지 추출 ({elapsed:.1f}초)\n")

    per_page = []
    for i, doc in enumerate(docs):
        page_no = doc.metadata.get("page", i)  # 0-based
        text = doc.page_content or ""
        stats = char_stats(text)
        stats["page"] = page_no
        stats["weak"] = (
            stats["chars"] < MIN_CHARS
            or stats["kor_ratio"] < MIN_KOR_RATIO
        )
        per_page.append(stats)

        # 페이지별 원문 저장 (디버깅 + Phase 2 비교용)
        (PAGES_DIR / f"page_{page_no:03d}.txt").write_text(text, encoding="utf-8")

    # 통계 요약
    chars = [p["chars"] for p in per_page]
    kor_ratios = [p["kor_ratio"] for p in per_page]
    weak_pages = [p for p in per_page if p["weak"]]

    # 분포 버킷 (Phase 2 에서 다양한 페이지 샘플링 위해)
    buckets = {
        "empty (0)": sum(1 for c in chars if c == 0),
        "tiny (1-99)": sum(1 for c in chars if 1 <= c < 100),
        "small (100-499)": sum(1 for c in chars if 100 <= c < 500),
        "medium (500-1499)": sum(1 for c in chars if 500 <= c < 1500),
        "large (1500+)": sum(1 for c in chars if c >= 1500),
    }

    summary = {
        "pdf_path": str(PDF_PATH.name),
        "total_pages": len(per_page),
        "extraction_seconds": round(elapsed, 2),
        "avg_chars_per_page": round(sum(chars) / len(chars), 1) if chars else 0,
        "median_chars": sorted(chars)[len(chars) // 2] if chars else 0,
        "min_chars": min(chars) if chars else 0,
        "max_chars": max(chars) if chars else 0,
        "avg_kor_ratio": round(sum(kor_ratios) / len(kor_ratios), 4) if kor_ratios else 0,
        "char_buckets": buckets,
        "weak_page_count": len(weak_pages),
        "weak_threshold": {"min_chars": MIN_CHARS, "min_kor_ratio": MIN_KOR_RATIO},
        "weak_pages": [
            {"page": p["page"], "chars": p["chars"], "kor_ratio": p["kor_ratio"]}
            for p in weak_pages
        ],
    }

    # 저장
    (OUT_DIR / "per_page.json").write_text(
        json.dumps(per_page, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 콘솔 출력
    print("=== 글자수 분포 ===")
    for label, n in buckets.items():
        bar = "█" * int(n / max(1, max(buckets.values())) * 40)
        print(f"  {label:<22} {n:>4} {bar}")
    print(f"\n  평균: {summary['avg_chars_per_page']}자  /  중앙값: {summary['median_chars']}자")
    print(f"  최소: {summary['min_chars']}자  /  최대: {summary['max_chars']}자")
    print(f"  평균 한글비율: {summary['avg_kor_ratio']:.1%}")

    print(f"\n=== 약한 페이지 (글자<{MIN_CHARS} 또는 한글비율<{MIN_KOR_RATIO}): {len(weak_pages)}장 ===")
    for p in weak_pages[:20]:
        print(f"  · page {p['page']:>3}: chars={p['chars']:>4} kor={p['kor_ratio']:.2%}")
    if len(weak_pages) > 20:
        print(f"  · ... +{len(weak_pages) - 20}장 더 (전체는 summary.json 참고)")

    print(f"\n✅ 저장 위치:")
    print(f"   {OUT_DIR / 'summary.json'}")
    print(f"   {OUT_DIR / 'per_page.json'}")
    print(f"   {PAGES_DIR}/ (페이지별 원문 {len(per_page)}개)")


if __name__ == "__main__":
    main()
