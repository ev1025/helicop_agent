"""
OCR POC 공통 설정.

다중 PDF 지원 구조 (Phase 5):
  data/pdfs/<책이름>.pdf              ← 입력
  results/ocr_poc/pdfs/<책이름>/...   ← PDF별 결과 (격리)
  results/ocr_poc/dashboard.html      ← 전체 대시보드 (모든 PDF 탭)

사용:
  - 단일 PDF 처리 (env 미지정): data/pdfs 의 첫 번째 PDF
  - 특정 PDF 처리: OCR_POC_PDF_STEM 환경변수에 stem 지정
    (예: OCR_POC_PDF_STEM=조종사표준교재(비행이론_헬리콥터))
  - 마스터 러너 (00_run_all.py)가 PDF마다 env 세팅 후 각 스크립트 순차 실행.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "pdfs"
RESULTS_ROOT = ROOT / "results" / "ocr_poc"
PDFS_RESULTS = RESULTS_ROOT / "pdfs"
DASHBOARD_HTML = RESULTS_ROOT / "dashboard.html"

ENV_KEY = "OCR_POC_PDF_STEM"


def list_pdfs() -> List[Path]:
    """data/pdfs/*.pdf 정렬 반환."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pdf"))


def list_pdf_stems() -> List[str]:
    """data/pdfs 의 PDF stem 목록."""
    return [p.stem for p in list_pdfs()]


def current_pdf() -> Path:
    """현재 처리 중인 PDF 경로.

    OCR_POC_PDF_STEM 환경변수가 있으면 그 stem 의 PDF, 없으면 첫 번째 PDF.
    """
    env = os.environ.get(ENV_KEY)
    pdfs = list_pdfs()
    if not pdfs:
        raise FileNotFoundError(
            f"❌ {DATA_DIR} 에 PDF 가 없습니다. data/pdfs/*.pdf 를 넣고 다시 실행하세요."
        )
    if env:
        for p in pdfs:
            if p.stem == env:
                return p
        raise FileNotFoundError(
            f"❌ {ENV_KEY}={env} 에 해당하는 PDF 가 {DATA_DIR} 에 없습니다."
        )
    return pdfs[0]


def current_stem() -> str:
    return current_pdf().stem


def results_dir(stem: str | None = None) -> Path:
    """PDF stem 에 대응하는 결과 폴더."""
    s = stem or current_stem()
    d = PDFS_RESULTS / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def baseline_dir(stem: str | None = None) -> Path:
    return results_dir(stem) / "baseline"


def images_dir(stem: str | None = None) -> Path:
    return results_dir(stem) / "images"


def pages_dir(stem: str | None = None) -> Path:
    """baseline/pages/ — PyPDF 페이지별 원문 (디버깅 + Phase 2 비교용)."""
    return baseline_dir(stem) / "pages"
