"""
OCR POC — EasyOCR.

설치: pip install easyocr
한국어 모델 자동 다운로드 (~64MB).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_model


def main():
    import easyocr
    print("[easyocr] reader 로드 중 (ko + en, GPU)...")
    reader = easyocr.Reader(["ko", "en"], gpu=True)

    def extract(img_path: Path) -> str:
        # detail=0 → 텍스트만 리스트로 반환 (좌표·confidence 제외)
        results = reader.readtext(str(img_path), detail=0, paragraph=True)
        return "\n".join(results) if results else ""

    run_model("easyocr", extract)


if __name__ == "__main__":
    main()
