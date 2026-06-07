"""
OCR POC — PaddleOCR (PP-OCRv4).

설치: pip install paddleocr paddlepaddle
한국어 모델 자동 다운로드 (~10MB).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_model


def main():
    # oneDNN 끄기 — PIR 실행기와 oneDNN backend 가 안 맞아 일부 op 에서 NotImplementedError
    import os
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"

    from paddleocr import PaddleOCR
    print("[paddleocr] PP-OCR 로드 중 (한국어, GPU 또는 CPU)...")
    # PaddleOCR 3.x 신 API
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang="korean", device="gpu", enable_mkldnn=False)
    except (TypeError, ValueError):
        try:
            ocr = PaddleOCR(use_textline_orientation=True, lang="korean", device="gpu")
        except (TypeError, ValueError):
            ocr = PaddleOCR(use_angle_cls=True, lang="korean", use_gpu=True)

    def extract(img_path: Path) -> str:
        # 3.x 는 predict() 권장, 구버전은 ocr()
        try:
            result = ocr.predict(str(img_path))
        except AttributeError:
            result = ocr.ocr(str(img_path), cls=True)
        if not result:
            return ""
        # 3.x predict 출력: [{"rec_texts": [...], "rec_scores": [...], ...}]
        # 구 ocr 출력:      [[[bbox, (text, conf)], ...]]
        first = result[0]
        if isinstance(first, dict) and "rec_texts" in first:
            return "\n".join(first["rec_texts"])
        if isinstance(first, list):
            return "\n".join(item[1][0] for item in first if item and item[1])
        return ""

    run_model("paddleocr", extract)


if __name__ == "__main__":
    main()
