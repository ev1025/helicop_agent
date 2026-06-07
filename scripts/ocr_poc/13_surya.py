"""
OCR POC — Surya OCR.

신세대 transformer 기반, layout-aware, 90+ 언어 (한국어 포함).
설치: pip install surya-ocr
모델 자동 다운로드 (~500MB).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_model


def main():
    from PIL import Image

    # surya API 는 버전마다 다름.
    new_api = None
    try:
        # 최신 (>=0.8): FoundationPredictor 필요
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        foundation = FoundationPredictor()
        rec = RecognitionPredictor(foundation)
        det = DetectionPredictor()
        new_api = "v2"
    except ImportError:
        try:
            # 중간 (~0.5): predictor 인자 없이
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            det = DetectionPredictor()
            rec = RecognitionPredictor()
            new_api = "v1"
        except ImportError:
            # 구 API
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            det_model, det_proc = load_det_model(), load_det_processor()
            rec_model, rec_proc = load_rec_model(), load_rec_processor()
            new_api = "old"

    print(f"[surya] 로드 완료 (new_api={new_api})")

    def extract(img_path: Path) -> str:
        img = Image.open(str(img_path)).convert("RGB")
        if new_api in ("v1", "v2"):
            # surya v2: task_names 는 string 리스트 ("ocr_with_boxes" 등),
            # langs 인자는 deprecated. det_predictor 만 넘기면 자동 detection.
            result = rec([img], det_predictor=det)
        else:
            from surya.ocr import run_ocr
            result = run_ocr(
                [img], [["ko", "en"]],
                det_model, det_proc, rec_model, rec_proc,
            )
        if not result:
            return ""
        page = result[0]
        lines = []
        for line in getattr(page, "text_lines", []):
            t = getattr(line, "text", None)
            if t:
                lines.append(t)
        return "\n".join(lines)

    run_model("surya", extract)


if __name__ == "__main__":
    main()
