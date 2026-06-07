"""
OCR POC — TrOCR (Korean).

Microsoft 의 transformer 기반 OCR. 한국어 fine-tune 모델 사용.
설치: transformers 이미 있음 (HuggingFace 모델 자동 다운로드 ~300MB)

주의: TrOCR 은 본디 한 줄 단위 인식용 — 페이지 통째 입력은 부정확.
      여기서는 단순 비교용으로 페이지 전체를 한 번에 줌. 표·다단 페이지는 약함.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_model

# 한국어 fine-tune 모델 후보 (다운로드 시도 순)
MODEL_CANDIDATES = [
    "team-lucid/trocr-small-korean",       # 가장 작음 ~80MB
    "ddobokki/ko-trocr",                   # ~330MB
    "microsoft/trocr-base-printed",        # 영문 폴백
]


def main():
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    processor = None
    used = None
    for name in MODEL_CANDIDATES:
        try:
            print(f"[trocr] '{name}' 로드 시도...")
            processor = TrOCRProcessor.from_pretrained(name)
            model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval()
            used = name
            print(f"  → 성공 (device={device})")
            break
        except Exception as e:
            print(f"  → 실패: {type(e).__name__}: {e}")

    if model is None:
        print("❌ 모든 TrOCR 후보 로드 실패")
        sys.exit(1)

    print(f"[trocr] 사용 모델: {used}\n")

    def extract(img_path: Path) -> str:
        img = Image.open(str(img_path)).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=256)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    run_model("trocr", extract)


if __name__ == "__main__":
    main()
