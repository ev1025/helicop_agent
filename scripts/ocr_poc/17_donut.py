"""
OCR POC — Donut (Naver CLOVA, OCR-free Document Transformer).

설치: transformers 이미 있음. 한국어 모델 자동 다운로드 (~800MB).

주의: Donut 은 본디 task-specific (영수증/명함/문서 파싱) — 일반 OCR 텍스트 dump 가 아님.
      여기서는 base pre-trained 모델로 시도 (donut-base 의 unconstrained 생성).
      결과 품질 평가는 참고용.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_model

# 후보 순서: base pre-trained → 한국어 fine-tune (있으면)
MODEL_CANDIDATES = [
    "naver-clova-ix/donut-base",                       # base, OCR-free, ~800MB
    "naver-clova-ix/donut-base-finetuned-rvlcdip",     # 문서 분류 fine-tune
]


def main():
    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    processor = None
    used = None
    for name in MODEL_CANDIDATES:
        try:
            print(f"[donut] '{name}' 로드 시도...")
            processor = DonutProcessor.from_pretrained(name)
            model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval()
            used = name
            print(f"  → 성공 (device={device})")
            break
        except Exception as e:
            print(f"  → 실패: {type(e).__name__}: {e}")

    if model is None:
        print("❌ 모든 Donut 후보 로드 실패")
        sys.exit(1)

    print(f"[donut] 사용 모델: {used}\n")

    # Donut 은 task prompt 가 필수. unconstrained read 용 prompt.
    task_prompt = "<s_synthdog>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    def extract(img_path: Path) -> str:
        img = Image.open(str(img_path)).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        seq = processor.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
        # task prompt 제거
        seq = seq.replace(task_prompt, "").replace("<pad>", "").replace("</s>", "").strip()
        return seq

    run_model("donut", extract)


if __name__ == "__main__":
    main()
