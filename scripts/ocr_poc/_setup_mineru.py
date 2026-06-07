"""
MinerU (magic-pdf) 모델 자동 다운로드 + ~/magic-pdf.json 생성.

opendatalab/PDF-Extract-Kit-1.0 + hantian/layoutreader 다운로드 (~3GB).
첫 실행만 필요.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIG_PATH = Path.home() / "magic-pdf.json"


def main():
    # Windows symlink 권한 회피 — local_dir 지정 + 환경변수
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from huggingface_hub import snapshot_download

    MODELS_ROOT = Path.home() / "mineru_models"
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1. PDF-Extract-Kit (layout, MFD, MFR, OCR, table 모델) ──
    print("=== MinerU 모델 다운로드 ===")
    print("[1/2] opendatalab/PDF-Extract-Kit-1.0 ...")
    extract_kit_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    extract_kit_local = MODELS_ROOT / "PDF-Extract-Kit-1.0"
    snapshot_download(
        "opendatalab/PDF-Extract-Kit-1.0",
        allow_patterns=extract_kit_patterns,
        local_dir=str(extract_kit_local),
        local_dir_use_symlinks=False,
    )
    extract_kit_dir = str(extract_kit_local / "models")
    print(f"  → {extract_kit_dir}")

    # ── 2. LayoutReader ──
    print("\n[2/2] hantian/layoutreader ...")
    layoutreader_local = MODELS_ROOT / "layoutreader"
    snapshot_download(
        "hantian/layoutreader",
        allow_patterns=["*.json", "*.safetensors"],
        local_dir=str(layoutreader_local),
        local_dir_use_symlinks=False,
    )
    layoutreader_dir = str(layoutreader_local)
    print(f"  → {layoutreader_dir}")

    # ── 3. magic-pdf.json 생성 ──
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "bucket_info": {},
        "models-dir": extract_kit_dir,
        "layoutreader-model-dir": layoutreader_dir,
        "device-mode": device,
        "layout-config": {"model": "doclayout_yolo"},
        "formula-config": {
            "mfd_model": "yolo_v8_mfd",
            "mfr_model": "unimernet_small",
            "enable": False,  # 수식 OFF (속도)
        },
        "table-config": {
            "model": "rapid_table",
            "sub_model": "slanet_plus",
            "enable": True,
            "max_time": 400,
        },
        "config_version": "1.2.1",
    }

    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"\n✅ 설정 파일 생성: {CONFIG_PATH}")
    print(f"   device={device}, models={extract_kit_dir}")


if __name__ == "__main__":
    main()
