"""
OCR POC 마스터 러너 (Phase 5+).

data/pdfs/*.pdf 를 모두 스캔해서, PDF 마다 풀 파이프라인을 순차 실행한다.

흐름:
  1. 01_pypdf_baseline.py  — PyPDF 전체 페이지 글자수/한글비율
  2. 02_select_samples.py  — 비교 대상 16장 자동 선정
  3. 03_extract_images.py  — 선정 페이지 PNG (300DPI)
  4. 10~17_{model}.py      — 7개 OCR 모델별 추출  ★ 시간 大
  5. 20_compare.py         — 정량 비교 종합
  6. 22_noise_eval.py      — 노이즈 비율 + clean 텍스트
  7. 24_sequence_eval.py   — 시퀀스 일치도 (vs PyPDF)
  8. 21_visual_dashboard.py — 대시보드 생성 (모든 PDF 탭 통합)

PDF 별 결과: results/ocr_poc/pdfs/<책이름>/...
대시보드:    results/ocr_poc/dashboard.html  (모든 PDF 탭으로)

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/00_run_all.py
  .venv/Scripts/python.exe scripts/ocr_poc/00_run_all.py --skip-existing
  .venv/Scripts/python.exe scripts/ocr_poc/00_run_all.py --only marker mineru
  .venv/Scripts/python.exe scripts/ocr_poc/00_run_all.py --pdf "조종사표준교재(비행이론_헬리콥터)"

옵션:
  --skip-existing  : 이미 결과가 있는 PDF/모델은 건너뛰기
  --only M1 M2 ... : 지정 모델만 실행 (기본: 7개 전부)
  --pdf STEM       : 특정 PDF stem 만 처리
  --no-ocr         : OCR 단계 건너뛰고 평가/대시보드만
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _config import DATA_DIR, list_pdfs, results_dir, ENV_KEY

PYTHON = sys.executable

ALL_MODELS = [
    ("easyocr", "10_easyocr.py"),
    ("paddleocr", "11_paddleocr.py"),
    ("tesseract", "12_tesseract.py"),
    ("surya", "13_surya.py"),
    ("marker", "15_marker.py"),
    ("mineru", "16_mineru.py"),
    ("donut", "17_donut.py"),
    # 14_trocr 는 페이지 단위 입력시 환각 — 제외
]

PRE_STEPS = [
    ("01_pypdf_baseline.py", "PyPDF 베이스라인"),
    ("02_select_samples.py", "샘플 16장 선정"),
    ("03_extract_images.py", "PNG 추출"),
]

POST_STEPS = [
    ("20_compare.py", "정량 통합 비교"),
    ("22_noise_eval.py", "노이즈 평가 + clean 텍스트"),
    ("24_sequence_eval.py", "시퀀스 일치도"),
]


def run(script: str, env: dict, label: str = "") -> bool:
    """스크립트 1개 실행. 실패 시 False."""
    print(f"\n{'─' * 70}\n▶ {label or script}\n{'─' * 70}")
    t0 = time.perf_counter()
    res = subprocess.run([PYTHON, str(SCRIPTS_DIR / script)], env=env)
    elapsed = time.perf_counter() - t0
    ok = res.returncode == 0
    flag = "✅" if ok else "❌"
    print(f"\n{flag} {script} ({elapsed:.1f}s)")
    return ok


def process_pdf(stem: str, args) -> bool:
    """PDF 1개 풀 파이프라인."""
    env = {**os.environ, ENV_KEY: stem}
    base = results_dir(stem)

    print(f"\n{'═' * 70}\n📘 {stem}\n   결과: {base}\n{'═' * 70}")

    # 전처리 (PyPDF / 샘플 선정 / 이미지 추출)
    for script, label in PRE_STEPS:
        if args.skip_existing and (base / "baseline" / "samples.json").exists() and script != "03_extract_images.py":
            print(f"  · {script} 건너뜀 (samples.json 있음)")
            continue
        if args.skip_existing and script == "03_extract_images.py":
            imgs = list((base / "images").glob("page_*.png"))
            if imgs:
                print(f"  · {script} 건너뜀 (이미지 {len(imgs)}장 있음)")
                continue
        if not run(script, env, label):
            return False

    # OCR 모델
    if not args.no_ocr:
        models = [(n, s) for n, s in ALL_MODELS if not args.only or n in args.only]
        for name, script in models:
            if args.skip_existing and (base / name / "timings.json").exists():
                print(f"  · {name} 건너뜀 (timings.json 있음)")
                continue
            if not run(script, env, f"{name} OCR"):
                print(f"  ⚠ {name} 실패 — 다음 모델로 계속")

    # 평가
    for script, label in POST_STEPS:
        if not run(script, env, label):
            return False

    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-existing", action="store_true", help="이미 결과가 있는 PDF/모델 건너뛰기")
    p.add_argument("--only", nargs="+", metavar="MODEL", help="특정 모델만 (예: --only marker mineru)")
    p.add_argument("--pdf", metavar="STEM", help="특정 PDF stem 만 처리")
    p.add_argument("--no-ocr", action="store_true", help="OCR 단계 건너뛰고 평가/대시보드만")
    return p.parse_args()


def main():
    args = parse_args()
    pdfs = list_pdfs()
    if not pdfs:
        print(f"❌ {DATA_DIR} 에 PDF 없음. data/pdfs/*.pdf 를 넣고 다시 실행하세요.")
        sys.exit(1)

    if args.pdf:
        pdfs = [p for p in pdfs if p.stem == args.pdf]
        if not pdfs:
            print(f"❌ --pdf {args.pdf} 에 해당하는 PDF 없음.")
            sys.exit(1)

    print(f"=== OCR POC 마스터 러너 ===")
    print(f"  대상: {len(pdfs)} PDF")
    for p in pdfs:
        print(f"    · {p.stem}")
    print(f"  모드: skip_existing={args.skip_existing} no_ocr={args.no_ocr}")

    ok_count = 0
    for pdf in pdfs:
        if process_pdf(pdf.stem, args):
            ok_count += 1

    # 대시보드 (모든 PDF 통합)
    print(f"\n{'═' * 70}\n📊 대시보드 생성 (모든 PDF 통합)\n{'═' * 70}")
    run("21_visual_dashboard.py", os.environ.copy(), "통합 대시보드")

    print(f"\n{'═' * 70}\n✅ 완료: {ok_count}/{len(pdfs)} PDF 처리\n{'═' * 70}")


if __name__ == "__main__":
    main()
