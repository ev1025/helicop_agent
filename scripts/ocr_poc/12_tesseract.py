"""
OCR POC — Tesseract 5.

선행: Tesseract 5 binary 설치 필요 (Windows: https://github.com/UB-Mannheim/tesseract/wiki)
       한국어 traineddata 포함 설치 (또는 별도 다운로드 후 tessdata 폴더에).
설치: pip install pytesseract
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_model


def main():
    # 사용자 홈에 받은 tessdata (kor.traineddata + eng.traineddata) 사용.
    # C:\Program Files\Tesseract-OCR\tessdata 는 쓰기 권한 없어서 kor 못 넣음.
    user_tessdata = os.path.join(os.path.expanduser("~"), "tessdata")
    if os.path.exists(os.path.join(user_tessdata, "kor.traineddata")):
        os.environ["TESSDATA_PREFIX"] = user_tessdata

    import pytesseract
    from PIL import Image

    # Windows 기본 설치 경로 시도 (없으면 PATH 의존)
    win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_default):
        pytesseract.pytesseract.tesseract_cmd = win_default

    try:
        ver = pytesseract.get_tesseract_version()
        print(f"[tesseract] v{ver}")
        langs = pytesseract.get_languages()
        print(f"  사용 가능 언어: {langs}")
        if "kor" not in langs:
            print("⚠️  'kor' traineddata 없음 — 한국어 인식 안 됨. https://github.com/tesseract-ocr/tessdata 에서 kor.traineddata 다운로드.")
    except Exception as e:
        print(f"❌ Tesseract binary 찾을 수 없음: {e}")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)

    def extract(img_path: Path) -> str:
        img = Image.open(str(img_path))
        return pytesseract.image_to_string(img, lang="kor+eng")

    run_model("tesseract", extract)


if __name__ == "__main__":
    main()
