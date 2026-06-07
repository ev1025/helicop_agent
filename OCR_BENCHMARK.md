# 🔎 헬리콥터 매뉴얼 OCR 벤치마크 (Phase 1+2)

> **목적**: 현재 RAG 인덱싱에 쓰는 PyPDF가 표·그림 페이지에서 텍스트 손실이 있는지 확인하고, 로컬에서 돌릴 수 있는 OCR/Document 모델 8종을 같은 페이지에 돌려 비교한다.
> **상태**: Phase 1+2 완료 (정량 비교). Phase 3 (정성/CER 평가) 대기.
> **데이터**: `조종사표준교재(비행이론_헬리콥터).pdf` 229장 중 샘플 21장.

---

## 1. 실험 설계

### 1.1 샘플 선정 (21장)

`01_pypdf_baseline.py` → `02_select_samples.py` 로 자동 선정:

| 카테고리 | 페이지 수 | 기준 | 페이지 예 |
|---|---:|---|---|
| **약함 (weak)** | 8 | PyPDF 글자수 < 100 또는 한글비율 < 5% | 0(표지), 1, 27, 131, 189, 227, 228 (구분지·뒷표지), 2 (목차) |
| **표·그림 의심** | 10 | small bucket(100-499자) 중 한글비율 낮은 순 | 28, 44, 54, 109, 123, 140, 172, 188, 190, 226 |
| **정상 본문** | 3 | medium bucket(500-1499자) 한글비율 50%+, 균등 분포 | 3, 76, 155 |

이미지 추출: `03_extract_images.py` (PyMuPDF, 300 DPI) → `results/ocr_poc/images/page_NNN.png`

### 1.2 OCR 모델 8종 (모두 로컬 GPU/CPU)

| # | 모델 | 분류 | 모델 크기 | 디바이스 (이번 실행) |
|---|---|---|---|---|
| 1 | **EasyOCR** | 인식 단독 | ~64MB | GPU |
| 2 | **PaddleOCR** (PP-OCRv3) | 인식 단독 | ~10MB | CPU (Windows GPU 호환 이슈) |
| 3 | **Tesseract 5** | 인식 단독 (전통) | ~30MB | CPU |
| 4 | **Surya** | 인식 + Layout (transformer) | ~500MB | GPU |
| 5 | **TrOCR Korean** (`team-lucid/trocr-small-korean`) | 인식 (single-line) | ~80MB | CPU |
| 6 | **Donut** (`naver-clova-ix/donut-base`) | Document Transformer | ~800MB | CPU |
| 7 | **Marker** | PDF→Markdown (Surya 기반) | ~2GB | GPU |
| 8 | **MinerU** (magic-pdf 1.3) | OCR + Layout + 표 + 수식 | ~3GB | CPU |

베이스라인: **PyPDF** (현재 RAG 인덱싱에 사용 중)

---

## 2. 결과 — 정량 비교

### 2.1 모델별 총량

| 모델 | 총 글자 | vs PyPDF | 평균 한글% | 비빈 페이지 | avg s/page |
|---|---:|---:|---:|---:|---:|
| **pypdf** (baseline) | 6,889 | 1.00x | 50.2% | 14/21 | - |
| **Surya** | **8,837** | **1.28x** ⭐ | 40.9% | 16/21 | 10.12 (느림) |
| **Tesseract** | 8,433 | 1.22x | 30.8% | 16/21 | **0.92** ⭐ (최속) |
| **EasyOCR** | 8,397 | 1.22x | 43.5% | 16/21 | 5.83 |
| **MinerU** | 8,022 | 1.16x | 41.8% | 21/21\* | 1.38 |
| **Marker** | 7,812 | 1.13x | 43.2% | 21/21\* | 8.25 |
| **PaddleOCR** | 7,420 | 1.08x | 48.3% | 16/21 | 3.16 |
| **Donut** | 5,855 | 0.85x | **51.6%** | 16/21 | 7.07 |
| **TrOCR** | 1,282 | 0.19x ❌ | 41.5% | 21/21\* | 0.50 |

> \* Marker / MinerU / TrOCR 은 페이지 분할이 어려워 균등분배(372자, 382자, 평균값)로 표시됨 — 페이지별 비교는 의미 없음, 총량만 유효.

### 2.2 카테고리별 평균 글자수

| 카테고리 | pypdf | easyocr | paddleocr | tesseract | **surya** | donut |
|---|---:|---:|---:|---:|---:|---:|
| weak (8장 — 대부분 진짜 백지) | 7.4 | 37 | 30 | 34 | 41 | 19 |
| table_suspect (10장 — 표·그림) | 334 | 456 | 391 | 400 | **490** ⭐ | 252 |
| normal (3장 — 빽빽한 본문) | 1,164 | 1,182 | 1,087 | **1,388** ⭐ | 1,201 | 1,061 |

> 표/그림 페이지에서 **Surya 가 PyPDF 대비 +47% 추가 회수**. 정상 본문에서는 **Tesseract 가 +19%** 로 최고.

### 2.3 페이지별 하이라이트

| 페이지 | 카테고리 | PyPDF | EasyOCR | PaddleOCR | Tesseract | Surya |
|---:|---|---:|---:|---:|---:|---:|
| 44 (표·그림) | table | 110 | 275 | 259 | 150 | **290** |
| 109 (표·그림) | table | 419 | **733** | 586 | 497 | 687 |
| 123 (표·그림) | table | 269 | 579 | 492 | 410 | **587** |
| 172 (표·그림) | table | 342 | 608 | 399 | 450 | **853** |
| 76 (본문) | normal | 1374 | 1450 | 1306 | **1684** | 1465 |
| 155 (본문) | normal | 1407 | 1406 | 1287 | **1466** | 1401 |

---

## 3. 핵심 관찰

1. **PyPDF 한계 확인 (표·그림 페이지)**
   - 표/그림 의심 페이지(10장)에서 OCR이 평균 +50~150 글자 추가 회수.
   - 가장 극적: **page 172 — PyPDF 342자 → Surya 853자** (×2.5 증가).
   - 즉, **현재 RAG 인덱스에 표·도식 정보가 상당히 누락**되고 있을 가능성 높음.

2. **약한 페이지 8장 중 5장은 진짜 백지**
   - page 1, 27, 131, 189, 227: 모든 모델 0자 → 구분지/간지 확정.
   - page 0, 2, 228: 표지·목차로 일부 텍스트 존재.
   - 즉, "한글비율 0% 페이지"는 OCR 도입해도 가치 없음.

3. **모델별 트레이드오프 (정량 한정)**
   | 사용처 | 추천 |
   |---|---|
   | 최대 회수율 | **Surya** (느리지만 1.28x) |
   | 속도+회수 균형 | **MinerU** (1.16x, 1.38s/page) |
   | 가장 빠름 | **Tesseract** (0.92s/page) — 단, 한글비율 30% 로 환각/잡음 의심 |
   | GPU 없는 환경 | Tesseract, PaddleOCR, MinerU (CPU 가능) |
   | 사실상 사용 불가 | **TrOCR** (single-line OCR → 페이지 입력 0.19x) |

4. **한글 비율로 본 신뢰도**
   - Tesseract 30.8% — 가장 낮음. 한글이 영문/특수문자로 잘못 인식됐을 의심.
   - Donut 51.6% — 가장 높음 (PyPDF 50.2% 보다도). 한글에 강하나 추출량 부족.

---

## 4. 한계 (반드시 짚고 가야 할 것)

> **정량(quantity) 만 봤지 정성(quality)은 검증 안 됨.**

- "글자수 1.28x"는 **글자를 더 많이 뽑았다**는 뜻일 뿐, **정확한지는 모름**.
- OCR이 텍스트를 **잘못된 순서로** 뽑을 수 있음 (특히 다단/표 페이지).
- **잘못 인식된 글자**도 글자수로는 똑같이 잡힘 (예: "베르누이"→"베르누어").
- **표 셀이 의미 단위로 묶였는지** 검증 안 됨.
- Marker/MinerU/TrOCR 은 페이지 분할이 안 되어 페이지별 비교가 균등분배 (총량만 유효).

이걸 검증하려면 **정답 셋(ground truth)** 이 필요:
- 대표 페이지 3~5장을 사람이 PDF 보고 수동으로 정확한 텍스트를 작성.
- 각 모델 출력과 비교: **CER (Character Error Rate)**, **단어 순서 보존률**, **표 셀 보존률**.

---

## 5. 다음 단계 (Phase 3)

| 옵션 | 설명 | 소요 |
|---|---|---|
| A. 수동 라벨링 + CER | 본문 1장 + 표 1장 + 그림 1장 정답 셋 작성 → 8개 모델 정확도 산출 | ~1시간 |
| B. 시각 비교 대시보드 | 21장 각자 [PDF 이미지 + 8개 모델 출력] HTML 으로 정렬 | ~30분 |
| C. RAG 통합 평가 | OCR 결과 중 상위 1~2개로 재인덱싱 → AutoRAG 셋으로 RAG 정확도 측정 | 별도 세션 |

**현재 추천 후속**: 표/그림 페이지에서 Surya·EasyOCR 출력 직접 눈으로 검증 → 진짜 본문이면 PyPDF + Surya 폴백 하이브리드 파이프라인 검토.

---

## 6. 산출물 위치

```
scripts/ocr_poc/
  01_pypdf_baseline.py         # PyPDF 베이스라인 진단
  02_select_samples.py         # 약함 8 + 표 10 + 정상 3 자동 선정
  03_extract_images.py         # PDF → PNG (300 DPI)
  common.py                    # 공통 러너 (timings, 카테고리)
  10_easyocr.py
  11_paddleocr.py
  12_tesseract.py              # TESSDATA_PREFIX=~/tessdata 사용
  13_surya.py
  14_trocr.py
  15_marker.py
  16_mineru.py
  17_donut.py
  20_compare.py                # 통합 비교 → comparison.{json,md}
  _setup_mineru.py             # magic-pdf 모델·설정 자동 다운로드

results/ocr_poc/
  baseline/
    per_page.json              # 229 페이지 PyPDF 추출 통계
    summary.json               # 전체 통계 + 약한 페이지 목록
    samples.json               # 선정된 21장
    pages/page_NNN.txt         # PyPDF 페이지별 원문
  images/page_NNN.png          # 샘플 21장 PNG (300 DPI)
  {model}/
    page_NNN.txt               # 모델별 추출 결과
    timings.json               # 페이지별 처리시간
  comparison.json              # 통합 비교 (모델×페이지 매트릭스)
  comparison.md                # 마크다운 리포트
```

---

## 7. 설치 노트 (재현용)

```bash
# 가벼운 OCR
pip install easyocr pytesseract pillow

# PaddleOCR (Windows: paddlepaddle 3.x 의 PIR+oneDNN 호환 버그 있음 → 2.6 권장)
pip install paddleocr==2.7.3 paddlepaddle==2.6.2

# Surya / Marker
pip install surya-ocr marker-pdf

# MinerU (magic-pdf)
pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
python scripts/ocr_poc/_setup_mineru.py  # 모델 ~3GB + ~/magic-pdf.json 생성

# numpy 1.x 필수 (cv2 ABI 호환)
pip install "numpy<2"

# Tesseract binary (Windows)
winget install --id UB-Mannheim.TesseractOCR --silent --accept-package-agreements
# 한국어 traineddata 별도 다운로드 → ~/tessdata/kor.traineddata
```

알려진 이슈:
- **HuggingFace Hub symlink 오류 (Windows 권한)** → `snapshot_download(local_dir=...)` 사용.
- **PaddleOCR 3.x + Windows 빌드** → `NotImplementedError: ConvertPirAttribute2RuntimeAttribute`. 2.7.3 + paddlepaddle 2.6.2 로 다운그레이드.
- **Surya v2 API** → `RecognitionPredictor(FoundationPredictor())` 필요, langs 인자는 deprecated.
- **MinerU 모델 파일명** → magic-pdf 1.3 은 `ch_PP-OCRv3_det_infer.pth` 를 기대하지만 hub 에는 `Multilingual_PP-OCRv3_det_infer.pth` 만 있음 → 같은 모델이라 copy 로 우회.
