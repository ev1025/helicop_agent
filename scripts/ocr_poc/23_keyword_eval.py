"""
OCR POC Phase 4-B-2 — 핵심 엔티티 회수율 (Keyword Recall) 평가.

사용자가 페이지마다 RAG 에서 살아남아야 할 핵심 키워드를 사전에 정의한다.
각 모델의 (raw / clean) 텍스트에서 그 키워드가 포함됐는지 자동 카운트.

입력:
  results/ocr_poc/keywords.json   (사용자 작성)
    {
      "0":   ["조종사", "표준교재", "비행이론", ...],   # 페이지 0 의 정답 키워드
      "76":  ["비상투하", "Jettison", "Main 탱크"],
      ...
    }

  (없으면 keywords_template.json 자동 생성 — 양식 채우라고 안내)

산출:
  results/ocr_poc/keyword_eval.json
    - 모델별 평균 recall (raw / clean)
    - 페이지별 상세

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/23_keyword_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import results_dir, baseline_dir, pages_dir

RESULTS_DIR = results_dir()
BASELINE_DIR = baseline_dir()
SAMPLES_JSON = BASELINE_DIR / "samples.json"
PYPDF_PAGES = pages_dir()

KEYWORDS_JSON = RESULTS_DIR / "keywords.json"
TEMPLATE_JSON = RESULTS_DIR / "keywords_template.json"

MODELS = ["pypdf", "easyocr", "paddleocr", "tesseract", "surya", "marker", "mineru", "donut"]


def load_text(model: str, page: int, clean: bool = False) -> str:
    if model == "pypdf":
        # pypdf 는 clean 도 거의 동일 — 기본 PyPDF baseline pages 사용
        f = PYPDF_PAGES / f"page_{page:03d}.txt" if not clean else RESULTS_DIR / "pypdf_clean" / f"page_{page:03d}.txt"
    else:
        if clean:
            f = RESULTS_DIR / f"{model}_clean" / f"page_{page:03d}.txt"
        else:
            f = RESULTS_DIR / model / f"page_{page:03d}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


import re
from collections import Counter

# 불용어 (한국어, 매뉴얼 전반에 흔히 나오나 의미가 낮음)
_STOPWORDS = {
    "있는", "있다", "없다", "있고", "있어", "있으며", "하는", "되는", "되어", "되어있",
    "같은", "대한", "위한", "위해", "이용", "사용", "사이", "통해", "이때", "또한",
    "그리고", "따라", "따른", "통한", "여러", "모든", "다른", "어떤", "이는", "이러한",
    "않는", "않은", "않으", "각각", "수있다", "수있는", "수있도", "비행이론", "헬리콥터",
    "Standard", "Pilot", "Handbook", "Flight", "Theory",  # 매뉴얼 어디에나 나오는 표지 단어
}


def auto_suggest_keywords(text: str, top_n: int = 5) -> list[str]:
    """PyPDF preview/전문 텍스트에서 자동 키워드 후보 상위 N개 추출.

    한국어 2-5자 명사 후보 + 대문자 시작 영단어(전문용어) + 식별번호.
    빈도 기반 + 불용어 제거.
    """
    if not text:
        return []
    kor = re.findall(r"[가-힯]{2,5}", text)
    eng = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    nums = re.findall(r"\b\d[\d\-]{4,}\b", text)
    cand = [c for c in kor + eng + nums if c not in _STOPWORDS]
    counter = Counter(cand)
    return [w for w, _ in counter.most_common(top_n)]


def write_template():
    """keywords_template.json 자동 생성 — PyPDF 텍스트에서 후보 자동 추출 + 사용자 검토용."""
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    template = {}
    for s in samples:
        p = s["page"]
        pypdf_text = load_text("pypdf", p)
        preview = pypdf_text[:120].replace("\n", " ")
        # 정상 페이지면 전문 텍스트로 후보 추출, 약함 페이지는 짧은 preview 만
        candidates = auto_suggest_keywords(pypdf_text, top_n=5)
        template[str(p)] = {
            "_category": s["category"],
            "_pypdf_chars": s["chars"],
            "_pypdf_preview": preview,
            "_image_path": f"results/ocr_poc/images/page_{p:03d}.png",
            "keywords": candidates,  # 자동 후보 — 사용자가 검토·수정·추가
        }
    TEMPLATE_JSON.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📝 키워드 양식 생성: {TEMPLATE_JSON}")
    print(f"   ✨ PyPDF 텍스트에서 자동 추출한 키워드 후보가 채워져 있음 — 검토·수정만 하면 됨.")
    print(f"   페이지 이미지 (results/ocr_poc/images/page_NNN.png) 확인하며:")
    print(f"     - 부적절한 키워드 삭제")
    print(f"     - 누락된 핵심 용어 추가 (특히 전문용어, 영문, 숫자)")
    print(f"     - 약함 페이지(표지 등)는 직접 핵심 메타데이터 입력")
    print(f"   완성 후 '{KEYWORDS_JSON.name}' 로 이름 변경 후 재실행.")


def eval_recall(text: str, keywords: list[str]) -> dict:
    """텍스트에 키워드 포함 여부 카운트 (대소문자 무관)."""
    if not keywords:
        return {"total": 0, "hit": 0, "recall": None, "missing": []}
    text_low = text.lower()
    hit = [k for k in keywords if k.lower() in text_low]
    missing = [k for k in keywords if k.lower() not in text_low]
    return {
        "total": len(keywords),
        "hit": len(hit),
        "recall": round(len(hit) / len(keywords), 4),
        "missing": missing,
    }


def main():
    if not KEYWORDS_JSON.exists():
        print(f"❌ {KEYWORDS_JSON} 없음.")
        if not TEMPLATE_JSON.exists():
            write_template()
        else:
            print(f"   양식은 이미 있음: {TEMPLATE_JSON}")
            print(f"   채워 넣고 '{KEYWORDS_JSON.name}' 로 이름 변경 후 재실행.")
        sys.exit(0)

    keywords_data = json.loads(KEYWORDS_JSON.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    page_cat = {s["page"]: s["category"] for s in samples}

    print(f"=== 키워드 회수율 평가: {len(MODELS)} 모델 × {len(keywords_data)} 페이지 ===\n")

    summary = {}
    for m in MODELS:
        per_page_raw = []
        per_page_clean = []
        cat_recalls_raw = {"weak": [], "table_suspect": [], "normal": []}
        cat_recalls_clean = {"weak": [], "table_suspect": [], "normal": []}

        for p_str, info in keywords_data.items():
            p = int(p_str)
            kw = info.get("keywords", []) if isinstance(info, dict) else info
            if not kw:
                continue
            cat = page_cat.get(p, "?")

            raw_text = load_text(m, p, clean=False)
            clean_text = load_text(m, p, clean=True)
            r_raw = eval_recall(raw_text, kw)
            r_clean = eval_recall(clean_text, kw)

            per_page_raw.append({"page": p, "category": cat, **r_raw})
            per_page_clean.append({"page": p, "category": cat, **r_clean})

            if cat in cat_recalls_raw and r_raw["recall"] is not None:
                cat_recalls_raw[cat].append(r_raw["recall"])
            if cat in cat_recalls_clean and r_clean["recall"] is not None:
                cat_recalls_clean[cat].append(r_clean["recall"])

        def avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else None

        all_raw = [r["recall"] for r in per_page_raw if r["recall"] is not None]
        all_clean = [r["recall"] for r in per_page_clean if r["recall"] is not None]

        summary[m] = {
            "n_pages_evaluated": len(all_raw),
            "overall_recall_raw": avg(all_raw),
            "overall_recall_clean": avg(all_clean),
            "by_category_raw": {c: avg(v) for c, v in cat_recalls_raw.items()},
            "by_category_clean": {c: avg(v) for c, v in cat_recalls_clean.items()},
            "per_page_raw": per_page_raw,
            "per_page_clean": per_page_clean,
        }

    out = RESULTS_DIR / "keyword_eval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 표
    print(f"{'모델':<12} {'raw recall':>11} {'clean recall':>13} "
          f"{'약함':>8} {'표·그림':>10} {'정상':>8}")
    print("-" * 72)

    def fmt(v):
        return f"{v:.1%}" if v is not None else "  -  "

    sorted_models = sorted(
        summary.items(),
        key=lambda kv: -(kv[1]["overall_recall_clean"] or 0)
    )
    for m, s in sorted_models:
        cb = s["by_category_clean"]
        print(f"  {m:<10} {fmt(s['overall_recall_raw']):>10} {fmt(s['overall_recall_clean']):>12} "
              f"{fmt(cb['weak']):>7} {fmt(cb['table_suspect']):>9} {fmt(cb['normal']):>7}")

    print(f"\n✅ {out}")


if __name__ == "__main__":
    main()
