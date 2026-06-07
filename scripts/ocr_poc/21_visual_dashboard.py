"""
OCR POC Phase 4-A/B — 시각 비교 대시보드 (Click-to-Rank).

설계 원칙:
  - 드래그 X, 클릭으로 순위 부여 (🥇🥈🥉)
  - 톱3 만 명확히, 4~8위는 자동 0pt
  - 페이지 헤더 sticky (가이드 + 진행률)
  - 좌측 PDF 이미지 sticky
  - 카드 텍스트에 키워드 자동 형광펜 + hit 카운트
  - 페이지별 건너뛰기 (백지 등)
  - localStorage 자동 저장 (v3 schema)

결과:
  results/ocr_poc/dashboard.html

실행:
  .venv/Scripts/python.exe scripts/ocr_poc/21_visual_dashboard.py
"""

from __future__ import annotations

import html as html_lib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "ocr_poc"
BASELINE_DIR = RESULTS_DIR / "baseline"
SAMPLES_JSON = BASELINE_DIR / "samples.json"
PYPDF_PAGES = BASELINE_DIR / "pages"
IMG_DIR = RESULTS_DIR / "images"
OUT_HTML = RESULTS_DIR / "dashboard.html"

OCR_MODELS = ["pypdf", "easyocr", "paddleocr", "tesseract", "surya", "marker", "mineru", "donut"]

MODEL_COLORS = {
    "pypdf": "#6b7280",
    "easyocr": "#3b82f6",
    "paddleocr": "#10b981",
    "tesseract": "#f59e0b",
    "surya": "#8b5cf6",
    "marker": "#06b6d4",
    "mineru": "#ec4899",
    "donut": "#14b8a6",
}

# 분류 코드 + 명확한 정량 기준식 (근거 자료용)
CATEGORY_CODE = {
    "weak": "TYPE_A_LOW_TEXT",
    "table_suspect": "TYPE_B_TABLE_FIGURE",
    "normal": "TYPE_C_BODY",
}
CATEGORY_RULE = {
    "weak": "pypdf_chars<100 OR kor_ratio<0.05",
    "table_suspect": "100<=pypdf_chars<500 AND NOT weak (sorted by lowest kor_ratio)",
    "normal": "500<=pypdf_chars<1500 AND kor_ratio>=0.50",
}
CATEGORY_LABEL = {
    "weak": "TYPE_A (저텍스트: chars<100 or kor<5%)",
    "table_suspect": "TYPE_B (표·그림 의심: 100-499chars, 낮은 한글비율)",
    "normal": "TYPE_C (정상 본문: 500-1499chars, 한글≥50%)",
}

CATEGORY_GUIDE = {
    "weak": "💡 메타데이터(책 제목·발간번호·출판사) 잘 잡았는지 — hit 많을수록 좋음",
    "table_suspect": "💡 표 셀·그림 캡션·본문 회수 — hit 참고 + 노이즈(이미지경로) 적은 게 좋음",
    "normal": "💡 본문·섹션 제목·그림 캡션은 <b>유지</b>, 책 반복 헤더(\"비행이론 Flight Theory\")·페이지번호만 <b>제거</b>가 좋음",
}

# 책 전역에 반복되는 헤더성 단어 — 정상/표 페이지에서는 키워드에서 제외
# (이 단어 hit 잡힌다 ≠ 본문 잘 뽑음. 단지 헤더 제거 안 됐다는 뜻)
HEADER_STOPWORDS = {
    "비행이론", "헬리콥터", "Flight", "Theory", "Standard", "Pilot", "Handbook",
    "표준교재", "조종사",
}

_STOPWORDS = {
    "있는","있다","없다","있고","있어","있으며","하는","되는","되어","되어있",
    "같은","대한","위한","위해","이용","사용","사이","통해","이때","또한",
    "그리고","따라","따른","통한","여러","모든","다른","어떤","이는","이러한",
    "않는","않은","않으","각각","수있다","수있는","수있도",
}


def load_text(model: str, page: int) -> str:
    if model == "pypdf":
        f = PYPDF_PAGES / f"page_{page:03d}.txt"
    else:
        f = RESULTS_DIR / model / f"page_{page:03d}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def page_checklist(text: str, category: str) -> list[str]:
    if not text:
        return []
    # 정상/표 페이지는 책 헤더 단어 (Flight, 비행이론 등) 제외 — 본문 키워드만
    extra_stops = HEADER_STOPWORDS if category != "weak" else set()

    fig_refs = list(set(re.findall(r"\[그림\s*\d+[-\.\d]*\]", text)))[:2]
    eng = [w for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text) if w not in extra_stops]
    eng_top = [w for w, _ in Counter(eng).most_common(2)]
    nums = list(set(re.findall(r"\b\d[\d\-]{4,}\b", text)))[:2]
    kor = [k for k in re.findall(r"[가-힯]{2,5}", text)
           if k not in _STOPWORDS and k not in extra_stops]
    kor_top = [k for k, _ in Counter(kor).most_common(5)]
    # 우선순위: 식별번호 → 그림 → 영문 → 한국어
    out, seen = [], set()
    for x in nums + fig_refs + eng_top + kor_top:
        if x and x not in seen:
            seen.add(x); out.append(x)
        if len(out) >= 5:
            break
    return out


def model_card_html(model: str, page: int) -> str:
    text = load_text(model, page)
    n = len(text)
    color = MODEL_COLORS[model]
    text_html = html_lib.escape(text) if text else "(빈 출력)"
    return f"""
    <div class="card" data-model="{model}" style="border-left:5px solid {color}">
        <div class="card-head">
            <span class="card-name" style="color:{color}">{model}</span>
            <span class="card-hit" data-hit="0/0">— hit</span>
            <span class="card-meta">{n:,}자</span>
        </div>
        <div class="card-text">{text_html}</div>
        <div class="card-actions">
            <button class="rank-btn" data-rank="1" data-model="{model}" title="1위 (3pt)">🥇 1위</button>
            <button class="rank-btn" data-rank="2" data-model="{model}" title="2위 (2pt)">🥈 2위</button>
            <button class="rank-btn" data-rank="3" data-model="{model}" title="3위 (1pt)">🥉 3위</button>
        </div>
    </div>
    """


def main():
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    print(f"=== 대시보드 생성: {len(samples)}장 × {len(OCR_MODELS)} 모델 ===")

    # 페이지 nav
    nav_items = []
    for s in samples:
        p = s["page"]
        nav_items.append(
            f'<a href="#page-{p}" class="nav-chip" id="nav-{p}" data-page="{p}">{p}</a>'
        )
    nav_html = "\n".join(nav_items)

    # 페이지 섹션들
    sections = []
    page_data = {}
    for s in samples:
        p = s["page"]
        cat = s["category"]
        pypdf_text = load_text("pypdf", p)
        checklist = page_checklist(pypdf_text, cat)
        page_data[p] = {"category": cat, "checklist": checklist}

        guide = CATEGORY_GUIDE[cat]
        cat_lbl = CATEGORY_LABEL[cat]
        kw_pills = " ".join(f'<span class="kw-pill">{html_lib.escape(k)}</span>' for k in checklist) \
                   if checklist else '<span class="muted">— PDF 이미지 직접 보고 평가</span>'
        cards = "".join(model_card_html(m, p) for m in OCR_MODELS)

        section = f"""
        <section class="page" id="page-{p}" data-page="{p}" data-category="{cat}">
            <div class="page-head">
                <div class="page-head-row">
                    <h2>Page {p}</h2>
                    <span class="cat-tag">{cat_lbl}</span>
                    <span class="char-info">{s['chars']:,}자 · 한글 {s['kor_ratio']:.0%}</span>
                    <button class="skip-btn" data-page="{p}">⊘ 건너뛰기</button>
                    <span class="page-status" id="status-{p}"></span>
                </div>
                <div class="guide-row">
                    <span class="guide-label">{guide}:</span>
                    <span class="kw-list">{kw_pills}</span>
                </div>
            </div>
            <div class="page-body">
                <div class="pdf-pane">
                    <img src="images/page_{p:03d}.png" alt="page {p}" loading="lazy">
                </div>
                <div class="cards-pane">{cards}</div>
            </div>
        </section>
        """
        sections.append(section)

    sections_html = "\n".join(sections)
    page_data_js = json.dumps(page_data, ensure_ascii=False)
    ocr_models_js = json.dumps(OCR_MODELS)

    # noise_eval + comparison 데이터를 JS 에 embed (자동 리포트용)
    noise_file = RESULTS_DIR / "noise_eval.json"
    noise_data_js = "{}"
    if noise_file.exists():
        nd = json.loads(noise_file.read_text(encoding="utf-8"))
        compact = {m: {"raw": v["raw_total_chars"], "noise": v["noise_total_chars"],
                       "clean": v["clean_total_chars"], "ratio": v["overall_noise_ratio"],
                       "patterns": v["pattern_counts"]} for m, v in nd.items()}
        noise_data_js = json.dumps(compact, ensure_ascii=False)

    comparison_file = RESULTS_DIR / "comparison.json"
    comparison_data_js = "{}"
    if comparison_file.exists():
        cd = json.loads(comparison_file.read_text(encoding="utf-8"))
        comparison_data_js = json.dumps(cd.get("model_summary", {}), ensure_ascii=False)

    sequence_file = RESULTS_DIR / "sequence_eval.json"
    sequence_data_js = "{}"
    if sequence_file.exists():
        sd = json.loads(sequence_file.read_text(encoding="utf-8"))
        compact = {m: {"avg": v["avg_sequence_similarity"], "by_cat": v["by_category"]}
                   for m, v in sd.items()}
        sequence_data_js = json.dumps(compact, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>OCR 모델 평가 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js"></script>
<script src="https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js"></script>
<style>
  :root {{
    --bg: #f9fafb;
    --surface: #ffffff;
    --border: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
    --primary: #2563eb;
    --gold: #fbbf24;
    --silver: #9ca3af;
    --bronze: #d97706;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo",
                 "Malgun Gothic", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}

  /* ───────── Top Bar (sticky) ───────── */
  .topbar {{
    position: sticky; top: 0; z-index: 100;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }}
  .topbar-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 10px; }}
  .topbar h1 {{ font-size: 16px; font-weight: 700; }}
  .progress-wrap {{ flex: 1; max-width: 400px; }}
  .progress-text {{ font-size: 12px; color: var(--muted); margin-bottom: 3px; }}
  .progress-bar {{ height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }}
  .progress-fill {{ height: 100%; background: linear-gradient(90deg, var(--primary), #10b981);
                    width: 0%; transition: width 0.3s; }}
  .topbar-actions {{ display: flex; gap: 6px; }}
  .btn {{
    padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--surface); cursor: pointer; font-size: 12.5px; font-weight: 500;
  }}
  .btn:hover {{ background: var(--bg); }}
  .btn-primary {{ background: var(--primary); color: white; border-color: var(--primary); }}
  .btn-primary:hover {{ background: #1d4ed8; }}
  .btn-danger {{ background: #fee2e2; color: #b91c1c; border-color: #fca5a5; }}

  /* nav chips */
  .nav {{ display: flex; gap: 4px; flex-wrap: wrap; }}
  .nav-chip {{
    width: 32px; height: 26px; line-height: 26px; text-align: center;
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--muted); text-decoration: none; font-size: 11.5px; font-weight: 600;
  }}
  .nav-chip:hover {{ background: var(--border); }}
  .nav-chip.done {{ background: #dcfce7; color: #15803d; border-color: #86efac; }}
  .nav-chip.skipped {{ background: #fef3c7; color: #a16207; border-color: #fcd34d; }}

  /* ───────── Page ───────── */
  main {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}
  .page {{
    background: var(--surface); border-radius: 10px; margin-bottom: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); scroll-margin-top: 160px;
    overflow: hidden;
  }}
  .page.skipped {{ opacity: 0.4; }}
  .page.skipped .cards-pane {{ pointer-events: none; }}

  .page-head {{
    background: var(--surface);
    padding: 14px 20px 12px 20px;
    border-bottom: 1px solid var(--border);
  }}
  .page-head-row {{
    display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
  }}
  .page-head h2 {{ font-size: 18px; font-weight: 700; }}
  .cat-tag {{
    padding: 3px 10px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; font-size: 12px; font-weight: 600; color: var(--muted);
  }}
  .char-info {{ font-size: 12px; color: var(--muted); }}
  .skip-btn {{
    margin-left: auto; padding: 5px 10px; font-size: 11.5px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 5px;
    cursor: pointer; color: var(--muted);
  }}
  .skip-btn:hover {{ background: #fef3c7; color: #a16207; border-color: #fbbf24; }}
  .page.skipped .skip-btn {{ background: #fef3c7; color: #a16207; border-color: #fbbf24; }}
  .page-status {{ font-size: 12px; font-weight: 600; }}
  .page-status.done {{ color: #15803d; }}

  .guide-row {{
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    font-size: 12.5px;
  }}
  .guide-label {{ color: var(--muted); font-weight: 500; }}
  .kw-list {{ display: flex; gap: 4px; flex-wrap: wrap; }}
  .kw-pill {{
    background: #fef9c3; border: 1px solid #fde047; border-radius: 4px;
    padding: 1px 7px; font-size: 11.5px; color: #713f12; font-weight: 500;
    font-family: "Consolas", "D2 Coding", monospace;
  }}
  .muted {{ color: var(--muted); font-size: 11.5px; }}

  /* ───────── Body Layout ───────── */
  .page-body {{
    display: grid;
    grid-template-columns: minmax(360px, 460px) 1fr;
    gap: 18px;
    padding: 18px 20px;
  }}
  .pdf-pane {{
    align-self: start;
    background: var(--bg);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 6px;
    max-height: 80vh;
    overflow: hidden;
    position: relative;
  }}
  .pdf-pane img {{
    width: 100%; height: auto; display: block; border-radius: 4px;
    cursor: zoom-in;
    transition: opacity 0.15s;
  }}
  .pdf-pane img:hover {{ opacity: 0.85; }}
  .pdf-pane::after {{
    content: '🔍 클릭하여 확대';
    position: absolute; top: 12px; right: 12px;
    background: rgba(0,0,0,0.65); color: white;
    padding: 4px 10px; border-radius: 14px;
    font-size: 11px; font-weight: 600;
    pointer-events: none;
  }}

  /* Lightbox — 클릭하면 큰 화면, 드래그로 패닝, 휠로 줌 */
  #lightbox {{
    position: fixed; inset: 0;
    background: transparent;
    z-index: 999;
    display: none;
    cursor: grab;
    overflow: hidden;
    user-select: none;
  }}
  #lightbox.open {{ display: block; }}
  #lightbox.dragging {{ cursor: grabbing; }}
  #lightbox-img {{
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%) scale(1);
    transform-origin: center center;
    max-width: 92vw; max-height: 92vh;
    width: auto; height: auto;
    object-fit: contain;
    will-change: transform;
    pointer-events: none;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
  }}
  #lightbox-close {{
    position: fixed; top: 16px; right: 20px;
    background: rgba(255,255,255,0.15); color: white;
    border: none; width: 40px; height: 40px;
    border-radius: 50%; cursor: pointer;
    font-size: 20px; z-index: 1001;
  }}
  #lightbox-close:hover {{ background: rgba(255,255,255,0.3); }}
  #lightbox-hint {{
    position: fixed; bottom: 24px; left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.6); color: white;
    padding: 8px 16px; border-radius: 8px;
    font-size: 12.5px; z-index: 1001;
    pointer-events: none;
  }}
  #lightbox-zoom {{
    position: fixed; top: 16px; left: 20px;
    background: rgba(255,255,255,0.15); color: white;
    padding: 6px 12px; border-radius: 16px;
    font-size: 12px; z-index: 1001;
    font-variant-numeric: tabular-nums;
  }}

  .cards-pane {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
    align-content: start;
  }}

  /* ───────── Card ───────── */
  .card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; display: flex; flex-direction: column;
    transition: background 0.2s, border-color 0.2s;
  }}
  .card[data-rank="1"] {{
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-color: var(--gold);
  }}
  .card[data-rank="2"] {{
    background: #e0e7ff; border-color: var(--silver);
  }}
  .card[data-rank="3"] {{
    background: #fed7aa; border-color: var(--bronze);
  }}
  .card-head {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
    padding-bottom: 6px; border-bottom: 1px dashed var(--border);
  }}
  .card-name {{ font-weight: 700; font-size: 13px; }}
  .card-hit {{
    padding: 1px 6px; border-radius: 8px; font-size: 10.5px; font-weight: 700;
    background: #e5e7eb; color: var(--muted);
  }}
  .card-hit.all {{ background: #16a34a; color: white; }}
  .card-hit.partial {{ background: #f59e0b; color: white; }}
  .card-meta {{ margin-left: auto; font-size: 10.5px; color: var(--muted); }}

  .card-text {{
    font-family: "Consolas", "Menlo", "D2 Coding", monospace;
    font-size: 12px; line-height: 1.55;
    color: #1f2937;
    max-height: 240px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
    margin-bottom: 8px;
    padding: 4px 2px;
  }}
  mark.kw-hit {{
    background: #fef08a; color: inherit; font-weight: 600;
    padding: 0 2px; border-radius: 2px;
  }}

  /* 카드 액션 — 순위 버튼 (CTA 영역) */
  .card-actions {{
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px;
    margin-top: auto;
  }}
  .rank-btn {{
    padding: 7px 4px; border-radius: 5px; cursor: pointer;
    font-size: 12px; font-weight: 600;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted);
    transition: all 0.15s;
  }}
  .rank-btn:hover {{ background: var(--bg); }}
  .rank-btn.active[data-rank="1"] {{ background: var(--gold); color: #78350f; border-color: var(--gold); }}
  .rank-btn.active[data-rank="2"] {{ background: var(--silver); color: white; border-color: var(--silver); }}
  .rank-btn.active[data-rank="3"] {{ background: var(--bronze); color: white; border-color: var(--bronze); }}

  /* ───────── Modal ───────── */
  .modal-bg {{
    position: fixed; inset: 0; background: rgba(0,0,0,0.5);
    display: none; z-index: 200; align-items: center; justify-content: center;
  }}
  .modal-bg.open {{ display: flex; }}
  .modal {{
    background: var(--surface); border-radius: 12px; padding: 24px;
    max-width: 880px; width: 92%; max-height: 85vh; overflow-y: auto;
  }}
  .modal h2 {{ font-size: 20px; margin-bottom: 12px; }}
  .modal-close {{
    float: right; background: var(--bg); border: none; padding: 4px 10px;
    border-radius: 4px; cursor: pointer; font-size: 16px;
  }}
  table.stats {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  table.stats th, table.stats td {{
    border-bottom: 1px solid var(--border); padding: 8px 10px; text-align: left;
  }}
  table.stats th {{ background: var(--bg); font-size: 11.5px; color: var(--muted); }}
  table.stats td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  table.stats tr.gold td {{ background: #fef3c7; font-weight: 600; }}

  @media (max-width: 1100px) {{
    .page-body {{ grid-template-columns: 1fr; }}
    .pdf-pane {{ position: static; max-height: 500px; }}
  }}
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-row">
    <h1>🔎 OCR 평가 대시보드</h1>
    <div class="progress-wrap">
      <div class="progress-text" id="progress-text">0/{len(samples)} 완료</div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    </div>
    <div class="topbar-actions">
      <button class="btn btn-primary" onclick="showStats()">📊 통계</button>
      <button class="btn" onclick="exportJSON()">💾 Export</button>
      <input type="file" id="import-file" style="display:none" accept=".json" onchange="importJSON(event)">
      <button class="btn" onclick="document.getElementById('import-file').click()">📂 Import</button>
      <button class="btn btn-danger" onclick="resetAll()">🗑️ 초기화</button>
    </div>
  </div>
  <div class="nav">{nav_html}</div>
</div>

<main>
{sections_html}
</main>

<div class="modal-bg" id="stats-modal" onclick="if(event.target===this)closeStats()">
  <div class="modal">
    <button class="modal-close" onclick="closeStats()">×</button>
    <h2>📊 종합 평가 통계</h2>
    <div id="stats-body">통계 계산 중...</div>
  </div>
</div>

<div id="lightbox">
  <button id="lightbox-close" onclick="closeLightbox()">×</button>
  <span id="lightbox-zoom">100%</span>
  <img id="lightbox-img" alt="확대">
  <div id="lightbox-hint">휠 = 카드 스크롤 · Ctrl+휠 = 확대/축소 · 드래그 = 이동 · 이미지 클릭 = 2배 토글 · 외부 클릭/ESC/✕ = 닫기</div>
</div>

<script>
const OCR_MODELS = {ocr_models_js};
const PAGE_DATA = {page_data_js};
const NOISE_DATA = {noise_data_js};
const COMPARISON_DATA = {comparison_data_js};
const SEQUENCE_DATA = {sequence_data_js};
const TOTAL_PAGES = {len(samples)};
const STORAGE_KEY = 'ocr_eval_v3';
const REPORT_NOTIFIED_KEY = 'ocr_report_notified_v3';

// 종합 점수 가중치
const SCORE_WEIGHTS = {{ qual: 0.5, clean: 0.2, seq: 0.3 }};

// Schema:
//   EVALS[pageId] = "skipped"
//   EVALS[pageId] = {{ "1": "modelA", "2": "modelB", "3": "modelC" }}
//   undefined = 미평가

function loadEvals() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{}}; }} catch {{ return {{}}; }}
}}
function saveEvals(e) {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(e)); }}
let EVALS = loadEvals();

const isSkipped = p => EVALS[p] === "skipped";
const getRanks = p => (typeof EVALS[p] === "object" && EVALS[p] !== null && EVALS[p] !== "skipped") ? EVALS[p] : null;
const isRated = p => {{
  const r = getRanks(p);
  return r && (r["1"] || r["2"] || r["3"]);  // 최소 1개 순위라도 부여됨
}};

// ───────── 순위 부여 (click-to-rank) ─────────
function applyRank(pageId, model, rank) {{
  let ranks = getRanks(pageId);
  if (!ranks) ranks = {{}};
  const rankStr = String(rank);

  // 이미 이 모델이 같은 순위 → 해제
  if (ranks[rankStr] === model) {{
    delete ranks[rankStr];
  }} else {{
    // 다른 모델이 이 순위였으면 그 모델 해제
    if (ranks[rankStr]) {{
      // (그 모델은 그냥 자리에서 빠짐)
    }}
    // 이 모델이 다른 순위에 있었으면 그 자리 해제
    for (const r of ["1","2","3"]) {{
      if (ranks[r] === model && r !== rankStr) delete ranks[r];
    }}
    ranks[rankStr] = model;
  }}

  if (Object.keys(ranks).length === 0) {{
    delete EVALS[pageId];
  }} else {{
    EVALS[pageId] = ranks;
  }}
  saveEvals(EVALS);
  refreshPage(pageId);
  refreshProgress();
}}

// 페이지 UI 갱신 (카드 배경, 버튼 active, nav, status)
function refreshPage(pageId) {{
  const section = document.getElementById('page-' + pageId);
  if (!section) return;
  const ranks = getRanks(pageId) || {{}};

  // 모든 카드 초기화
  section.querySelectorAll('.card').forEach(card => {{
    delete card.dataset.rank;
    card.querySelectorAll('.rank-btn').forEach(b => b.classList.remove('active'));
  }});

  // 순위 부여된 카드 마킹
  for (const r of ["1","2","3"]) {{
    const m = ranks[r];
    if (!m) continue;
    const card = section.querySelector('.card[data-model="' + m + '"]');
    if (!card) continue;
    card.dataset.rank = r;
    const btn = card.querySelector('.rank-btn[data-rank="' + r + '"]');
    if (btn) btn.classList.add('active');
  }}

  // skipped 시각 처리
  const skipped = isSkipped(pageId);
  section.classList.toggle('skipped', skipped);

  // status 뱃지
  const statusEl = document.getElementById('status-' + pageId);
  const navChip = document.getElementById('nav-' + pageId);
  if (skipped) {{
    statusEl.textContent = '⊘ 통계 제외';
    statusEl.className = 'page-status';
    statusEl.style.color = '#a16207';
    navChip.className = 'nav-chip skipped';
  }} else if (isRated(pageId)) {{
    const count = Object.keys(ranks).filter(k => ranks[k]).length;
    statusEl.textContent = '✓ ' + count + '/3 순위';
    statusEl.className = 'page-status done';
    statusEl.style.color = '';
    navChip.className = 'nav-chip done';
  }} else {{
    statusEl.textContent = '';
    statusEl.className = 'page-status';
    statusEl.style.color = '';
    navChip.className = 'nav-chip';
  }}
}}

function refreshProgress() {{
  const done = Object.keys(PAGE_DATA).filter(p => isRated(p) || isSkipped(p)).length;
  document.getElementById('progress-text').textContent = done + '/' + TOTAL_PAGES + ' 완료';
  document.getElementById('progress-fill').style.width = (done / TOTAL_PAGES * 100) + '%';
  // 모두 완료 → 1회만 알림 + 자동 CSV 다운로드
  if (done === TOTAL_PAGES && !localStorage.getItem(REPORT_NOTIFIED_KEY)) {{
    localStorage.setItem(REPORT_NOTIFIED_KEY, '1');
    setTimeout(() => {{
      if (confirm('🎉 모든 페이지 처리 완료!\\n\\n종합 평가 엑셀(요약·상세·페이지별 3시트)을 자동 다운로드합니다.')) {{
        downloadXLSX();
      }}
    }}, 300);
  }}
}}

// ───────── Click handlers ─────────
document.querySelectorAll('.rank-btn').forEach(btn => {{
  btn.addEventListener('click', e => {{
    e.stopPropagation();
    applyRank(btn.closest('.page').dataset.page, btn.dataset.model, btn.dataset.rank);
  }});
}});
document.querySelectorAll('.skip-btn').forEach(btn => {{
  btn.addEventListener('click', e => {{
    const p = btn.dataset.page;
    if (isSkipped(p)) delete EVALS[p];
    else EVALS[p] = "skipped";
    saveEvals(EVALS);
    refreshPage(p);
    refreshProgress();
  }});
}});

// ───────── 키워드 형광펜 + hit 카운트 ─────────
function escapeRegex(s) {{ return s.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'); }}

function highlightAll() {{
  Object.keys(PAGE_DATA).forEach(p => {{
    const checklist = PAGE_DATA[p].checklist || [];
    document.querySelectorAll('#page-' + p + ' .card').forEach(card => {{
      const textEl = card.querySelector('.card-text');
      if (!textEl) return;
      const original = textEl.dataset.original || textEl.innerHTML;
      textEl.dataset.original = original;
      let html = original;
      let hit = 0;
      const plainText = original.replace(/<[^>]+>/g, '');
      checklist.forEach(kw => {{
        const re = new RegExp(escapeRegex(kw), 'gi');
        if (re.test(plainText)) hit++;
        html = html.replace(new RegExp(escapeRegex(kw), 'gi'), m => '<mark class="kw-hit">' + m + '</mark>');
      }});
      textEl.innerHTML = html;
      const hitEl = card.querySelector('.card-hit');
      const total = checklist.length;
      hitEl.textContent = total ? hit + '/' + total + ' hit' : '— hit';
      hitEl.className = 'card-hit' + (total === 0 ? '' : (hit === total ? ' all' : (hit > 0 ? ' partial' : '')));
    }});
  }});
}}

// ───────── 통계 모달 ─────────
const POINTS = {{1: 3, 2: 2, 3: 1}};
function showStats() {{
  const pts = {{}};       // model → [pt, ...]
  const ptsByCat = {{ weak: {{}}, table_suspect: {{}}, normal: {{}} }};
  const top = {{}};       // model → {{first, second, third}}
  OCR_MODELS.forEach(m => {{
    pts[m] = [];
    top[m] = {{ first: 0, second: 0, third: 0 }};
    Object.keys(ptsByCat).forEach(c => ptsByCat[c][m] = []);
  }});

  Object.keys(EVALS).forEach(p => {{
    if (isSkipped(p)) return;
    const ranks = getRanks(p);
    if (!ranks) return;
    const cat = PAGE_DATA[p].category;
    // 1~3위 부여된 모델만 점수, 나머지는 0pt 처리 안 함 (포함 안함)
    OCR_MODELS.forEach(m => {{
      let myRank = null;
      for (const r of ["1","2","3"]) if (ranks[r] === m) myRank = parseInt(r);
      const pt = myRank ? POINTS[myRank] : 0;
      pts[m].push(pt);
      ptsByCat[cat][m].push(pt);
      if (myRank === 1) top[m].first++;
      else if (myRank === 2) top[m].second++;
      else if (myRank === 3) top[m].third++;
    }});
  }});

  const avg = a => a.length ? (a.reduce((x,y)=>x+y,0)/a.length) : null;
  const fmt = v => v === null ? '-' : v.toFixed(2);

  const ranked = OCR_MODELS.map(m => ({{
    model: m,
    overall: avg(pts[m]),
    count: pts[m].length,
    weak: avg(ptsByCat.weak[m]),
    table: avg(ptsByCat.table_suspect[m]),
    normal: avg(ptsByCat.normal[m]),
    top: top[m],
  }})).sort((a, b) => (b.overall ?? -1) - (a.overall ?? -1));

  const rated = Object.keys(PAGE_DATA).filter(p => isRated(p)).length;
  const skipped = Object.keys(PAGE_DATA).filter(p => isSkipped(p)).length;

  let html = '<p style="color:#6b7280;font-size:13px;margin-bottom:14px">';
  html += '평가 ' + rated + '장 · 건너뛰기 ' + skipped + '장 / 전체 ' + TOTAL_PAGES + '장. ';
  html += '<b>높을수록 좋음</b> (1위=3pt, 2위=2pt, 3위=1pt, 4~8위=0pt).</p>';

  html += '<table class="stats"><thead><tr>';
  html += '<th>순위</th><th>모델</th><th class="num">평균 점수</th><th class="num">🥇/🥈/🥉</th>';
  html += '<th class="num">약함</th><th class="num">표·그림</th><th class="num">정상</th></tr></thead><tbody>';

  ranked.forEach((r, i) => {{
    const cls = (i === 0 && r.overall) ? 'gold' : '';
    html += '<tr class="' + cls + '">';
    html += '<td>' + (r.overall === null || r.overall === 0 ? '-' : (i+1)) + '</td>';
    html += '<td><span style="color:' + (({{pypdf:"#6b7280",easyocr:"#3b82f6",paddleocr:"#10b981",tesseract:"#f59e0b",surya:"#8b5cf6",marker:"#06b6d4",mineru:"#ec4899",donut:"#14b8a6"}})[r.model]) + ';font-weight:600">' + r.model + '</span></td>';
    html += '<td class="num">' + fmt(r.overall) + '</td>';
    html += '<td class="num">' + r.top.first + ' / ' + r.top.second + ' / ' + r.top.third + '</td>';
    html += '<td class="num">' + fmt(r.weak) + '</td>';
    html += '<td class="num">' + fmt(r.table) + '</td>';
    html += '<td class="num">' + fmt(r.normal) + '</td>';
    html += '</tr>';
  }});
  html += '</tbody></table>';

  // ── 종합 점수 표 (정성 + 노이즈 + 시퀀스 가중 합) ──
  if (Object.keys(NOISE_DATA).length && Object.keys(SEQUENCE_DATA).length) {{
    const finalScores = computeFinalScores();
    html += '<h3 style="margin-top:20px;font-size:16px">🏆 종합 점수 (정성 50% + 노이즈 20% + 시퀀스 30%)</h3>';
    html += '<p style="color:#6b7280;font-size:11.5px;margin:4px 0 8px">';
    html += 'final = 0.5·(qual_avg/3) + 0.2·(1-noise) + 0.3·seq_similarity_vs_pypdf  (0~1, 높을수록 좋음)</p>';
    html += '<table class="stats"><thead><tr>';
    html += '<th>순위</th><th>모델</th><th class="num">종합</th>';
    html += '<th class="num">정성/3</th><th class="num">1-노이즈</th><th class="num">시퀀스</th>';
    html += '</tr></thead><tbody>';
    finalScores.forEach((s, i) => {{
      const cls = i === 0 ? 'gold' : '';
      html += '<tr class="' + cls + '">';
      html += '<td>' + (i+1) + '</td>';
      html += '<td><span style="color:' + (({{pypdf:"#6b7280",easyocr:"#3b82f6",paddleocr:"#10b981",tesseract:"#f59e0b",surya:"#8b5cf6",marker:"#06b6d4",mineru:"#ec4899",donut:"#14b8a6"}})[s.model]) + ';font-weight:600">' + s.model + '</span></td>';
      html += '<td class="num"><b>' + s.final_score.toFixed(2) + '</b></td>';
      html += '<td class="num">' + (s.qual_norm).toFixed(2) + '</td>';
      html += '<td class="num">' + (s.clean_norm).toFixed(2) + '</td>';
      html += '<td class="num">' + (s.seq_norm).toFixed(2) + '</td>';
      html += '</tr>';
    }});
    html += '</tbody></table>';
  }}

  // ── CSV 다운로드 버튼 ──
  html += '<div style="display:flex;gap:8px;margin-top:16px">';
  html += '<button class="btn btn-primary" onclick="downloadXLSX()">📊 엑셀 다운로드 (요약·상세·페이지별 3시트)</button>';
  html += '<button class="btn" onclick="downloadReport()">📋 Markdown 리포트</button>';
  html += '</div>';

  document.getElementById('stats-body').innerHTML = html;
  document.getElementById('stats-modal').classList.add('open');
}}
function closeStats() {{ document.getElementById('stats-modal').classList.remove('open'); }}

// ───────── 종합 리포트 생성 (Markdown) ─────────
function computeStatsForReport() {{
  const pts = {{}}, ptsByCat = {{ weak: {{}}, table_suspect: {{}}, normal: {{}} }};
  const top = {{}};
  OCR_MODELS.forEach(m => {{
    pts[m] = []; top[m] = {{ first: 0, second: 0, third: 0 }};
    Object.keys(ptsByCat).forEach(c => ptsByCat[c][m] = []);
  }});
  Object.keys(EVALS).forEach(p => {{
    if (isSkipped(p)) return;
    const ranks = getRanks(p); if (!ranks) return;
    const cat = PAGE_DATA[p].category;
    OCR_MODELS.forEach(m => {{
      let myRank = null;
      for (const r of ["1","2","3"]) if (ranks[r] === m) myRank = parseInt(r);
      const pt = myRank ? POINTS[myRank] : 0;
      pts[m].push(pt);
      ptsByCat[cat][m].push(pt);
      if (myRank === 1) top[m].first++;
      else if (myRank === 2) top[m].second++;
      else if (myRank === 3) top[m].third++;
    }});
  }});
  const avg = a => a.length ? (a.reduce((x,y)=>x+y,0)/a.length) : null;
  return OCR_MODELS.map(m => ({{
    model: m, overall: avg(pts[m]), count: pts[m].length,
    weak: avg(ptsByCat.weak[m]), table: avg(ptsByCat.table_suspect[m]),
    normal: avg(ptsByCat.normal[m]), top: top[m],
  }})).sort((a, b) => (b.overall ?? -1) - (a.overall ?? -1));
}}

function generateReport() {{
  const today = new Date().toISOString().slice(0,10);
  const stats = computeStatsForReport();
  const finalScores = computeFinalScores();
  const rated = Object.keys(PAGE_DATA).filter(p => isRated(p)).length;
  const skipped = Object.keys(PAGE_DATA).filter(p => isSkipped(p)).length;
  const fmt = v => v === null ? '-' : v.toFixed(2);

  let md = '# 🔎 OCR 모델 평가 종합 리포트\\n\\n';
  md += '> 헬리콥터 매뉴얼 (조종사 표준교재) PDF 의 OCR 모델 8종 비교 결과입니다.\\n';
  md += '> RAG 인덱싱에 사용할 최종 모델을 선정하기 위한 근거 자료입니다.\\n\\n';
  md += '- **생성일**: ' + today + '\\n';
  md += '- **평가 페이지**: ' + rated + '장 (사용자 정성 평가) / 건너뛰기 ' + skipped + '장 / 전체 ' + TOTAL_PAGES + '장\\n';
  md += '- **비교 모델 (8개)**: ' + OCR_MODELS.join(', ') + '\\n\\n';

  md += '---\\n\\n## 📐 평가 방법론\\n\\n';
  md += '### 페이지 분류 (3가지 카테고리, 정량 기준)\\n\\n';
  md += '| 페이지 종류 | 정량 기준 | 평가 포인트 |\\n';
  md += '|---|---|---|\\n';
  md += '| 표지목차 | PyPDF 글자수<100 OR 한글비율<5% | 메타데이터(책 제목·발간번호·출판사) 회수 |\\n';
  md += '| 표그림 | 100<=글자수<500 + 한글비율 낮음 | 표 셀·그림 캡션 보존, 본문 회수 |\\n';
  md += '| 정상본문 | 500<=글자수<1500 + 한글비율>=50% | 글자 정확성, 반복 헤더 제거, 문장 순서 |\\n\\n';

  md += '### 종합 점수 산정식 (0~1, 높을수록 RAG 친화적)\\n\\n';
  md += '```\\n';
  md += '종합점수 = 0.5 × 정성평가_정규화      (사용자 드래그 순위)\\n';
  md += '         + 0.2 × 깨끗함_정규화         (자동 노이즈 비율의 역수)\\n';
  md += '         + 0.3 × 시퀀스일치도          (PyPDF 단어 순서 대비 일치율)\\n';
  md += '```\\n\\n';
  md += '- **정성평가 정규화** = 평균 순위 점수 ÷ 3 (1위=3pt, 2위=2pt, 3위=1pt, 4~8위=0pt)\\n';
  md += '- **깨끗함 정규화** = 1 - 노이즈 비율 (이미지 마크다운·HTML 태그·절대 경로 자동 검출)\\n';
  md += '- **시퀀스 일치도** = PyPDF clean text vs 모델 clean text의 단어 시퀀스 일치율 (Ratcliff-Obershelp 알고리즘)\\n\\n';
  md += '### 가중치 근거\\n\\n';
  md += '- **정성 50%**: 사용자가 직접 RAG 친화도를 판단한 것이라 가장 중요\\n';
  md += '- **깨끗함 20%**: 노이즈는 정규식 전처리로 자동 제거 가능하므로 상대적으로 낮은 가중치\\n';
  md += '- **시퀀스 30%**: 본문 흐름이 깨지면 RAG 청크의 의미가 손상되므로 정성 다음으로 중요\\n\\n';

  md += '---\\n\\n## 🏆 1. 종합 순위 (Final Score)\\n\\n';
  md += '| 순위 | 모델 | **종합점수** | 정성/3 | 1-노이즈 | 시퀀스 |\\n';
  md += '|---:|---|---:|---:|---:|---:|\\n';
  finalScores.forEach((s, i) => {{
    const star = i === 0 ? ' ⭐' : '';
    md += '| ' + (i+1) + star + ' | **' + s.model + '** | **' + s.final_score.toFixed(2) + '** | '
        + s.qual_norm.toFixed(2) + ' | ' + s.clean_norm.toFixed(2) + ' | ' + s.seq_norm.toFixed(2) + ' |\\n';
  }});

  md += '\\n---\\n\\n## 📊 2. 정성 평가 상세 (사용자 드래그 순위 — 16장)\\n\\n';
  md += '점수: 🥇 1위=3pt · 🥈 2위=2pt · 🥉 3위=1pt · 4~8위=0pt\\n\\n';
  md += '| 순위 | 모델 | 평균 점수 | 🥇/🥈/🥉 | 표지목차 | 표·그림 | 정상본문 |\\n';
  md += '|---:|---|---:|---:|---:|---:|---:|\\n';
  stats.forEach((r, i) => {{
    const rank = (r.overall === null || r.overall === 0) ? '-' : (i+1);
    md += '| ' + rank + ' | **' + r.model + '** | ' + fmt(r.overall) +
          ' | ' + r.top.first + ' / ' + r.top.second + ' / ' + r.top.third +
          ' | ' + fmt(r.weak) + ' | ' + fmt(r.table) + ' | ' + fmt(r.normal) + ' |\\n';
  }});

  if (Object.keys(NOISE_DATA).length) {{
    md += '\\n---\\n\\n## 🧹 3. 자동 노이즈 평가\\n\\n';
    md += '검출 패턴: Markdown 이미지 링크 `![](...)`, HTML 태그 `<tag>`, 절대 경로 `C:\\\\...`, URL, 환각 반복 토큰\\n\\n';
    md += '| 모델 | 원본 글자 | 노이즈 글자 | 정화 후 | **노이즈 비율** | 주요 패턴 |\\n';
    md += '|---|---:|---:|---:|---:|---|\\n';
    const noiseSorted = Object.entries(NOISE_DATA).sort((a, b) => a[1].ratio - b[1].ratio);
    noiseSorted.forEach(([m, n]) => {{
      const tops = Object.entries(n.patterns || {{}}).filter(([k,v]) => v > 0)
                        .sort((a,b) => b[1]-a[1]).map(([k,v]) => k+'='+v).join(', ') || '-';
      md += '| ' + m + ' | ' + n.raw + ' | ' + n.noise + ' | ' + n.clean +
            ' | **' + (n.ratio * 100).toFixed(1) + '%** | ' + tops + ' |\\n';
    }});
    md += '\\n*노이즈 비율이 높아도 전처리로 자동 제거 가능합니다. 단 노이즈가 많을수록 정화 과정의 부담이 큽니다.*\\n';
  }}

  if (Object.keys(SEQUENCE_DATA).length) {{
    md += '\\n---\\n\\n## 🔁 4. 시퀀스 일치도 (PyPDF 대비 단어 순서)\\n\\n';
    md += '본문 흐름이 자연스러운지 측정. 도식 라벨이 본문 사이에 끼어들거나 단어 순서가 흐트러지면 점수 하락.\\n\\n';
    md += '| 모델 | **전체 평균** | 표지목차 | 표·그림 | 정상본문 |\\n';
    md += '|---|---:|---:|---:|---:|\\n';
    const seqSorted = Object.entries(SEQUENCE_DATA).sort((a, b) => (b[1].avg || 0) - (a[1].avg || 0));
    seqSorted.forEach(([m, s]) => {{
      const cb = s.by_cat || {{}};
      md += '| ' + m + ' | **' + (s.avg !== null ? s.avg.toFixed(2) : '-') + '** | '
          + (cb.weak !== null ? cb.weak.toFixed(2) : '-') + ' | '
          + (cb.table_suspect !== null ? cb.table_suspect.toFixed(2) : '-') + ' | '
          + (cb.normal !== null ? cb.normal.toFixed(2) : '-') + ' |\\n';
    }});
    md += '\\n*PyPDF 자기 자신 비교는 항상 1.000 입니다.*\\n';
  }}

  if (Object.keys(COMPARISON_DATA).length) {{
    md += '\\n---\\n\\n## 📈 5. 정량 비교 (총 글자수 · 속도)\\n\\n';
    md += '| 모델 | 총 글자수 | vs PyPDF | 평균 한글비율 | 페이지당 평균 시간 |\\n';
    md += '|---|---:|---:|---:|---:|\\n';
    ['pypdf'].concat(OCR_MODELS.filter(m => m !== 'pypdf')).forEach(m => {{
      const c = COMPARISON_DATA[m]; if (!c) return;
      md += '| ' + m + ' | ' + c.total_chars + ' | ' + c.vs_pypdf_chars_ratio + 'x | ' +
            (c.avg_kor_ratio * 100).toFixed(1) + '% | ' + (c.avg_seconds_per_page || '-') + 's |\\n';
    }});
  }}

  md += '\\n---\\n\\n## 🎯 6. 결론 — RAG 인덱싱 추천 모델\\n\\n';
  const top1 = finalScores[0];
  const top2 = finalScores[1];
  if (top1) {{
    md += '### 🥇 1순위: `' + top1.model + '` (종합점수 ' + top1.final_score.toFixed(2) + ')\\n';
    md += '- 정성 평가: 평균 ' + fmt(top1.qual_avg) + 'pt (1위 ' + top1.gold + '회)\\n';
    md += '- 노이즈 비율: ' + ((top1.noise_ratio || 0) * 100).toFixed(1) + '%\\n';
    md += '- 시퀀스 일치도: ' + (top1.seq_avg !== null ? top1.seq_avg.toFixed(2) : '-') + '\\n\\n';
  }}
  if (top2) {{
    md += '### 🥈 2순위: `' + top2.model + '` (종합점수 ' + top2.final_score.toFixed(2) + ')\\n';
    md += '- 1순위와 격차: ' + ((top1.final_score - top2.final_score) * 100).toFixed(1) + '%pt\\n';
    md += '- 백업/하이브리드 후보로 검토\\n\\n';
  }}

  md += '### 전처리 권장사항 (RAG 인덱싱 전에 필수)\\n\\n';
  md += '```regex\\n';
  md += '\\\\!\\\\[.*?\\\\]\\\\(.*?\\\\)        # Markdown 이미지 링크 제거\\n';
  md += '<[^>]+>                  # HTML 태그 제거\\n';
  md += '[A-Z]:[\\\\\\\\/][^\\\\s)\\\\]]+  # Windows 절대 경로 제거\\n';
  md += 'https?://\\\\S+             # URL 제거\\n';
  md += '```\\n\\n';
  md += '추가로 책 반복 헤더 (\"비행이론(헬리콥터) Flight Theory\") 와 페이지번호 (\"·N\") 는 별도 stopword 처리 권장.\\n';

  md += '\\n---\\n\\n## 📄 7. 페이지별 raw 평가 내역\\n\\n';
  md += '| 페이지 | 카테고리 | 상태 | 🥇 1위 | 🥈 2위 | 🥉 3위 |\\n';
  md += '|---:|---|---|---|---|---|\\n';
  const code_map = {{ weak: '표지목차', table_suspect: '표·그림', normal: '정상본문' }};
  Object.keys(PAGE_DATA).sort((a,b) => parseInt(a) - parseInt(b)).forEach(p => {{
    const cat = code_map[PAGE_DATA[p].category] || PAGE_DATA[p].category;
    if (isSkipped(p)) {{
      md += '| ' + p + ' | ' + cat + ' | ⊘ 건너뛰기 | - | - | - |\\n';
    }} else if (isRated(p)) {{
      const r = getRanks(p);
      md += '| ' + p + ' | ' + cat + ' | ✓ 평가완료 | ' + (r["1"] || '-') + ' | ' + (r["2"] || '-') + ' | ' + (r["3"] || '-') + ' |\\n';
    }} else {{
      md += '| ' + p + ' | ' + cat + ' | 미평가 | - | - | - |\\n';
    }}
  }});

  return md;
}}

function downloadReport() {{
  const md = generateReport();
  const blob = new Blob([md], {{ type: 'text/markdown;charset=utf-8' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'OCR_REPORT_' + new Date().toISOString().slice(0,10) + '.md';
  a.click();
  URL.revokeObjectURL(url);
}}

// ───────── 종합 점수 (정성 50% + 노이즈 20% + 시퀀스 30%) ─────────
function computeFinalScores() {{
  const stats = computeStatsForReport();  // 정성 평균
  return stats.map(s => {{
    const qual_norm = (s.overall || 0) / 3.0;
    const noise = NOISE_DATA[s.model];
    const clean_norm = noise ? Math.max(0, 1 - noise.ratio) : 0;
    const seq = SEQUENCE_DATA[s.model];
    const seq_norm = seq ? (seq.avg || 0) : 0;
    const final = SCORE_WEIGHTS.qual * qual_norm
                + SCORE_WEIGHTS.clean * clean_norm
                + SCORE_WEIGHTS.seq * seq_norm;
    return {{
      model: s.model,
      qual_avg: s.overall, qual_norm,
      gold: s.top.first, silver: s.top.second, bronze: s.top.third,
      score_TYPE_A: s.weak, score_TYPE_B: s.table, score_TYPE_C: s.normal,
      noise_ratio: noise ? noise.ratio : null, clean_norm,
      raw_chars: noise ? noise.raw : null, clean_chars: noise ? noise.clean : null,
      seq_avg: seq ? seq.avg : null, seq_norm,
      seq_TYPE_A: seq ? seq.by_cat.weak : null,
      seq_TYPE_B: seq ? seq.by_cat.table_suspect : null,
      seq_TYPE_C: seq ? seq.by_cat.normal : null,
      final_score: round4(final),
    }};
  }}).sort((a, b) => b.final_score - a.final_score);
}}
function round4(v) {{ return Math.round(v * 100) / 100; }}  // 사용자 요청: 소수점 2자리

// ───────── CSV 생성 + 다운로드 ─────────
function csvEscape(v) {{
  if (v === null || v === undefined) return '';
  const s = String(v);
  if (s.includes(',') || s.includes('"') || s.includes('\\n')) {{
    return '"' + s.replace(/"/g, '""') + '"';
  }}
  return s;
}}
function csvRow(arr) {{ return arr.map(csvEscape).join(',') + '\\n'; }}

// 카테고리별 정성+시퀀스 가중평균 (0~1). 종합점수와 같은 비율 (정성 0.5, 시퀀스 0.3) 정규화.
function categoryScore(qual_pt, seq_val) {{
  const wq = SCORE_WEIGHTS.qual, ws = SCORE_WEIGHTS.seq;
  if (qual_pt === null && seq_val === null) return null;
  const qual_norm = (qual_pt || 0) / 3;
  const seq_norm = seq_val || 0;
  return round4((wq * qual_norm + ws * seq_norm) / (wq + ws));
}}

function generateSimpleSummaryCSV() {{
  const scores = computeFinalScores();
  let csv = '';
  csv += '# OCR 모델 평가 — 요약\\n';
  csv += '# 생성일: ' + new Date().toISOString() + '\\n';
  csv += '# 종합점수 = 0.5×정성/3 + 0.2×(1-노이즈) + 0.3×시퀀스일치도  (0~1)\\n';
  csv += '# 카테고리 평균점수 = 정성과 시퀀스를 종합점수와 같은 비율로 가중평균한 값 (0~1)\\n';
  csv += '#   = (0.5×정성/3 + 0.3×시퀀스) / 0.8\\n';
  csv += '\\n';
  csv += csvRow(['순위','모델명','종합점수','표지목차(평균점수)','정상본문(평균점수)','표그림(평균점수)']);
  scores.forEach((s, i) => {{
    const seq = SEQUENCE_DATA[s.model] ? SEQUENCE_DATA[s.model].by_cat : {{}};
    csv += csvRow([
      i + 1, s.model, s.final_score,
      categoryScore(s.score_TYPE_A, seq.weak),
      categoryScore(s.score_TYPE_C, seq.normal),
      categoryScore(s.score_TYPE_B, seq.table_suspect),
    ]);
  }});
  return csv;
}}

function generateSummaryCSV() {{
  const scores = computeFinalScores();
  let csv = '';
  // 메타 헤더 — 한글 설명 + 분류 기준 + 가중치 정의
  csv += '# OCR 모델 종합 평가 (요약)\\n';
  csv += '# 생성일: ' + new Date().toISOString() + '\\n';
  csv += '# 페이지 분류:\\n';
  csv += '#   표지목차  = PyPDF 글자수<100 OR 한글비율<5%   (표지·목차·뒷표지)\\n';
  csv += '#   표그림    = 100<=PyPDF 글자수<500 + 한글비율 낮음 (도식·캡션 위주)\\n';
  csv += '#   정상본문  = 500<=PyPDF 글자수<1500 + 한글비율>=50% (빽빽한 본문)\\n';
  csv += '# 종합 점수 산정식:\\n';
  csv += '#   종합점수 = 0.5×정성평가_정규화 + 0.2×깨끗함_정규화 + 0.3×시퀀스일치도\\n';
  csv += '#   정성평가_정규화 = 평균순위점수 / 3        (사용자 드래그 순위: 1위=3, 2위=2, 3위=1, 4~8위=0)\\n';
  csv += '#   깨끗함_정규화   = 1 - 노이즈비율          (이미지 마크다운·HTML 태그·절대경로 자동 제거 후)\\n';
  csv += '#   시퀀스일치도    = PyPDF 단어 시퀀스 대비 일치율 (Ratcliff-Obershelp, 0~1)\\n';
  csv += '#   종합점수 범위: 0~1 (높을수록 RAG 인덱싱 친화적)\\n';
  csv += '\\n';
  const cols = [
    '순위','모델명','종합점수',
    '정성_평균점수(0~3)','정성_정규화(0~1)','1위_횟수','2위_횟수','3위_횟수',
    '정성_표지목차','정성_표그림','정성_정상본문',
    '노이즈_비율','깨끗함_정규화(0~1)','원본_글자수','정화후_글자수',
    '시퀀스_평균일치도(0~1)','시퀀스_정규화(0~1)',
    '시퀀스_표지목차','시퀀스_표그림','시퀀스_정상본문',
  ];
  csv += csvRow(cols);
  scores.forEach((s, i) => {{
    csv += csvRow([
      i + 1, s.model, s.final_score,
      s.qual_avg, round4(s.qual_norm), s.gold, s.silver, s.bronze,
      s.score_TYPE_A, s.score_TYPE_B, s.score_TYPE_C,
      s.noise_ratio, round4(s.clean_norm), s.raw_chars, s.clean_chars,
      s.seq_avg, round4(s.seq_norm), s.seq_TYPE_A, s.seq_TYPE_B, s.seq_TYPE_C,
    ]);
  }});
  return csv;
}}

function generatePagesCSV() {{
  let csv = '';
  csv += '# OCR 평가 — 페이지별 상세\\n';
  csv += '# 생성일: ' + new Date().toISOString() + '\\n';
  csv += '# 상태: 평가완료(RATED) / 건너뛰기(SKIPPED) / 미평가(UNRATED)\\n';
  csv += '# 1위·2위·3위 = 사용자가 드래그로 선택한 모델명 (4위 이하는 동순위 0pt 라 별도 표기 안 함)\\n';
  csv += '\\n';
  const cols = ['페이지번호','분류코드','분류기준','상태','1위_모델','2위_모델','3위_모델'];
  csv += csvRow(cols);
  const rule_map = {{
    weak: 'PyPDF 글자수<100 OR 한글비율<5% (저텍스트)',
    table_suspect: '100<=글자수<500 + 한글비율 낮음 (표·그림 의심)',
    normal: '500<=글자수<1500 + 한글비율>=50% (정상 본문)',
  }};
  const code_map = {{
    weak: '표지목차',
    table_suspect: '표그림',
    normal: '정상본문',
  }};
  Object.keys(PAGE_DATA).sort((a,b) => parseInt(a) - parseInt(b)).forEach(p => {{
    const cat = PAGE_DATA[p].category;
    const code = code_map[cat] || cat;
    const rule = rule_map[cat] || '';
    let status, g='', s='', b='';
    if (isSkipped(p)) {{ status = '건너뛰기'; }}
    else if (isRated(p)) {{
      const r = getRanks(p);
      status = '평가완료';
      g = r["1"] || ''; s = r["2"] || ''; b = r["3"] || '';
    }} else {{ status = '미평가'; }}
    csv += csvRow([p, code, rule, status, g, s, b]);
  }});
  return csv;
}}

// ───────── XLSX 1파일 3시트 다운로드 ─────────
// CSV 텍스트 → 2D array (단순 split, BOM/메타 # 라인은 그대로 1열에 들어감)
function csvToRows(csv) {{
  return csv.split('\\n').map(line => {{
    if (!line) return [];
    // 메타 # 라인 또는 빈 줄은 단일 셀로
    if (line.startsWith('#')) return [line];
    // CSV 파싱 (간단 — 따옴표 처리)
    const cells = [];
    let cur = '', q = false;
    for (let i = 0; i < line.length; i++) {{
      const c = line[i];
      if (q) {{
        if (c === '"' && line[i+1] === '"') {{ cur += '"'; i++; }}
        else if (c === '"') q = false;
        else cur += c;
      }} else {{
        if (c === ',') {{ cells.push(cur); cur = ''; }}
        else if (c === '"') q = true;
        else cur += c;
      }}
    }}
    cells.push(cur);
    return cells.map(v => {{
      // 숫자 자동 변환
      if (v === '') return '';
      const n = Number(v);
      return (!isNaN(n) && v.trim() !== '' && /^-?\\d/.test(v)) ? n : v;
    }});
  }});
}}

function downloadXLSX() {{
  if (typeof XLSX === 'undefined') {{
    alert('XLSX 라이브러리 로드 실패. 인터넷 연결 확인 후 새로고침하세요.');
    return;
  }}
  const date = new Date().toISOString().slice(0,10);
  const wb = XLSX.utils.book_new();

  const sheets = [
    ['요약', generateSimpleSummaryCSV()],
    ['상세', generateSummaryCSV()],
    ['페이지별', generatePagesCSV()],
  ];

  sheets.forEach(([name, csv]) => {{
    const rows = csvToRows(csv);
    const ws = XLSX.utils.aoa_to_sheet(rows);
    // 열 너비 자동 (대략)
    if (rows.length > 0) {{
      const colCount = Math.max(...rows.map(r => r.length));
      ws['!cols'] = Array.from({{length: colCount}}, (_, i) => {{
        const maxLen = Math.max(...rows.map(r => {{
          const v = r[i]; return v === undefined || v === null ? 0 : String(v).length;
        }}));
        return {{ wch: Math.min(50, Math.max(8, maxLen + 2)) }};
      }});
    }}
    XLSX.utils.book_append_sheet(wb, ws, name);
  }});

  XLSX.writeFile(wb, 'ocr_평가_' + date + '.xlsx');
}}

// ───────── Export/Import/Reset ─────────
function exportJSON() {{
  const data = JSON.stringify({{ schema: 'ocr_eval_v3_click', evaluations: EVALS,
                                exported_at: new Date().toISOString() }}, null, 2);
  const blob = new Blob([data], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'ocr_eval_' + new Date().toISOString().slice(0,10) + '.json'; a.click();
  URL.revokeObjectURL(url);
}}
function importJSON(e) {{
  const file = e.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {{
    try {{
      const data = JSON.parse(ev.target.result);
      EVALS = data.evaluations || data;
      saveEvals(EVALS);
      Object.keys(PAGE_DATA).forEach(refreshPage);
      refreshProgress();
      alert('Import 완료');
    }} catch (err) {{ alert('Import 실패: ' + err.message); }}
  }};
  reader.readAsText(file);
}}
function resetAll() {{
  if (!confirm('모든 평가를 초기화할까요?')) return;
  EVALS = {{}};
  localStorage.removeItem(STORAGE_KEY);
  Object.keys(PAGE_DATA).forEach(refreshPage);
  refreshProgress();
}}

// ───────── Lightbox (클릭 확대 + 드래그 패닝 + 휠 줌) ─────────
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxZoom = document.getElementById('lightbox-zoom');
let lbState = {{ scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 }};

function applyLbTransform() {{
  lightboxImg.style.transform =
    'translate(calc(-50% + ' + lbState.x + 'px), calc(-50% + ' + lbState.y + 'px)) scale(' + lbState.scale + ')';
  lightboxZoom.textContent = Math.round(lbState.scale * 100) + '%';
}}
function openLightbox(src) {{
  lightboxImg.src = src;
  lbState = {{ scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 }};
  applyLbTransform();
  lightbox.classList.add('open');
  document.body.style.overflow = 'hidden';
}}
function closeLightbox() {{
  lightbox.classList.remove('open');
  document.body.style.overflow = '';
}}
document.querySelectorAll('.pdf-pane img').forEach(img => {{
  img.addEventListener('click', () => openLightbox(img.src));
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape' && lightbox.classList.contains('open')) closeLightbox();
}});
lightbox.addEventListener('mousedown', e => {{
  if (e.target.id === 'lightbox-close') return;
  lbState.dragging = true;
  lbState.moved = false;
  lbState.startX = e.clientX - lbState.x;
  lbState.startY = e.clientY - lbState.y;
  lbState.downX = e.clientX;
  lbState.downY = e.clientY;
  lightbox.classList.add('dragging');
}});
lightbox.addEventListener('mousemove', e => {{
  if (!lbState.dragging) return;
  if (Math.abs(e.clientX - lbState.downX) > 5 || Math.abs(e.clientY - lbState.downY) > 5) {{
    lbState.moved = true;
  }}
  lbState.x = e.clientX - lbState.startX;
  lbState.y = e.clientY - lbState.startY;
  applyLbTransform();
}});
lightbox.addEventListener('mouseup', e => {{
  // 드래그 아닌 단순 클릭만 처리
  if (lbState.dragging && !lbState.moved && e.target.id !== 'lightbox-close') {{
    const rect = lightboxImg.getBoundingClientRect();
    const insideImg = e.clientX >= rect.left && e.clientX <= rect.right
                   && e.clientY >= rect.top && e.clientY <= rect.bottom;
    if (insideImg) {{
      // 이미지 클릭 → 2배 토글 확대
      lbState.scale = lbState.scale < 2 ? 2 : 1;
      lbState.x = 0; lbState.y = 0;
      applyLbTransform();
    }} else {{
      // 외부 클릭 → 닫기
      closeLightbox();
    }}
  }}
  lbState.dragging = false;
  lightbox.classList.remove('dragging');
}});
lightbox.addEventListener('mouseleave', () => {{
  lbState.dragging = false;
  lightbox.classList.remove('dragging');
}});
lightbox.addEventListener('wheel', e => {{
  // Ctrl + 휠 = 이미지 확대 / 축소
  if (e.ctrlKey || e.metaKey) {{
    e.preventDefault();
    const delta = -e.deltaY * 0.001;
    lbState.scale = Math.max(0.3, Math.min(8, lbState.scale * (1 + delta)));
    applyLbTransform();
    return;
  }}
  // 일반 휠 = 휠 위치 아래의 카드(또는 페이지) 스크롤
  e.preventDefault();
  lightbox.style.pointerEvents = 'none';
  const el = document.elementFromPoint(e.clientX, e.clientY);
  lightbox.style.pointerEvents = '';
  const scrollable = el ? (el.closest('.card-text') || document.scrollingElement) : document.scrollingElement;
  scrollable.scrollBy({{ top: e.deltaY, behavior: 'auto' }});
}}, {{ passive: false }});
lightbox.addEventListener('dblclick', () => {{
  lbState = {{ scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 }};
  applyLbTransform();
}});

// ───────── init ─────────
highlightAll();
Object.keys(PAGE_DATA).forEach(refreshPage);
refreshProgress();
</script>
</body>
</html>
"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ {OUT_HTML}")


if __name__ == "__main__":
    main()
