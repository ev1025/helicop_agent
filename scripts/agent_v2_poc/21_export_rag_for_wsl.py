"""
WSL/Linux 측정용 RAG 컨텍스트 미리 추출.

Windows 에서 한 번 실행 → results/agent_v2_changes/awq_test_inputs.json 저장
→ WSL 의 vLLM 측정 스크립트가 그 JSON 로드.

WSL 안에서 chroma_db_new 다시 만들 필요 없게 만듦.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QUERIES = [
    {
        "id": "Q1",
        "query": "베르누이 원리로 양력이 어떻게 발생하나요?",
        "expected_keywords": ["베르누이", "압력", "속도", "양력"],
    },
    {
        "id": "Q2",
        "query": "Vortex Ring State 란 무엇인가요?",
        "expected_keywords": ["하강", "공기", "양력", "수직"],
    },
    {
        "id": "Q3",
        "query": "동적 롤오버는 어떤 상황에서 발생하나요?",
        "expected_keywords": ["기울", "회전", "지면", "측"],
    },
]


def main():
    from app.core import models
    from app.services.rag_service import rag_search_with_rerank

    print("[1/2] 임베딩 + Chroma 로드")
    models.load_embedding_model()
    models.load_vector_db()

    print("[2/2] 각 질의 RAG 검색 → JSON 저장")
    out_data = []
    for item in QUERIES:
        results = rag_search_with_rerank(item["query"])
        context = "\n\n".join(
            f"[p.{r.get('page')}] {r.get('content','')}" for r in results
        )
        out_data.append({
            "id": item["id"],
            "query": item["query"],
            "expected_keywords": item["expected_keywords"],
            "rag_context": context,
            "context_length": len(context),
        })
        print(f"  {item['id']}: {len(context)}자, {len(results)}개 문서")

    out_path = ROOT / "results" / "agent_v2_changes" / "awq_test_inputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")
    print(f"WSL 에서 읽을 경로: /mnt/c{str(out_path)[2:].replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
