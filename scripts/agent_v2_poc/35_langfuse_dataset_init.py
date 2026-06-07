"""
Langfuse Dataset 초기화 — 1회 실행.

agent_v2 평가용 7질의 데이터셋 생성 (28_fair_3way 셋트와 동일).

idempotent: 데이터셋이 이미 있어도 안전. 같은 input 인 아이템은 skip.

실행:
    .venv/Scripts/python.exe scripts/agent_v2_poc/35_langfuse_dataset_init.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


DATASET_NAME = "agent_v2_eval_7q"
DATASET_DESC = "헬리콥터 RAG 멀티에이전트 평가용 7질의 — 사실/절차/잡담 분포"

ITEMS = [
    {
        "id": "Q1_bernoulli",
        "input": {"query": "베르누이 원리로 양력이 어떻게 발생하나요?"},
        "expected_output": {
            "expected_route": "researcher",
            "expected_keywords": ["베르누이", "압력", "속도", "양력"],
            "kind": "사실",
        },
    },
    {
        "id": "Q2_vrs",
        "input": {"query": "Vortex Ring State 란 무엇인가요?"},
        "expected_output": {
            "expected_route": "researcher",
            "expected_keywords": ["하강", "공기", "양력", "수직"],
            "kind": "사실",
        },
    },
    {
        "id": "Q3_dynamic_rollover",
        "input": {"query": "동적 롤오버는 어떤 상황에서 발생하나요?"},
        "expected_output": {
            "expected_route": "researcher",
            "expected_keywords": ["기울", "회전", "지면", "측"],
            "kind": "사실",
        },
    },
    {
        "id": "Q4_lift_4factors",
        "input": {"query": "헬리콥터의 양력 4요소는 무엇인가요?"},
        "expected_output": {
            "expected_route": "researcher",
            "expected_keywords": ["회전", "속도", "공기", "각도"],
            "kind": "사실",
        },
    },
    {
        "id": "Q5_startup_procedure",
        "input": {"query": "헬리콥터 시동 절차를 단계별로 설명해주세요."},
        "expected_output": {
            "expected_route": "procedure",
            "expected_keywords": ["단계", "확인", "시동"],
            "kind": "절차",
        },
    },
    {
        "id": "Q6_smalltalk",
        "input": {"query": "안녕하세요"},
        "expected_output": {
            "expected_route": "smalltalk",
            "expected_keywords": ["안녕"],
            "kind": "잡담",
        },
    },
    {
        "id": "Q7_autogyro",
        "input": {"query": "오토자이로와 헬리콥터의 차이가 뭔가요?"},
        "expected_output": {
            "expected_route": "researcher",
            "expected_keywords": ["로터", "동력", "차이"],
            "kind": "어려움",
        },
    },
]


def main():
    from app.core.agent_v2.langfuse_handler import is_enabled, _init

    _init()
    if not is_enabled():
        print("❌ Langfuse 비활성화. .env 의 LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST 확인.")
        sys.exit(1)

    from langfuse import get_client
    client = get_client()

    # 데이터셋 생성 (이미 있으면 무시)
    try:
        client.create_dataset(name=DATASET_NAME, description=DATASET_DESC)
        print(f"✅ 데이터셋 생성: {DATASET_NAME}")
    except Exception as e:
        print(f"ℹ️ 데이터셋 이미 존재하거나 생성 스킵: {e}")

    # 기존 아이템 input 으로 중복 체크
    try:
        existing = client.get_dataset(name=DATASET_NAME)
        existing_inputs = {str(it.input) for it in (existing.items or [])}
    except Exception:
        existing_inputs = set()

    n_added, n_skipped = 0, 0
    for item in ITEMS:
        key = str(item["input"])
        if key in existing_inputs:
            print(f"  · skip (중복): {item['id']}")
            n_skipped += 1
            continue
        try:
            client.create_dataset_item(
                dataset_name=DATASET_NAME,
                input=item["input"],
                expected_output=item["expected_output"],
                metadata={"item_id": item["id"], "kind": item["expected_output"]["kind"]},
            )
            print(f"  + 추가: {item['id']}  ({item['input']['query'][:40]})")
            n_added += 1
        except Exception as e:
            print(f"  💥 {item['id']} 추가 실패: {e}")

    client.flush()
    print()
    print(f"=== 완료: 추가 {n_added}개, 스킵 {n_skipped}개, 총 {len(ITEMS)} 아이템 ===")
    print(f"   Langfuse UI → Datasets → {DATASET_NAME} 에서 확인")


if __name__ == "__main__":
    main()
