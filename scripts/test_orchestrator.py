#!/usr/bin/env python3
"""
Orchestrator 테스트 스크립트

이 스크립트는 Phase 1 Step 7에서 추가되었습니다.
Orchestrator의 동작을 CLI에서 테스트할 수 있습니다.

Usage:
    python scripts/test_orchestrator.py
    python scripts/test_orchestrator.py "사용자 질문"
"""

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.orchestrator import Orchestrator
from app.core.tool_executor import ToolExecutor
from app.core.tools import FinalAnswerTool, RAGSearchTool


def setup_orchestrator() -> Orchestrator:
    """
    Orchestrator를 설정하고 반환합니다.

    Returns:
        Orchestrator: 설정된 Orchestrator 인스턴스
    """
    # ToolExecutor 준비
    executor = ToolExecutor()
    executor.register_tool(FinalAnswerTool())
    executor.register_tool(RAGSearchTool())

    # Orchestrator 생성
    orchestrator = Orchestrator(executor)

    return orchestrator


def run_test(query: str):
    """
    주어진 질문으로 Orchestrator를 테스트합니다.

    Args:
        query: 사용자 질문
    """
    print("=" * 60)
    print("Orchestrator 테스트 시작")
    print("=" * 60)

    # Orchestrator 설정
    orchestrator = setup_orchestrator()
    print(f"\n✅ Orchestrator 설정 완료: {orchestrator}")

    # 질문 실행
    print(f"\n📝 사용자 질문: {query}")
    print("\n처리 중...")
    result = orchestrator.run(query)

    # 결과 출력
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(result)

    # 대화 통계
    print("\n" + "=" * 60)
    print("대화 통계")
    print("=" * 60)
    print(f"대화 턴 수: {orchestrator.get_turn_count()}")
    print(f"메시지 수: {len(orchestrator.get_conversation().messages)}")

    # 대화 히스토리 출력
    print("\n" + "=" * 60)
    print("대화 히스토리")
    print("=" * 60)
    conv_str = orchestrator.get_conversation().to_string(include_system=False)
    print(conv_str)

    print("\n✅ 테스트 완료!")


def main():
    """메인 함수"""
    # 명령줄 인자 확인
    if len(sys.argv) > 1:
        # 명령줄에서 질문을 받은 경우
        query = " ".join(sys.argv[1:])
    else:
        # 기본 질문 사용
        query = "양력이란 무엇인가요?"

    # 테스트 실행
    run_test(query)


if __name__ == "__main__":
    main()
