"""
Supervisor + Workers 멀티에이전트.

분석정리.md §10.1 / CHANGES.md #006 참조.

구성:
  supervisor.py  : 라우팅 결정 (질문 종류 → Worker 선택)
  researcher.py  : 일반 사실 검색 + 답변 (rag_search 도구)
  procedure.py   : 절차/단계별 답변 (rag_search 도구, 답변 형식이 단계 위주)
  critic.py      : 답변 품질 평가 (구조화 출력)
  graph.py 는 상위 레벨에서 이들을 조립 (별도 파일)

모든 Worker 는 같은 QwenChatModel 인스턴스를 공유한다 (메모리 절약).
시스템 프롬프트와 도구 바인딩만 다르다.
"""

# Researcher / Procedure 워커가 공유하는 "검색 쿼리 추출용" 시스템 프롬프트.
# 워커 시스템 프롬프트(답변 작성용)와 분리 — 첫 LLM 호출(tool_choice="rag_search" 강제) 은
# query 인자만 정하면 되므로 답변 작성 지시 / 단계별 형식 지시 등이 들어가면 토큰 낭비 + 역할 혼란.
SEARCH_QUERY_SYSTEM = (
    "당신은 검색 쿼리 추출기입니다. 사용자 질문에서 헬리콥터 매뉴얼 검색에 쓸 "
    "핵심 명사 키워드만 뽑아 rag_search 도구의 query 인자로 호출하세요.\n"
    "\n"
    "규칙:\n"
    "- 명사·전문용어만 추출 (조사·동사·서술어·문서형식 단어는 절대 제외)\n"
    "- 금지어: 'PDF', '문서', '매뉴얼', '교재', '알려주세요', '설명', '에 대해', '에 대한'\n"
    "- 2~5단어, 30자 이내\n"
    "\n"
    "예시:\n"
    "- 사용자: '베르누이 정리에 대해 알려주세요' → query: '베르누이 정리'\n"
    "- 사용자: '헬리콥터 시동 절차를 단계별로 설명해주세요' → query: '헬리콥터 시동 절차'\n"
    "- 사용자: 'Vortex Ring State 란 무엇인가요?' → query: 'Vortex Ring State'\n"
    "- 사용자: '동적 롤오버는 언제 발생하나요' → query: '동적 롤오버'\n"
    "\n"
    "답변 생성은 절대 하지 않습니다."
)
