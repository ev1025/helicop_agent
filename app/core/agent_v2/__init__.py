"""
GGUF (llama.cpp) 기반 RAG 챗봇 부품.

모듈:
  - tools.py                       : LangChain @tool 로 래핑된 rag_search
  - llm_qwen.py                    : Qwen / Gemma GGUF ChatModel 팩토리
  - qwen_llama_cpp_chat_model.py   : LangChain BaseChatModel 호환 llama.cpp 래퍼
  - langfuse_handler.py            : Langfuse trace (env 없으면 no-op)

운영 흐름은 app/api/routes/chat_v2.py 의 stream_answer() —
tool_choice='rag_search' 강제 → RAG → 답변 token streaming (2-step).
"""
