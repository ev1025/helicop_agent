"""
벡터 스토어 관련 기능 모듈

이 모듈은 Chroma 벡터 데이터베이스 생성 및 로딩 기능을 제공합니다.
"""

import os
import json
import logging
import re
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


def _make_chroma_client(vector_db_path: str):
    """Chroma 클라이언트를 공통 설정으로 생성 (allow_reset=True로 충돌 방지)"""
    return chromadb.Client(Settings(
        is_persistent=True,
        persist_directory=vector_db_path,
        allow_reset=True,
        anonymized_telemetry=False,
    ))


def _sanitize_model_name(model_name: str) -> str:
    """
    모델명을 파일명/컬렉션명으로 사용 가능한 형태로 변환

    Args:
        model_name: 원본 모델명 (예: "intfloat/e5-large-v2")

    Returns:
        str: 정제된 모델명 (예: "intfloat_e5_large_v2")
    """
    # 슬래시, 콜론, 공백 등을 언더스코어로 변환
    sanitized = re.sub(r'[/:.\s-]+', '_', model_name)
    return sanitized


def _clean_text(text: str) -> str:
    """
    텍스트에서 과도한 개행 문자를 정리 (v1 - 기본)

    Args:
        text: 원본 텍스트

    Returns:
        str: 정리된 텍스트
    """
    # 연속된 개행 문자(\n)를 하나의 공백으로 변환
    text = re.sub(r'\n+', ' ', text)
    # 연속된 공백을 하나로 축약
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text


def _clean_text_v2(text: str) -> str:
    """
    텍스트 정리 v2 - 표 패턴 보존

    표 형태의 텍스트(숫자, 기호가 정렬된 라인)는 개행을 유지하고,
    일반 텍스트만 공백으로 병합

    Args:
        text: 원본 텍스트

    Returns:
        str: 정리된 텍스트
    """
    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        line_stripped = line.strip()

        # 빈 줄 스킵
        if not line_stripped:
            continue

        # 표 패턴 감지: 숫자나 기호가 많은 라인 (50% 이상)
        non_alpha_count = sum(1 for c in line_stripped if not c.isalpha())
        if non_alpha_count / max(len(line_stripped), 1) > 0.5:
            # 표 라인으로 판단 - 개행 유지
            processed_lines.append(line_stripped + '\n')
        else:
            # 일반 텍스트 - 공백으로 병합
            processed_lines.append(line_stripped + ' ')

    # 합치고 연속 공백 제거
    text = ''.join(processed_lines)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text


def _clean_text_v3(text: str) -> str:
    """
    텍스트 정리 v3 - 수식/기호 보존 + 표 보존

    수식 패턴, 표 패턴을 감지하여 구조를 최대한 보존

    Args:
        text: 원본 텍스트

    Returns:
        str: 정리된 텍스트
    """
    lines = text.split('\n')
    processed_lines = []

    # 수식 관련 기호 패턴
    math_symbols = r'[=+\-*/×÷≈≠<>≤≥∑∫√∏∂∇]'
    # 표 관련 패턴 (탭, 다중 공백, 구분자)
    table_pattern = r'[\t|]{2,}|\s{3,}'

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            continue

        # 수식 패턴 감지
        has_math = bool(re.search(math_symbols, line_stripped))
        # 표 패턴 감지
        has_table = bool(re.search(table_pattern, line_stripped))
        # 숫자 비율 높음
        digit_ratio = sum(1 for c in line_stripped if c.isdigit()) / max(len(line_stripped), 1)

        if has_math or has_table or digit_ratio > 0.3:
            # 구조화된 데이터 - 개행 유지
            processed_lines.append(line_stripped + '\n')
        else:
            # 일반 텍스트 - 공백으로 병합
            processed_lines.append(line_stripped + ' ')

    text = ''.join(processed_lines)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text


# 텍스트 클리닝 함수 매핑
CLEANING_FUNCTIONS = {
    "v1": _clean_text,
    "v2": _clean_text_v2,
    "v3": _clean_text_v3,
}


def get_cleaning_function(mode: str = "v1"):
    """
    텍스트 클리닝 모드에 따라 함수 반환

    Args:
        mode: "v1", "v2", "v3"

    Returns:
        함수: 클리닝 함수
    """
    return CLEANING_FUNCTIONS.get(mode, _clean_text)


def load_vector_db(embedding_function, model_name: str, vector_db_path: str, collection_name: str,
                   pdf_path: str = None, chunk_size: int = 950, chunk_overlap: int = 50,
                   text_cleaning_mode: str = "v1"):
    """
    벡터 DB 로딩/생성 + 임베딩 모델 및 청크 설정별 컬렉션 관리

    모델명과 청크 설정을 컬렉션 이름에 포함시켜 설정별로 벡터 DB를 분리 관리합니다.
    예: "new_manual" + "intfloat/e5-large-v2" + chunk_size(950) + chunk_overlap(50) + "v1"
        -> "new_manual_intfloat_e5_large_v2_950_50_v1"

    Args:
        embedding_function: 임베딩 함수 (CustomEmbeddings 인스턴스)
        model_name: 임베딩 모델 이름 (예: "intfloat/e5-large-v2")
        vector_db_path: 벡터 DB 저장 경로
        collection_name: 기본 컬렉션 이름 (모델명과 청크 설정이 자동으로 붙음)
        pdf_path: PDF 파일 경로 (컬렉션이 없을 때만 사용)
        chunk_size: 텍스트 청크 크기
        chunk_overlap: 청크 간 오버랩 크기
        text_cleaning_mode: 텍스트 클리닝 모드 ("v1", "v2", "v3")

    Returns:
        Chroma: 벡터 DB 컬렉션

    Raises:
        ValueError: PDF 경로가 없을 때
    """
    # 모델명과 청크 설정을 컬렉션 이름에 추가
    # 예: new_manual_intfloat_e5_large_v2_950_50_v1
    sanitized_model = _sanitize_model_name(model_name)
    full_collection_name = f"{collection_name}_{sanitized_model}_{chunk_size}_{chunk_overlap}_{text_cleaning_mode}"

    logger.info(f"컬렉션 이름 (모델+청크설정+클리닝): {full_collection_name}")

    # 메타데이터는 각 컬렉션별로 저장
    metadata_path = os.path.join(vector_db_path, f"_{full_collection_name}_metadata.json")

    # 벡터 DB 디렉토리 생성 (없으면)
    os.makedirs(vector_db_path, exist_ok=True)

    # Chroma DB에서 컬렉션 존재 여부 확인
    client = _make_chroma_client(vector_db_path)
    existing_collections = [col.name for col in client.list_collections()]
    collection_exists = full_collection_name in existing_collections

    if collection_exists:
        logger.info(f"기존 컬렉션 발견: {full_collection_name}")
    else:
        logger.info(f"컬렉션 없음: {full_collection_name}, 새로 생성합니다.")

    # 컬렉션이 없으면 새로 생성 + 문서 삽입
    if not collection_exists:
        if not pdf_path:
            raise ValueError("PDF 경로가 제공되지 않았습니다. 벡터 DB를 생성할 수 없습니다.")

        logger.info(f"'{full_collection_name}' 컬렉션이 없습니다. 새로 생성합니다...")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()    # PDF 문서 로딩

        # 텍스트 청킹
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = splitter.split_documents(docs)

        # 텍스트 정리: 선택된 모드로 정리
        cleaning_func = get_cleaning_function(text_cleaning_mode)
        logger.info(f"텍스트 정리 중 (모드: {text_cleaning_mode})...")
        for doc in texts:
            doc.page_content = cleaning_func(doc.page_content)

        # 배치 크기 설정 (메모리 절약을 위해)
        batch_size = 100
        total_chunks = len(texts)
        logger.info(f"전체 청크 수: {total_chunks}, 배치 크기: {batch_size}")

        # 첫 번째 배치로 컬렉션 생성
        first_batch = texts[:batch_size]
        logger.info(f"첫 번째 배치 처리 중 (0-{len(first_batch)})...")
        collection = Chroma.from_documents(
            first_batch,
            embedding=embedding_function,
            persist_directory=vector_db_path,
            collection_name=full_collection_name,  # 모델명 포함된 컬렉션명 사용
            client=client,
        )
        logger.info(f"첫 번째 배치 완료 ({len(first_batch)}개)")

        # 나머지 배치 처리
        for i in range(batch_size, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch = texts[i:batch_end]
            logger.info(f"배치 처리 중 ({i}-{batch_end})...")

            # 배치 추가
            collection.add_documents(batch)
            logger.info(f"배치 완료 ({len(batch)}개)")

            # 메모리 정리를 위해 배치 삭제
            del batch

        # 전체 texts 메모리 해제
        del texts
        del docs

        # 임베딩 모델 정보 저장
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "embedding_model": model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "text_cleaning_mode": text_cleaning_mode
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"새 컬렉션 생성 완료 — 전체 문서 수: {total_chunks}, 모델: {model_name}")

    else:
        # 기존 컬렉션이 있으면 로딩
        logger.info(f"기존 '{full_collection_name}' 컬렉션 발견, 로딩합니다...")

        # 메타데이터 확인 (있으면)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                saved_metadata = json.load(f)
                saved_model = saved_metadata.get('embedding_model', 'Unknown')
                logger.info(f"저장된 메타데이터: 모델={saved_model}, chunk_size={saved_metadata.get('chunk_size')}")

        # 기존 컬렉션 로드
        collection = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embedding_function,
            collection_name=full_collection_name,  # 모델명 포함된 컬렉션명 사용
            client=client,
        )
        logger.info(f"기존 컬렉션 로드 완료: {full_collection_name}")

    return collection


# ============================================================
# Parent Document Retriever 관련 함수
# ============================================================

def save_parent_store(parent_store: dict, vector_db_path: str, collection_name: str):
    """
    부모 문서를 JSON 파일로 저장

    Args:
        parent_store: {parent_id: document} 형태의 딕셔너리
        vector_db_path: 저장 경로
        collection_name: 컬렉션 이름
    """
    docstore_dir = os.path.join(vector_db_path, "docstore")
    os.makedirs(docstore_dir, exist_ok=True)

    docstore_path = os.path.join(docstore_dir, f"{collection_name}_parent_store.json")

    # parent_store는 이미 딕셔너리 형태이므로 바로 저장
    with open(docstore_path, 'w', encoding='utf-8') as f:
        json.dump(parent_store, f, indent=2, ensure_ascii=False)

    logger.info(f"부모 문서 저장 완료: {docstore_path} ({len(parent_store)}개)")


def load_parent_store(vector_db_path: str, collection_name: str):
    """
    부모 문서 JSON 파일 로드

    Args:
        vector_db_path: 저장 경로
        collection_name: 컬렉션 이름

    Returns:
        dict: {parent_id: {content, metadata}} 딕셔너리
    """
    docstore_path = os.path.join(vector_db_path, "docstore", f"{collection_name}_parent_store.json")

    if not os.path.exists(docstore_path):
        logger.warning(f"부모 문서 파일을 찾을 수 없습니다: {docstore_path}")
        return {}

    with open(docstore_path, 'r', encoding='utf-8') as f:
        parent_store = json.load(f)

    logger.info(f"부모 문서 로드 완료: {len(parent_store)}개")
    return parent_store


def create_parent_document_vectordb(embedding_function, model_name: str, vector_db_path: str,
                                    collection_name: str, pdf_path: str,
                                    child_chunk_size: int = 300, child_chunk_overlap: int = 30,
                                    parent_chunk_size: int = 1500, parent_chunk_overlap: int = 150,
                                    text_cleaning_mode: str = "v1"):
    """
    Parent Document Retriever용 벡터 DB 생성

    1. PDF를 큰 청크(부모)로 분할
    2. 각 부모를 작은 청크(자식)로 재분할
    3. 자식 청크만 벡터 DB에 저장 (검색용)
    4. 부모 청크는 DocStore에 JSON으로 저장 (컨텍스트용)

    Args:
        embedding_function: 임베딩 함수
        model_name: 임베딩 모델 이름
        vector_db_path: 벡터 DB 저장 경로
        collection_name: 컬렉션 이름
        pdf_path: PDF 파일 경로
        child_chunk_size: 작은 청크 크기 (검색용)
        child_chunk_overlap: 작은 청크 오버랩
        parent_chunk_size: 큰 청크 크기 (컨텍스트용)
        parent_chunk_overlap: 큰 청크 오버랩
        text_cleaning_mode: 텍스트 클리닝 모드 ("v1", "v2", "v3")

    Returns:
        tuple: (child_collection, parent_store)
    """
    from langchain_core.documents import Document

    # 컬렉션 이름 생성 (child + parent 정보 + cleaning mode 포함)
    sanitized_model = _sanitize_model_name(model_name)
    child_collection_name = f"{collection_name}_{sanitized_model}_child_{child_chunk_size}_{child_chunk_overlap}_parent_{parent_chunk_size}_{parent_chunk_overlap}_{text_cleaning_mode}"

    logger.info(f"Parent Document Retriever 모드")
    logger.info(f"자식 청크: {child_chunk_size}자 (오버랩: {child_chunk_overlap})")
    logger.info(f"부모 청크: {parent_chunk_size}자 (오버랩: {parent_chunk_overlap})")
    logger.info(f"컬렉션 이름: {child_collection_name}")

    # 벡터 DB 디렉토리 생성
    os.makedirs(vector_db_path, exist_ok=True)

    # Chroma DB에서 컬렉션 존재 여부 확인
    client = _make_chroma_client(vector_db_path)
    existing_collections = [col.name for col in client.list_collections()]
    collection_exists = child_collection_name in existing_collections

    if collection_exists:
        logger.info(f"기존 자식 컬렉션 발견: {child_collection_name}")

        # 메타데이터 확인 (chunk size 변경 여부 체크)
        metadata_path = os.path.join(vector_db_path, f"_{child_collection_name}_metadata.json")
        needs_rebuild = False

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                old_metadata = json.load(f)

            # Chunk size 변경 확인
            if (old_metadata.get('child_chunk_size') != child_chunk_size or
                old_metadata.get('child_chunk_overlap') != child_chunk_overlap or
                old_metadata.get('parent_chunk_size') != parent_chunk_size or
                old_metadata.get('parent_chunk_overlap') != parent_chunk_overlap):

                logger.warning(f"⚠️  Chunk size 변경 감지!")
                logger.warning(f"   Old: child={old_metadata.get('child_chunk_size')}/{old_metadata.get('child_chunk_overlap')}, parent={old_metadata.get('parent_chunk_size')}/{old_metadata.get('parent_chunk_overlap')}")
                logger.warning(f"   New: child={child_chunk_size}/{child_chunk_overlap}, parent={parent_chunk_size}/{parent_chunk_overlap}")
                needs_rebuild = True
        else:
            logger.warning(f"⚠️  메타데이터 파일 없음, 재생성 필요")
            needs_rebuild = True

        if needs_rebuild:
            logger.info(f"🔄 기존 컬렉션 삭제 후 재생성...")
            try:
                client.delete_collection(child_collection_name)
                logger.info(f"   ✅ 기존 컬렉션 삭제 완료")
            except Exception as e:
                logger.warning(f"   ⚠️  컬렉션 삭제 실패 (무시하고 계속): {e}")

            # Fall through to creation below (collection_exists를 False로 변경하여 새로 생성)
            collection_exists = False
        else:
            # Chunk size 동일 - 기존 컬렉션 재사용
            logger.info(f"✅ Chunk size 동일 - 기존 컬렉션 재사용")
            child_collection = Chroma(
                persist_directory=vector_db_path,
                embedding_function=embedding_function,
                collection_name=child_collection_name,
                client=client,
            )
            # 부모 문서 로드
            parent_store = load_parent_store(vector_db_path, child_collection_name)
            logger.info(f"기존 Parent Document 시스템 로드 완료")
            return child_collection, parent_store

    # 새로 생성
    logger.info(f"새 Parent Document 시스템 생성 중...")

    # 1. PDF 로드
    logger.info(f"PDF 로딩: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. 부모 청크 생성
    logger.info(f"부모 청크 생성 중 ({parent_chunk_size}자)...")
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap
    )
    parent_docs = parent_splitter.split_documents(docs)

    # 부모 문서 텍스트 정리
    cleaning_func = get_cleaning_function(text_cleaning_mode)
    for doc in parent_docs:
        doc.page_content = cleaning_func(doc.page_content)

    logger.info(f"부모 청크 생성 완료: {len(parent_docs)}개")

    # 3. 각 부모에 ID 할당 및 자식 생성
    logger.info(f"자식 청크 생성 중 ({child_chunk_size}자)...")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap
    )

    child_docs = []
    parent_store = {}

    for i, parent_doc in enumerate(parent_docs):
        parent_id = f"parent_{i}"

        # 부모에 ID 추가
        parent_doc.metadata['parent_id'] = parent_id
        parent_doc.metadata['parent_chunk_id'] = i

        # 부모 문서 저장 (딕셔너리 형태로 즉시 직렬화)
        parent_store[parent_id] = {
            "content": parent_doc.page_content,
            "metadata": parent_doc.metadata
        }

        # 부모를 자식으로 분할
        children = child_splitter.split_documents([parent_doc])

        # 각 자식에 부모 ID 추가
        for j, child in enumerate(children):
            child.metadata['parent_id'] = parent_id
            child.metadata['child_chunk_id'] = j
            child.metadata['parent_chunk_id'] = i
            # 텍스트 정리
            child.page_content = cleaning_func(child.page_content)
            child_docs.append(child)

    logger.info(f"자식 청크 생성 완료: {len(child_docs)}개")

    # 4. 자식 청크만 벡터 DB에 저장 (배치 처리)
    batch_size = 100
    total_children = len(child_docs)
    logger.info(f"자식 청크를 벡터 DB에 저장 중... (배치 크기: {batch_size})")

    # 첫 번째 배치로 컬렉션 생성
    first_batch = child_docs[:batch_size]
    logger.info(f"첫 번째 배치 처리 중 (0-{len(first_batch)})...")
    child_collection = Chroma.from_documents(
        first_batch,
        embedding=embedding_function,
        persist_directory=vector_db_path,
        collection_name=child_collection_name,
        client=client,
    )
    logger.info(f"첫 번째 배치 완료 ({len(first_batch)}개)")

    # 나머지 배치 처리
    for i in range(batch_size, total_children, batch_size):
        batch_end = min(i + batch_size, total_children)
        batch = child_docs[i:batch_end]
        logger.info(f"배치 처리 중 ({i}-{batch_end})...")
        child_collection.add_documents(batch)
        logger.info(f"배치 완료 ({len(batch)}개)")
        del batch

    # 메모리 정리
    del child_docs
    del parent_docs
    del docs

    logger.info(f"자식 청크 벡터 DB 저장 완료: {total_children}개")

    # 5. 부모 문서를 JSON으로 저장
    save_parent_store(parent_store, vector_db_path, child_collection_name)

    # 6. 메타데이터 저장
    metadata_path = os.path.join(vector_db_path, f"_{child_collection_name}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "mode": "parent_document",
            "embedding_model": model_name,
            "child_chunk_size": child_chunk_size,
            "child_chunk_overlap": child_chunk_overlap,
            "parent_chunk_size": parent_chunk_size,
            "parent_chunk_overlap": parent_chunk_overlap,
            "text_cleaning_mode": text_cleaning_mode,
            "total_parents": len(parent_store),
            "total_children": total_children
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Parent Document 시스템 생성 완료!")
    logger.info(f"  - 부모 문서: {len(parent_store)}개")
    logger.info(f"  - 자식 문서: {total_children}개")

    return child_collection, parent_store
