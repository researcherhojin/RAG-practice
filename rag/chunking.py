# rag/chunking.py
# RAG Lab — Phase 4 Chunking + Metadata 설계 모듈
#
# 이 파일의 역할:
#   Phase 3 판정 결과(outputs/readiness_report.csv)에서 Ready/Partial 문서만 골라,
#   본문 저장소(outputs/extracted_text.json)의 텍스트를 검색 가능한 Chunk 로 나눈다.
#   각 Chunk 에 출처 추적용 Metadata 를 붙이고 outputs/chunk_report.csv 로 저장한다.
#   (Embedding / Vector DB / Retriever 는 아직 사용하지 않는다.)
#   (Streamlit 에 의존하지 않는 순수 로직만 둔다 — 화면은 app.py 담당)

import csv
import os

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion import get_text, load_text_store

# chunk_report.csv 컬럼 (순서 고정).
# 요구된 10개 metadata + 실제 chunk 본문(text) 컬럼.
CHUNK_FIELDS = [
    "source", "file_type", "parser_type", "page",
    "readiness_status", "warning",
    "chunk_id", "chunk_index", "token_count", "char_count", "text",
]

# 기본 분할 설정 (chunk_size 는 400 / 800 / 1200 으로 바꿔가며 비교할 수 있다).
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Chunking 대상으로 삼을 판정 상태 (Blocked 는 제외).
ELIGIBLE_STATUS = {"Ready", "Partial"}

# 토큰 수 계산에 쓰는 인코더 (분할 기준과 동일하게 o200k_base).
# tiktoken 은 인코더 파일을 갖고 있지 않으면 실행 중에 인터넷에서 받는다.
# 프로젝트 로컬 폴더에 한 번 받아두면 오프라인에서도 동작하고,
# 네트워크가 끊겨도 인코더는 처음 쓸 때만 불러오므로 앱 자체는 켜진다.
_ENCODING_NAME = "o200k_base"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".tiktoken_cache")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.abspath(_CACHE_DIR))

_encoder = None  # 처음 쓸 때 채운다 (lazy load)


def _get_encoder():
    """o200k_base 인코더를 처음 쓸 때 한 번만 불러와 재사용한다."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoder


def chunk_documents(chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    readiness_path="outputs/readiness_report.csv",
                    store_path="outputs/extracted_text.json") -> list:
    """Ready/Partial 문서를 token 기준으로 Chunk 로 나눠 metadata 와 함께 돌려준다.

    chunk_size / chunk_overlap 은 토큰 단위다.
    """
    if not os.path.exists(readiness_path):
        return []

    with open(readiness_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    store = load_text_store(store_path)

    # 토큰 기준 분할기. chunk_size/overlap 이 '토큰 수'로 측정된다.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_ENCODING_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for row in rows:
        # Blocked 등 적격이 아닌 문서는 건너뛴다.
        if row.get("readiness_status") not in ELIGIBLE_STATUS:
            continue

        source = row.get("source", "")
        page = row.get("page", "")
        text = get_text(store, source, page)
        if not text or not text.strip():
            # 본문이 저장소에 없으면(예: 세션 만료 전 미적재) 건너뛴다.
            continue

        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append({
                # readiness 행에서 그대로 유지하는 metadata
                "source": source,
                "file_type": row.get("file_type", ""),
                "parser_type": row.get("parser_type", ""),
                "page": page,
                "readiness_status": row.get("readiness_status", ""),
                "warning": row.get("warning", ""),
                # Chunk 단위로 새로 추가하는 metadata
                "chunk_id": f"{source}_p{page}_c{i}",
                "chunk_index": i,
                "token_count": len(_get_encoder().encode(piece)),
                "char_count": len(piece),
                "text": piece,
            })
    return chunks


def summarize_chunks(chunks: list) -> dict:
    """전체 chunk 수와 source 별 chunk 수를 센다."""
    by_source = {}
    for c in chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    return {"total": len(chunks), "by_source": by_source}


def save_chunk_report(chunks: list, path="outputs/chunk_report.csv") -> str:
    """Chunk metadata 를 CSV 로 덮어쓴다.

    반환: 저장한 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHUNK_FIELDS)
        writer.writeheader()
        for c in chunks:
            writer.writerow(c)
    return path
