# rag/index.py
# RAG Lab — Phase 5 Embedding + Chroma Vector DB Indexing 모듈
#
# 이 파일의 역할:
#   Phase 4 결과(outputs/chunk_report.csv)를 읽어, 각 Chunk 본문을 OpenAI
#   Embedding 으로 변환하고, 본문 + Metadata 를 Chroma Vector DB 에 저장한다.
#   (Retriever 검색 / RAG 답변 / Source Citation 은 아직 구현하지 않는다.)
#   (Streamlit 에 의존하지 않는 순수 로직만 둔다 — 화면은 app.py 담당)

import csv
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# .env 에서만 API Key 를 읽는다.
load_dotenv()

# --- 쉽게 바꿀 수 있는 설정 상수 ---
EMBEDDING_MODEL = "text-embedding-3-small"   # Embedding 모델 (요구 8·9)
CHROMA_PATH = "chroma_db"                     # Vector DB 저장 위치 (요구 6)
COLLECTION_NAME = "rag_docs"                  # Chroma collection 이름 (요구 7)
DISTANCE = "cosine"                           # 거리 기준 (요구 10)

# Chroma 에 함께 저장할 Metadata 항목 (10개).
METADATA_FIELDS = [
    "source", "file_type", "parser_type", "page",
    "readiness_status", "warning",
    "chunk_id", "chunk_index", "token_count", "char_count",
]

# 정수로 보정할 Metadata 항목.
INT_FIELDS = {"chunk_index", "token_count", "char_count"}

# chunk_report.csv 에서 실제 Chunk 본문이 담긴 컬럼 후보.
BODY_COLUMNS = ["text", "chunk_text", "content"]

# vector_db_report.csv 컬럼.
VECTOR_DB_REPORT_FIELDS = [
    "chunk_id", "source", "page", "chunk_index",
    "token_count", "char_count", "readiness_status", "warning",
    "embedding_model", "collection", "status",
]

# 한 번에 Chroma 에 upsert 할 배치 크기.
BATCH_SIZE = 100


def check_chunk_report(path="outputs/chunk_report.csv") -> dict:
    """chunk_report.csv 가 Indexing 에 쓸 수 있는 형태인지 확인한다.

    반환: {"ok": bool, "message": str, "body_column": str|None}
    """
    if not os.path.exists(path):
        return {"ok": False, "message": "chunk_report.csv 가 없습니다. 먼저 Chunking 을 실행하세요.",
                "body_column": None}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])

    if "chunk_id" not in header:
        return {"ok": False, "message": "chunk_report.csv 에 chunk_id 컬럼이 없습니다.",
                "body_column": None}

    # 실제 본문 컬럼을 찾는다 (text / chunk_text / content).
    body_column = next((c for c in BODY_COLUMNS if c in header), None)
    if body_column is None:
        if "content_preview" in header:
            return {"ok": False,
                    "message": "chunk_report.csv 에 preview 만 있고 실제 Chunk 본문 컬럼이 "
                               "없습니다. Embedding 은 본문이 필요합니다 (Chunking 단계 확인).",
                    "body_column": None}
        return {"ok": False,
                "message": "chunk_report.csv 에 본문 컬럼(text/chunk_text/content)이 없습니다.",
                "body_column": None}

    return {"ok": True, "message": "ok", "body_column": body_column}


def _coerce_metadata(row: dict) -> dict:
    """Metadata 를 Chroma 가 허용하는 타입(str/int/float/bool)으로 보정한다."""
    meta = {}
    for field in METADATA_FIELDS:
        value = row.get(field, "")
        if field in INT_FIELDS:
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0
        else:
            # None 금지 — 빈 문자열로.
            value = "" if value is None else str(value)
        meta[field] = value
    return meta


def clear_collection(persist_dir=CHROMA_PATH, collection_name=COLLECTION_NAME):
    """collection 을 통째로 삭제한다 (현재 업로드가 0 chunk 일 때 인덱스를 비우는 용도)."""
    try:
        chromadb.PersistentClient(path=persist_dir).delete_collection(collection_name)
    except Exception:
        pass  # 아직 없으면 무시.


def build_index(recreate=False,
                chunk_report_path="outputs/chunk_report.csv",
                persist_dir=CHROMA_PATH,
                collection_name=COLLECTION_NAME,
                model=EMBEDDING_MODEL) -> dict:
    """chunk_report.csv 의 Chunk 본문을 임베딩해 Chroma 에 저장한다.

    반환: 요약 dict (read/indexed/model/collection/path/count/report_rows)
    """
    check = check_chunk_report(chunk_report_path)
    if not check["ok"]:
        raise ValueError(check["message"])
    body_column = check["body_column"]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 가 설정되어 있지 않습니다. .env 를 확인하세요.")

    with open(chunk_report_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    client = chromadb.PersistentClient(path=persist_dir)

    # 재생성 옵션: 기존 collection 을 지우고 새로 만든다 (요구 11).
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            # 아직 없으면 무시.
            pass

    emb_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": DISTANCE},   # 거리 기준 cosine 명시
    )

    # 임베딩 대상: 실제 Chunk 본문 (preview 아님, 요구 확인 4).
    ids, documents, metadatas, report_rows = [], [], [], []
    for row in rows:
        body = (row.get(body_column) or "").strip()
        if not body:
            continue  # 본문이 없으면 건너뛴다.
        ids.append(row["chunk_id"])
        documents.append(body)
        metadatas.append(_coerce_metadata(row))
        report_rows.append({
            "chunk_id": row.get("chunk_id", ""),
            "source": row.get("source", ""),
            "page": row.get("page", ""),
            "chunk_index": row.get("chunk_index", ""),
            "token_count": row.get("token_count", ""),
            "char_count": row.get("char_count", ""),
            "readiness_status": row.get("readiness_status", ""),
            "warning": row.get("warning", ""),
            "embedding_model": model,
            "collection": collection_name,
            "status": "indexed",
        })

    # 배치로 나눠 임베딩 + 저장 (upsert 로 중복 id 도 안전하게).
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return {
        "read": len(rows),
        "indexed": len(ids),
        "model": model,
        "collection": collection_name,
        "path": persist_dir,
        "count": collection.count(),
        "report_rows": report_rows,
    }


def save_vector_db_report(rows, path="outputs/vector_db_report.csv") -> str:
    """Indexing 결과를 CSV 로 덮어쓴다.

    반환: 저장한 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VECTOR_DB_REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
