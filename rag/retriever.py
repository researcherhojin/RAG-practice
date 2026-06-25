# rag/retriever.py
# RAG Lab — Phase 6 Retriever 검색 모듈
#
# 이 파일의 역할:
#   사용자 질문을 Embedding 으로 바꿔 Chroma Vector DB(rag_docs)에서 Top-K Chunk 를
#   검색하고, 그 결과(rank/distance/metadata/preview + 답변 생성용 전체 text)를 돌려준다.
#   (RAG 답변 / Source Citation 은 Phase 7 rag/answer.py 담당. Reranker / Hybrid Search 는 아직 구현 안 함.)
#   (Streamlit 에 의존하지 않는 순수 로직만 둔다 — 화면은 app.py 담당)
#
# 보안 주의:
#   검색되어 돌아온 Chunk(Context)는 "명령"이 아니라 "데이터"로만 취급한다.
#   Context 안에 있는 어떤 문장도 시스템/모델에 대한 지시로 해석하지 않는다.

import csv
import os
from collections import Counter

import chromadb
import yaml
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Phase 5 와 동일한 설정을 재사용한다 (저장 위치·collection·모델 일치).
from rag.index import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# .env 에서만 API Key 를 읽는다.
load_dotenv()

DEFAULT_K = 4          # 기본 Top-K (요구 7)
PREVIEW_CHARS = 300    # preview 에 담을 앞부분 글자 수

# 검색 결과 CSV 컬럼.
# "score 또는 distance" 요구를 distance(코사인 거리) + score(1-거리) 둘 다로 제공.
SEARCH_RESULT_FIELDS = [
    "rank", "distance", "score",
    "source", "file_type", "parser_type", "page", "chunk_id",
    "warning", "preview",
]


def _embedding_function(api_key: str):
    """질문을 임베딩할 때 인덱싱과 동일한 모델을 쓰는 embedding function."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=EMBEDDING_MODEL,
    )


def collection_count(persist_dir=CHROMA_PATH, collection_name=COLLECTION_NAME) -> int:
    """Vector DB collection 의 chunk 개수를 돌려준다. 없으면 0 (UI 가드용)."""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        return client.get_collection(collection_name).count()
    except Exception:
        return 0


def collection_sources(persist_dir=CHROMA_PATH, collection_name=COLLECTION_NAME) -> dict:
    """인덱스에 들어있는 문서별 chunk 개수 {source: count} 를 돌려준다. 없으면 빈 dict.

    인덱스(chroma)가 업로드와 섞였는지(이전 세션 누적) 점검하는 UI 용.
    """
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        got = client.get_collection(collection_name).get(include=["metadatas"])
        return dict(Counter(m.get("source", "") for m in got["metadatas"]))
    except Exception:
        return {}


def search(query, k=DEFAULT_K,
           persist_dir=CHROMA_PATH,
           collection_name=COLLECTION_NAME,
           model=EMBEDDING_MODEL) -> list:
    """질문을 임베딩해 Top-K Chunk 를 검색한다.

    반환: rank 순 dict 리스트 (SEARCH_RESULT_FIELDS).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 가 설정되어 있지 않습니다. .env 를 확인하세요.")

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection = client.get_collection(
            collection_name,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key, model_name=model,
            ),
        )
    except Exception:
        raise ValueError(
            f"Vector DB collection '{collection_name}' 이 없습니다. 먼저 Vector DB 를 생성하세요."
        )

    # 질문을 같은 모델로 임베딩해 Top-K 검색 (요구 6).
    res = collection.query(query_texts=[query], n_results=k)

    ids = res["ids"][0]
    distances = res["distances"][0]
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]

    rows = []
    for i, (dist, meta, doc) in enumerate(zip(distances, metadatas, documents)):
        rows.append({
            "rank": i + 1,
            "distance": round(dist, 4),          # 코사인 거리 (작을수록 유사)
            "score": round(1 - dist, 4),         # 코사인 유사도 (클수록 유사)
            "source": meta.get("source", ""),
            "file_type": meta.get("file_type", ""),
            "parser_type": meta.get("parser_type", ""),
            "page": meta.get("page", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "warning": meta.get("warning", ""),
            "preview": (doc or "")[:PREVIEW_CHARS],
            # Phase 7 답변 생성용 Chunk 전체 원문. CSV(SEARCH_RESULT_FIELDS)에는
            # 담지 않고 화면/답변 생성에서만 사용한다 (CSV 컬럼 구조 유지).
            "text": doc or "",
        })
    return rows


def save_search_results(rows, path="outputs/vector_search_results.csv") -> str:
    """검색 결과를 CSV 로 덮어쓴다.

    반환: 저장한 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        # extrasaction="ignore": row 에 새로 생긴 "text" 키는 CSV 에 쓰지 않는다.
        # → vector_search_results.csv 컬럼 구조(SEARCH_RESULT_FIELDS)를 그대로 유지.
        writer = csv.DictWriter(
            f, fieldnames=SEARCH_RESULT_FIELDS, extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def load_questions(path="eval/questions.yaml") -> list:
    """평가 질문(eval/questions.yaml)을 로드한다. 없으면 빈 리스트."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("questions", [])


def expected_in_results(rows, expected_source) -> bool:
    """검색 결과 Top-K 의 source 들 중에 expected_source 가 있는지 확인한다."""
    if not expected_source:
        return False
    return expected_source in {r["source"] for r in rows}
