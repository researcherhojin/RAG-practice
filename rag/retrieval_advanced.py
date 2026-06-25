# rag/retrieval_advanced.py
# RAG Lab — Phase 9 Retrieval 고도화 모듈
#
# 이 파일의 역할:
#   기본 Vector Search(Phase 6) 위에 검색 전략을 선택적으로 얹어 비교 실험한다.
#   - Query Rewriting : 질문을 검색에 적합하게 LLM 으로 재작성
#   - Keyword Search  : chunk_report.csv 의 text 에서 키워드 매칭(외부 엔진/BM25 없이)
#   - Hybrid Search   : Vector + Keyword 결과를 RRF 로 병합
#   - Reranker        : 후보 Chunk 를 LLM 으로 재정렬
#   기존 rag/retriever.py 의 search() 는 건드리지 않고 그대로 재사용한다.
#   (Streamlit 에 의존하지 않는 순수 로직만 둔다 — 화면은 app.py 담당)
#
# 중요:
#   병합·재정렬 결과 row 는 Vector row 와 같은 키(rank/source/page/chunk_id/text ...)를
#   유지하고 rank 를 1..n 으로 다시 매긴다. 그래야 답변 생성의 [#n] Citation 이 맞는다.

import csv
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from rag.retriever import PREVIEW_CHARS, search

load_dotenv()

logger = logging.getLogger("rag-lab")

# 검색 전략 키 → 화면 라벨 (app.py selectbox 에서 사용).
STRATEGIES = [
    ("vector", "기본 Vector Search"),
    ("rewrite", "Query Rewriting + Vector Search"),
    ("hybrid", "Hybrid Search (Vector + Keyword)"),
    ("hybrid_rerank", "Hybrid Search + Reranker"),
]

RRF_K = 60   # Reciprocal Rank Fusion 상수 (작을수록 상위 rank 가중치↑)

RETRIEVAL_EXPERIMENT_FIELDS = [
    "timestamp", "query", "rewritten_query", "strategy", "top_k",
    "retrieved_sources", "retrieved_chunk_ids", "expected_source",
    "retrieval_hit", "reranker_used", "model",
]


def _client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 가 설정되어 있지 않습니다. .env 를 확인하세요.")
    return OpenAI(api_key=api_key)


def rewrite_query(query, model) -> str:
    """사용자 질문을 검색에 적합한 query 로 재작성한다. 실패하면 원본을 그대로 돌려준다."""
    try:
        response = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "당신은 검색 query 최적화 도우미입니다. 사용자의 질문을 벡터 검색에 잘 맞도록 "
                    "핵심 키워드 중심의 검색 query 한 줄로 재작성하세요. 설명 없이 query 만 출력하세요."
                )},
                {"role": "user", "content": query},
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception as e:
        # 재작성 실패 시 원본 질문으로 검색 (검색 자체는 계속 가능하게).
        logger.warning("query 재작성 실패, 원본 사용: %s", e)
        return query


def keyword_search(query, chunk_report_path="outputs/chunk_report.csv", top_k=4) -> list:
    """chunk_report.csv 의 text 에서 질문 키워드 매칭 수로 점수를 매기는 간단한 키워드 검색.

    Vector row 와 같은 키 형식으로 반환한다(score=매칭 수, rank=1..top_k).
    외부 검색 엔진/BM25 라이브러리는 쓰지 않는다.
    """
    # 질문에서 2글자 이상 토큰만 키워드로 사용 (소문자).
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 2]
    if not tokens or not os.path.exists(chunk_report_path):
        return []

    scored = []
    with open(chunk_report_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "") or ""
            text_lower = text.lower()
            # 단어 경계 매칭 — "ai" 가 "rain" 안에서 잡히는 과다 카운트 방지.
            hits = sum(len(re.findall(rf"\b{re.escape(tok)}\b", text_lower)) for tok in tokens)
            if hits > 0:
                # 길이 편향 보정 — 긴 chunk 가 단순히 길어서 유리해지지 않도록 정규화.
                score = hits / (len(text) ** 0.5)
                scored.append((score, row, text))

    # 점수 내림차순 → 상위 top_k.
    scored.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for i, (score, row, text) in enumerate(scored[:top_k]):
        rows.append({
            "rank": i + 1,
            "score": round(score, 4),            # 길이 정규화 키워드 점수 (Vector score 와 척도 다름)
            "source": row.get("source", ""),
            "file_type": row.get("file_type", ""),
            "parser_type": row.get("parser_type", ""),
            "page": row.get("page", ""),
            "chunk_id": row.get("chunk_id", ""),
            "warning": row.get("warning", ""),
            "preview": text[:PREVIEW_CHARS],
            "text": text,
            "retriever": "keyword",
        })
    return rows


def merge_results(vector_rows, keyword_rows) -> list:
    """Vector 결과와 Keyword 결과를 RRF(Reciprocal Rank Fusion)로 병합한다.

    chunk_id 로 중복 제거하고 융합 점수로 정렬, rank 를 1..n 으로 다시 매긴다.
    """
    fused = {}        # chunk_id -> 융합 점수
    rep = {}          # chunk_id -> 대표 row (Vector row 우선)
    for rows in (vector_rows, keyword_rows):
        for r in rows:
            cid = r["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r["rank"])
            # 대표 row 는 처음 본 것을 쓰되, text 가 비어 있으면 갱신.
            if cid not in rep or not rep[cid].get("text"):
                rep[cid] = r

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    merged = []
    for i, (cid, score) in enumerate(ordered):
        row = dict(rep[cid])               # 원본 보존 위해 복사
        row["rank"] = i + 1                # rank 재부여 (Citation 정확성)
        row["score"] = round(score, 6)     # RRF 점수
        row["retriever"] = "hybrid"
        merged.append(row)
    return merged


def rerank_with_llm(query, rows, model, top_n=4) -> list:
    """후보 Chunk 를 LLM 으로 관련도 재정렬한다. rank_before 를 보존하고 rank 를 다시 매긴다."""
    if not rows:
        return []

    # 후보를 번호 매겨 제시 (토큰 절약 위해 preview 길이만 사용).
    listing = "\n".join(
        f"[{r['rank']}] (source: {r['source']} · chunk_id: {r['chunk_id']})\n"
        f"{(r.get('text') or '')[:PREVIEW_CHARS]}"
        for r in rows
    )
    try:
        response = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "당신은 검색 결과 재정렬기입니다. 질문과의 관련도가 높은 순서대로 후보의 번호만 "
                    "쉼표로 나열하세요. 예: 3, 1, 2. 설명은 하지 마세요."
                )},
                {"role": "user", "content": f"[질문]\n{query}\n\n[후보]\n{listing}"},
            ],
        )
        raw = response.choices[0].message.content or ""
        order = [int(n) for n in re.findall(r"\d+", raw)]
    except Exception as e:
        logger.warning("Reranker 실패, 원래 순서 유지: %s", e)
        order = []

    # 원래 rank 로 row 를 찾기 위한 맵.
    by_rank = {r["rank"]: r for r in rows}
    reranked = []
    seen = set()
    for old_rank in order:
        if old_rank in by_rank and old_rank not in seen:
            seen.add(old_rank)
            row = dict(by_rank[old_rank])
            row["rank_before"] = old_rank          # 재정렬 전 순위 보존
            reranked.append(row)
    # LLM 이 빠뜨린 후보는 원래 순서로 뒤에 채운다 (파싱 실패 시 원본 순서 유지).
    for r in rows:
        if r["rank"] not in seen:
            row = dict(r)
            row["rank_before"] = r["rank"]
            reranked.append(row)

    reranked = reranked[:top_n]
    for i, row in enumerate(reranked):
        row["rank"] = i + 1                        # rank 재부여 (Citation 정확성)
        row["retriever"] = "rerank"
    return reranked


def run_retrieval(query, strategy, k, model) -> dict:
    """선택한 검색 전략을 실행한다.

    반환: {rewritten_query, rows, reranker_used, strategy}
      - rows 는 항상 rank 1..n 연속 (답변 생성 [#n] Citation 호환).
    """
    rewritten_query = ""
    reranker_used = False

    if strategy == "vector":
        rows = search(query, k=k)

    elif strategy == "rewrite":
        rewritten_query = rewrite_query(query, model)
        rows = search(rewritten_query, k=k)

    elif strategy in ("hybrid", "hybrid_rerank"):
        vector_rows = search(query, k=k)
        keyword_rows = keyword_search(query, top_k=k)
        rows = merge_results(vector_rows, keyword_rows)[:k]
        if strategy == "hybrid_rerank":
            rows = rerank_with_llm(query, rows, model, top_n=k)
            reranker_used = True

    else:
        raise ValueError(f"알 수 없는 검색 전략: {strategy}")

    return {
        "rewritten_query": rewritten_query,
        "rows": rows,
        "reranker_used": reranker_used,
        "strategy": strategy,
    }


def save_retrieval_experiment(record, path="outputs/retrieval_experiments.csv") -> str:
    """검색 전략 실험 결과를 CSV 에 append 한다. 파일이 없으면 헤더부터 쓴다.

    반환: 저장한 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=RETRIEVAL_EXPERIMENT_FIELDS, extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    return path
