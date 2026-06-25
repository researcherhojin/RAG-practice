# ui/helpers.py
# 화면에서 쓰는 작은 유틸 (토큰 추정 · 업로드 문서 본문 결합).

import streamlit as st
import tiktoken

from rag.retriever import collection_sources

# 토큰 개수를 추정할 때 쓰는 인코더. 최신 모델용 o200k_base 를 사용한다.
_encoder = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """주어진 텍스트가 대략 몇 토큰인지 세어 돌려준다 (추정값)."""
    return len(_encoder.encode(text))


def combined_document_text() -> str:
    """업로드된 모든 문서의 추출 본문을 이어 붙인다 (Baseline Long Context 용)."""
    cache = st.session_state.get("ingest_cache", {})
    parts = [c["result"]["text"] for c in cache.values() if c["result"]["text"].strip()]
    return "\n\n".join(parts)


def uploaded_sources() -> set:
    """이번 세션에 업로드된(진단 캐시에 있는) 문서 파일명 집합."""
    cache = st.session_state.get("ingest_cache", {})
    return {c["result"]["records"][0]["source"] for c in cache.values() if c["result"]["records"]}


def index_vs_upload():
    """인덱스 문서와 현재 업로드 문서를 비교한다.

    반환: (indexed: {source: count}, uploaded: set, extra: set)
      - extra = 인덱스에는 있지만 지금 업로드하지 않은 문서 (이전 세션 누적 → 혼동 유발)
    """
    indexed = collection_sources()
    uploaded = uploaded_sources()
    extra = set(indexed) - uploaded
    return indexed, uploaded, extra
