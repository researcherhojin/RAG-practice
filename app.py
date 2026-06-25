# app.py
# RAG Lab — 진입점. 페이지 설정/스타일/세션 초기화 후 사이드바 + 탭 3개를 그린다.
#
# 화면 로직은 ui/ 패키지에 모듈별로 나눠져 있고, RAG 로직은 rag/ 에 있다:
#   ui/sidebar.py      문서 업로드 + 진단 처리
#   ui/tab_prep.py     ① 문서 준비 (Phase 2~5)
#   ui/tab_search.py   ② 검색·답변·평가 (Phase 6~9)
#   ui/tab_baseline.py ③ Baseline (Phase 1, Long Context)

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from ui.sidebar import render_sidebar
from ui.styles import inject_styles
from ui.tab_baseline import render_baseline
from ui.tab_prep import render_prep
from ui.tab_search import render_search

# .env 에서 환경변수를 읽는다 (API Key는 .env 에서만).
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 페이지 설정 + 스타일 (첫 st 호출이어야 한다).
st.set_page_config(page_title="RAG Lab", page_icon="📄", layout="centered")
inject_styles()

# API Key 확인 — 없으면 더 진행하지 않고 안내한다.
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 가 설정되어 있지 않습니다. .env 파일에 키를 추가해주세요.")
    st.stop()

# 세션 기본값.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("ingest_cache", {})

render_sidebar()

st.title("RAG Lab")
st.caption("문서를 검색 가능한 Vector DB 로 만들고, 근거 기반으로 답변하고, 품질을 평가하는 RAG 파이프라인.")

tab_prep, tab_search, tab_baseline = st.tabs(
    ["　문서 준비　", "　검색 · 답변 · 평가　", "　Baseline　"]
)

with tab_prep:
    render_prep()
with tab_search:
    render_search()
with tab_baseline:
    render_baseline()
