# ui/sidebar.py
# 사이드바 — 문서 업로드 + 형식 진단 처리 (상세 표시는 '문서 준비' 탭).

import streamlit as st

from rag.ingestion import SUPPORTED_EXTENSIONS, ingest_file, save_report, save_text_store
from rag.retriever import collection_count
from ui.config import MODEL, logger


def render_sidebar():
    """업로드된 파일을 진단·저장하고(세션 캐시), 간단한 상태를 보여준다."""
    with st.sidebar:
        st.markdown("#### 문서 업로드")
        uploaded_files = st.file_uploader(
            "PDF · TXT · DOCX · HWP · HWPX · 이미지 (여러 개 가능)",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        # 이미지·스캔 PDF 에 OpenAI Vision 적용 (호출당 비용 발생 → 기본 OFF).
        use_vision = st.checkbox("🖼 이미지·스캔 PDF 에 OpenAI Vision 적용 (비용 발생)")

        if not uploaded_files:
            st.session_state.ingest_cache = {}
        else:
            for uploaded in uploaded_files:
                # Vision 토글을 캐시 키에 포함 → 켜고 끄면 다시 추출한다.
                file_id = (uploaded.name, uploaded.size, use_vision)
                if file_id in st.session_state.ingest_cache:
                    continue
                result = ingest_file(uploaded, use_vision=use_vision)
                saved_path = save_report(result["records"])
                save_text_store(result["records"])
                st.session_state.ingest_cache[file_id] = {"result": result, "saved": saved_path}
                total_len = sum(r["text_length"] for r in result["records"])
                logger.info(
                    "INGEST | file=%s pages=%d chars=%d -> %s",
                    uploaded.name, len(result["records"]), total_len, saved_path,
                )
            st.caption(f"문서 {len(uploaded_files)}개 처리됨 · 상세는 '문서 준비' 탭")

        st.divider()
        st.caption(f"인덱싱된 Chunk: **{collection_count()}**")
        st.caption(f"모델: `{MODEL}`")
