# ui/tab_prep.py
# ① 문서 준비 탭 — 진단 → Readiness → Chunking → Vector DB 인덱싱 (Phase 2~5).

import streamlit as st

from rag.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
    save_chunk_report,
    summarize_chunks,
)
from rag.index import build_index, check_chunk_report, save_vector_db_report
from rag.readiness import evaluate_report, save_readiness, summarize
from ui.config import logger
from ui.helpers import count_tokens
from ui.styles import preview_box, section


def render_prep():
    _render_diagnosis()
    _render_readiness()
    _render_chunking()
    _render_indexing()


def _render_diagnosis():
    with st.container(border=True):
        section("문서 진단", "업로드한 문서의 형식을 진단하고 본문을 추출합니다.", "STEP 1")
        cache = st.session_state.ingest_cache
        if not cache:
            st.info("왼쪽 사이드바에서 문서를 업로드하면 진단 결과가 여기 표시됩니다.")
            return
        for (name, _size), cached in cache.items():
            records = cached["result"]["records"]
            document_text = cached["result"]["text"]
            file_format = records[0]["file_type"]
            doc_tokens = count_tokens(document_text)

            with st.expander(
                f"{name}　·　{file_format}　·　{len(document_text):,}자 / {doc_tokens:,} tokens"
            ):
                warnings = sorted({r["warning"] for r in records if r["warning"]})
                for w in warnings:
                    st.warning(w)
                if any(r["scanned"] for r in records):
                    st.info("일부 페이지가 스캔본(이미지)으로 의심됩니다.")
                st.dataframe(
                    [
                        {"page": r["page"], "parser": r["parser_type"],
                         "chars": r["text_length"], "scanned": r["scanned"],
                         "warning": r["warning"]}
                        for r in records
                    ],
                    width="stretch", hide_index=True,
                )
                st.caption("본문 미리보기")
                preview_box(document_text[:2000])


def _render_readiness():
    with st.container(border=True):
        section("Readiness Gate",
                "Ready / Partial / Blocked 를 판정합니다 (Blocked 는 다음 단계 제외).", "STEP 2")

        if st.button("Readiness 판정 실행"):
            records = evaluate_report()
            if not records:
                st.warning("진단 결과가 없습니다. 먼저 문서를 업로드하세요.")
                st.session_state.pop("readiness", None)
            else:
                saved_path = save_readiness(records)
                counts = summarize(records)
                st.session_state.readiness = {
                    "records": records, "counts": counts, "saved": saved_path,
                }
                logger.info(
                    "READINESS | ready=%d partial=%d blocked=%d -> %s",
                    counts["Ready"], counts["Partial"], counts["Blocked"], saved_path,
                )

        readiness = st.session_state.get("readiness")
        if not readiness:
            return
        counts = readiness["counts"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Ready", counts["Ready"])
        c2.metric("Partial", counts["Partial"])
        c3.metric("Blocked", counts["Blocked"])

        show_columns = [
            "source", "page", "file_type", "text_length", "readiness_status",
            "rag_ready", "needs_ocr", "needs_vision", "needs_conversion", "warning", "reason",
        ]
        for status in ["Ready", "Partial", "Blocked"]:
            subset = [r for r in readiness["records"] if r["readiness_status"] == status]
            if not subset:
                continue
            with st.expander(f"{status} ({len(subset)})", expanded=(status == "Blocked")):
                if status == "Blocked":
                    st.error("Blocked 문서는 다음 단계로 넘기지 않습니다. (OCR/Vision/형식 변환 필요)")
                st.dataframe(
                    [{c: r[c] for c in show_columns} for r in subset],
                    width="stretch", hide_index=True,
                )
        st.caption(f"저장: `{readiness['saved']}`")


def _render_chunking():
    with st.container(border=True):
        section("Chunking", "Ready/Partial 문서를 token 기준으로 나누고 출처 Metadata 를 붙입니다.", "STEP 3")

        col_size, col_overlap = st.columns(2)
        chunk_size = col_size.selectbox(
            "chunk_size (tokens)", [400, 800, 1200],
            index=[400, 800, 1200].index(DEFAULT_CHUNK_SIZE),
        )
        chunk_overlap = col_overlap.number_input(
            "chunk_overlap (tokens)", min_value=0, max_value=chunk_size - 1,
            value=DEFAULT_CHUNK_OVERLAP, step=10,
        )

        if st.button("Chunking 실행"):
            chunks = chunk_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                st.warning("Chunk 를 만들 문서가 없습니다. 먼저 Readiness 판정으로 Ready/Partial 문서를 만드세요.")
                st.session_state.pop("chunks", None)
            else:
                saved_path = save_chunk_report(chunks)
                stats = summarize_chunks(chunks)
                st.session_state.chunks = {
                    "chunks": chunks, "stats": stats, "saved": saved_path,
                    "params": (chunk_size, chunk_overlap),
                }
                logger.info(
                    "CHUNK | size=%d overlap=%d total=%d sources=%d -> %s",
                    chunk_size, chunk_overlap, stats["total"], len(stats["by_source"]), saved_path,
                )

        chunk_state = st.session_state.get("chunks")
        if not chunk_state:
            return
        chunks = chunk_state["chunks"]
        stats = chunk_state["stats"]
        used_size, used_overlap = chunk_state["params"]

        st.metric("총 Chunk 개수", stats["total"])
        st.caption(f"chunk_size={used_size} · chunk_overlap={used_overlap} (tokens) · Blocked 제외")

        with st.expander("Chunk 목록 · source 별 개수"):
            st.dataframe(
                [{"source": s, "chunks": n} for s, n in stats["by_source"].items()],
                width="stretch", hide_index=True,
            )
            st.dataframe(
                [
                    {"chunk_id": c["chunk_id"], "page": c["page"],
                     "tokens": c["token_count"], "chars": c["char_count"], "warning": c["warning"]}
                    for c in chunks
                ],
                width="stretch", hide_index=True,
            )
        with st.expander(f"Chunk Preview (상위 10개 / 총 {len(chunks)}개)"):
            for c in chunks[:10]:
                st.markdown(
                    f"**{c['chunk_id']}** · {c['token_count']} tokens"
                    + (f" · ⚠ {c['warning']}" if c["warning"] else "")
                )
                preview_box(c["text"])
        st.caption(f"저장: `{chunk_state['saved']}`")


def _render_indexing():
    with st.container(border=True):
        section("Vector DB 인덱싱", "Chunk 본문을 Embedding 으로 변환해 Chroma 에 저장합니다.", "STEP 4")

        recreate = st.checkbox("기존 chroma_db 재생성 (collection 을 비우고 다시 만듭니다)")

        if st.button("Vector DB 생성", type="primary"):
            check = check_chunk_report()
            if not check["ok"]:
                st.warning(check["message"])
                st.session_state.pop("indexing", None)
            else:
                try:
                    with st.spinner("Embedding 생성 및 Chroma 저장 중..."):
                        summary = build_index(recreate=recreate)
                        saved_path = save_vector_db_report(summary["report_rows"])
                    summary["saved"] = saved_path
                    st.session_state.indexing = summary
                    logger.info(
                        "INDEX | read=%d indexed=%d count=%d model=%s collection=%s -> %s",
                        summary["read"], summary["indexed"], summary["count"],
                        summary["model"], summary["collection"], saved_path,
                    )
                except Exception as e:
                    st.error(f"Vector DB 생성 중 오류가 발생했습니다: {e}")
                    st.session_state.pop("indexing", None)

        indexing = st.session_state.get("indexing")
        if not indexing:
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("읽은 Chunk", indexing["read"])
        c2.metric("저장된 Chunk", indexing["indexed"])
        c3.metric("collection 총 개수", indexing["count"])
        st.caption(
            f"모델 `{indexing['model']}` · collection `{indexing['collection']}` · "
            f"저장 `{indexing['path']}/` · 리포트 `{indexing['saved']}`"
        )
