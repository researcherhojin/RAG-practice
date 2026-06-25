# ui/tab_prep.py
# ① 문서 준비 탭 — 진단 → Readiness → Chunking → Vector DB 인덱싱 (Phase 2~5).

import os

import streamlit as st

from rag.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
    save_chunk_report,
    summarize_chunks,
)
from rag.index import build_index, check_chunk_report, clear_collection, save_vector_db_report
from rag.ingestion import ingest_bytes, save_report, save_text_store
from rag.readiness import evaluate_report, save_readiness, summarize
from ui.config import logger
from ui.helpers import count_tokens, index_vs_upload
from ui.styles import preview_box, section

# 현재 업로드 문서로 재인덱싱할 때 비우는 누적 산출물.
_PIPELINE_OUTPUTS = [
    "outputs/ingestion_report.csv", "outputs/extracted_text.json",
    "outputs/readiness_report.csv", "outputs/chunk_report.csv",
    "outputs/vector_db_report.csv",
]


def _rebuild_from_current_uploads():
    """누적 산출물·인덱스를 비우고, 지금 업로드된 문서만으로 파이프라인을 다시 돌린다.

    반환: build_index 요약 dict (chunk 가 없으면 None).
    """
    cache = st.session_state.ingest_cache
    # 1) 이전 세션 누적분 제거
    for f in _PIPELINE_OUTPUTS:
        if os.path.exists(f):
            os.remove(f)
    # 2) 현재 업로드 파일의 진단·본문만 다시 기록 (캐시에 이미 추출 결과가 있음)
    for cached in cache.values():
        records = cached["result"]["records"]
        save_report(records)
        save_text_store(records)
    # 3) readiness → chunking → index(recreate)
    rec = evaluate_report()
    saved = save_readiness(rec)
    st.session_state.readiness = {"records": rec, "counts": summarize(rec), "saved": saved}
    chunks = chunk_documents()
    if not chunks:
        # 현재 업로드가 0 chunk(전부 Blocked 등)면 인덱스를 비워 stale 문서를 제거한다.
        clear_collection()
        st.session_state.pop("chunks", None)
        st.session_state.pop("indexing", None)
        return None
    saved = save_chunk_report(chunks)
    st.session_state.chunks = {
        "chunks": chunks, "stats": summarize_chunks(chunks), "saved": saved,
        "params": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP),
    }
    summary = build_index(recreate=True)
    summary["saved"] = save_vector_db_report(summary["report_rows"])
    st.session_state.indexing = summary
    logger.info("REINDEX | sources=%d count=%d", len(cache), summary["count"])
    return summary


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
        for cached in cache.values():
            records = cached["result"]["records"]
            document_text = cached["result"]["text"]
            # 파일명은 캐시 키가 아니라 record 에서 읽는다(키 구조 변경에 안전).
            name = records[0]["source"]
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

                # 텍스트가 없고 Vision 이 필요한 파일이면, 그 자리에서 Vision 추출 버튼 제공.
                needs_vision = (
                    "data" in cached
                    and any("Vision" in (r["warning"] or "") for r in records)
                )
                if needs_vision and st.button(
                    "🖼 OpenAI Vision 으로 텍스트 추출 (비용 발생)", key=f"vision_{name}"
                ):
                    try:
                        with st.spinner("Vision 으로 텍스트 추출 중..."):
                            new_result = ingest_bytes(cached["data"], cached["name"], use_vision=True)
                        cached["result"] = new_result
                        cached["saved"] = save_report(new_result["records"])
                        save_text_store(new_result["records"])
                        logger.info("VISION_INGEST | file=%s chars=%d",
                                    name, len(new_result["text"]))
                    except Exception as e:
                        logger.error("Vision 추출 실패: %s", e)
                        st.error("Vision 추출 중 오류가 발생했습니다. .env 의 API Key 와 네트워크를 확인하세요.")
                    else:
                        st.rerun()

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
                    logger.error("Vector DB 생성 실패: %s", e)
                    st.error("Vector DB 생성 중 오류가 발생했습니다. .env 의 API Key 와 네트워크를 확인하세요.")
                    st.session_state.pop("indexing", None)

        indexing = st.session_state.get("indexing")
        if indexing:
            c1, c2, c3 = st.columns(3)
            c1.metric("읽은 Chunk", indexing["read"])
            c2.metric("저장된 Chunk", indexing["indexed"])
            c3.metric("collection 총 개수", indexing["count"])
            st.caption(
                f"모델 `{indexing['model']}` · collection `{indexing['collection']}` · "
                f"저장 `{indexing['path']}/` · 리포트 `{indexing['saved']}`"
            )

        # --- 인덱스 ↔ 업로드 일치 점검 (이전 세션 누적 혼동 방지) ---
        indexed, uploaded, extra = index_vs_upload()
        if indexed:
            st.divider()
            st.markdown("**현재 인덱스에 있는 문서**")
            for s, n in indexed.items():
                st.caption(f"• {s} — {n} chunks")
            if extra:
                st.warning(
                    "인덱스에 지금 업로드하지 않은 문서가 있습니다: "
                    + ", ".join(sorted(extra))
                    + ".\n검색·답변이 이 문서까지 포함합니다 (이전 세션에 인덱싱된 문서가 남아 있음)."
                )
                if st.button("현재 업로드 문서만으로 다시 인덱싱", type="primary"):
                    if not st.session_state.ingest_cache:
                        st.warning("현재 업로드된 문서가 없습니다. 사이드바에서 문서를 올린 뒤 다시 시도하세요.")
                    else:
                        try:
                            with st.spinner("현재 업로드 문서로 인덱스를 다시 만드는 중..."):
                                summary = _rebuild_from_current_uploads()
                        except Exception as e:
                            logger.error("재인덱싱 실패: %s", e)
                            st.error("재인덱싱 중 오류가 발생했습니다. 잠시 후 다시 시도하거나 로그를 확인하세요.")
                        else:
                            if summary:
                                st.success(f"재인덱싱 완료 — {summary['count']} chunk (현재 업로드 문서만)")
                            else:
                                st.info("현재 업로드 문서로는 인덱싱할 Chunk 가 없어 인덱스를 비웠습니다.")
                            st.rerun()
