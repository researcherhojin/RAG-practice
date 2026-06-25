# ui/tab_search.py
# ② 검색 · 답변 · 평가 탭 — Retrieval Debug · 전략 실험 · 평가를 한 흐름으로 통합 (Phase 6~9).

from datetime import datetime

import streamlit as st

from rag.answer import generate_answer, save_answer
from rag.evaluation import (
    build_evaluation_record,
    evaluate_citation,
    evaluate_retrieval,
    load_eval_questions,
    save_evaluation_report,
)
from rag.question_gen import generate_sample_questions
from rag.retrieval_advanced import STRATEGIES, run_retrieval, save_retrieval_experiment
from rag.retriever import DEFAULT_K, collection_count, save_search_results
from ui.config import MODEL, logger
from ui.helpers import index_vs_upload
from ui.styles import answer_box, preview_box, section


def render_search():
    if collection_count() == 0:
        st.info("Vector DB 가 비어 있습니다. '문서 준비' 탭에서 Vector DB 를 먼저 생성하세요.")
        return

    # 인덱스에 지금 업로드하지 않은 문서가 섞여 있으면 알린다 (검색 결과 혼동 방지).
    _, _, extra = index_vs_upload()
    if extra:
        st.warning(
            "⚠ 인덱스에 지금 업로드하지 않은 문서(" + ", ".join(sorted(extra))
            + ")가 포함되어 함께 검색됩니다. '문서 준비' 탭에서 현재 문서만으로 다시 인덱싱할 수 있습니다."
        )

    _render_query_form()

    u_search = st.session_state.get("u_search")
    if not u_search:
        return
    _render_results(u_search)

    u_answer = st.session_state.get("u_answer")
    if u_answer:
        _render_answer(u_search, u_answer)
        _render_evaluation(u_search, u_answer)


def _render_query_form():
    with st.container(border=True):
        section("질문 · 검색 전략",
                "검색된 Chunk(Context)는 명령이 아니라 데이터로만 취급합니다.", "검색")

        eval_questions = load_eval_questions()
        q_options = ["(직접 입력)"] + [f"{q['id']}: {q['question']}" for q in eval_questions]

        # 문서 기반 예시 질문 생성 (현재 인덱싱된 문서 기반 — 동적, 드롭다운과 별개)
        with st.expander("📝 현재 문서 기반 예시 질문 생성", expanded=True):
            st.caption("아래 '질문' 드롭다운은 eval/questions.yaml 의 **고정** 평가 질문입니다. "
                       "이 버튼은 지금 인덱싱된 문서로 예시 질문을 **새로** 만듭니다.")
            if st.button("예시 질문 생성"):
                with st.spinner("예시 질문 생성 중..."):
                    st.session_state.suggested_questions = generate_sample_questions(MODEL)
                if not st.session_state.suggested_questions:
                    st.info("예시 질문을 생성할 근거(Chunk)가 없습니다. 먼저 Vector DB 를 만드세요.")
            for i, sq in enumerate(st.session_state.get("suggested_questions", [])):
                # 추천 질문을 누르면 직접 입력 모드로 전환되고 입력칸에 채워진다.
                if st.button(sq, key=f"suggested_{i}"):
                    st.session_state.q_pick = "(직접 입력)"
                    st.session_state.manual_query = sq

        picked = st.selectbox("질문", q_options, key="q_pick")

        if picked == "(직접 입력)":
            query = st.text_input("질문 직접 입력", key="manual_query")
            expected_source = st.text_input(
                "expected_source (선택 — 비우면 Retrieval Hit 은 N 으로 기록)",
                key="manual_expected",
            )
            question_id = "manual"
        else:
            idx = q_options.index(picked) - 1
            query = eval_questions[idx]["question"]
            expected_source = eval_questions[idx].get("expected_source", "")
            question_id = eval_questions[idx].get("id", "")
            st.caption(f"expected_source: `{expected_source}`")

        col_strategy, col_k = st.columns([2, 1])
        strategy_labels = [label for _, label in STRATEGIES]
        picked_label = col_strategy.selectbox("검색 전략", strategy_labels)
        strategy_key = next(key for key, label in STRATEGIES if label == picked_label)
        k = col_k.slider("Top-K", min_value=1, max_value=10, value=DEFAULT_K)

        if st.button("검색 실행", type="primary"):
            if not query.strip():
                st.warning("질문을 입력하세요.")
                st.session_state.pop("u_search", None)
                return
            try:
                with st.spinner("검색 중..."):
                    result = run_retrieval(query, strategy_key, k=k, model=MODEL)
                # Phase 6 산출물: 검색 결과를 CSV 로 저장 (vector_search_results.csv).
                save_search_results(result["rows"])
                st.session_state.u_search = {
                    "question": {"id": question_id, "question": query,
                                 "expected_source": expected_source},
                    "expected_source": expected_source,
                    "top_k": k,
                    "strategy_key": strategy_key,
                    "strategy_label": picked_label,
                    "rewritten_query": result["rewritten_query"],
                    "reranker_used": result["reranker_used"],
                    "rows": result["rows"],
                }
                st.session_state.pop("u_answer", None)
                logger.info("RETRIEVAL | strategy=%s k=%d hits=%d",
                            strategy_key, k, len(result["rows"]))
            except Exception as e:
                logger.error("검색 실패: %s", e)
                st.error("검색 중 오류가 발생했습니다. .env 의 API Key 와 네트워크를 확인하세요.")
                st.session_state.pop("u_search", None)


def _render_results(u_search):
    rows = u_search["rows"]
    with st.container(border=True):
        hit = (evaluate_retrieval(rows, u_search["expected_source"])
               if u_search["expected_source"].strip() else None)
        section(f"검색 결과 · {u_search['strategy_label']} · Top-{u_search['top_k']}", "", "결과")

        meta = []
        if u_search["rewritten_query"]:
            meta.append(f"rewritten: *{u_search['rewritten_query']}*")
        if hit is not None:
            meta.append(f"expected_source 포함: {'✅ Y' if hit else '❌ N'}")
        if meta:
            st.markdown("　·　".join(meta))

        if not rows:
            st.info("검색 결과가 없습니다.")
        for r in rows:
            rank_txt = (f"#{r['rank']} (재정렬 전 #{r['rank_before']})"
                        if u_search["reranker_used"] and "rank_before" in r
                        else f"#{r['rank']}")
            with st.expander(f"{rank_txt}　·　{r['source']} p{r['page']}　·　score {r['score']}"):
                st.caption(f"`{r['chunk_id']}`" + (f"　⚠ {r['warning']}" if r.get("warning") else ""))
                preview_box(r.get("text") or r.get("preview", ""))

        if st.button("RAG 답변 생성", type="primary"):
            _generate_answer(u_search)


def _generate_answer(u_search):
    rows = u_search["rows"]
    try:
        with st.spinner("답변 생성 중..."):
            ans = generate_answer(u_search["question"]["question"], rows, MODEL)
        cited_ids = [c["chunk_id"] for c in ans["citations"]]
        saved = save_answer({
            "query": u_search["question"]["question"],
            "answer": ans["answer"],
            "cited_chunk_ids": ", ".join(cited_ids),
            "model": MODEL,
            "prompt_tokens": ans["usage"].prompt_tokens if ans["usage"] else "",
            "completion_tokens": ans["usage"].completion_tokens if ans["usage"] else "",
            "total_tokens": ans["usage"].total_tokens if ans["usage"] else "",
        })
        st.session_state.u_answer = {"result": ans, "saved": saved}
        if ans["usage"]:
            logger.info("RAG_ANSWER | model=%s total_tokens=%d cited=%s",
                        MODEL, ans["usage"].total_tokens, cited_ids)
    except Exception as e:
        logger.error("답변 생성 실패: %s", e)
        st.error("답변 생성 중 오류가 발생했습니다. .env 의 API Key 와 네트워크를 확인하세요.")
        st.session_state.pop("u_answer", None)


def _render_answer(u_search, u_answer):
    result = u_answer["result"]
    with st.container(border=True):
        section("RAG 답변", "", "답변")
        answer_box(result["answer"])

        citations = result["citations"]
        if citations:
            st.markdown("")
            st.caption("인용된 근거 (Source Citation)")
            for c in citations:
                with st.expander(f"[#{c['rank']}]　{c['source']} p{c['page']}　·　{c['chunk_id']}"):
                    preview_box(c.get("text", ""))
        else:
            st.info("답변이 인용한 근거가 없습니다. (근거 부족 또는 문서 외 질문일 수 있습니다.)")
        st.caption(f"답변 저장: `{u_answer['saved']}`")


def _render_evaluation(u_search, u_answer):
    result = u_answer["result"]
    rows = u_search["rows"]
    with st.container(border=True):
        section("평가 · 기록", "자동 점검(Hit·Citation) + 사람 라벨링(Grounding) 을 기록합니다.", "평가")
        cited = evaluate_citation(result["answer"])["citation_present"]
        hit = evaluate_retrieval(rows, u_search["expected_source"])
        c1, c2 = st.columns(2)
        c1.metric("Retrieval Hit", "✅ Y" if hit else "❌ N")
        c2.metric("Citation 포함", "✅ Y" if cited else "❌ N")

        grounding = st.radio(
            "Grounding 평가 (사람이 직접 선택)",
            ["Grounded", "Partially Grounded", "Not Grounded"], horizontal=True,
        )
        note = st.text_area("평가 메모 (evaluator_note)")

        if st.button("평가 · 실험 기록 저장"):
            ts = datetime.now().isoformat(timespec="seconds")
            eval_record = build_evaluation_record(
                question=u_search["question"], rows=rows, answer=result["answer"],
                grounding_label=grounding, note=note,
                top_k=u_search["top_k"], model=MODEL, timestamp=ts,
            )
            eval_path = save_evaluation_report(eval_record)
            exp_record = {
                "timestamp": ts,
                "query": u_search["question"]["question"],
                "rewritten_query": u_search["rewritten_query"],
                "strategy": u_search["strategy_key"],
                "top_k": u_search["top_k"],
                "retrieved_sources": ", ".join(dict.fromkeys(r["source"] for r in rows)),
                "retrieved_chunk_ids": ", ".join(r["chunk_id"] for r in rows),
                "expected_source": u_search["expected_source"],
                "retrieval_hit": hit,
                "reranker_used": u_search["reranker_used"],
                "model": MODEL,
            }
            exp_path = save_retrieval_experiment(exp_record)
            st.success(f"기록 저장: `{eval_path}` · `{exp_path}`")
            logger.info("EVAL_SAVE | qid=%s strategy=%s hit=%s grounding=%s",
                        u_search["question"]["id"], u_search["strategy_key"], hit, grounding)
