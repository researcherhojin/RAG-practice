# rag/evaluation.py
# RAG Lab — Phase 8 Evaluation Loop 모듈
#
# 이 파일의 역할:
#   eval/questions.yaml 의 평가 질문으로 RAG 검색·답변 품질을 점검한다.
#   - 자동 점검: expected_source 가 Top-K 에 들어왔는지(retrieval_hit),
#               답변에 [#n] Citation 이 있는지(citation_present)
#   - 수동 점검: 사용자가 고른 Grounding 라벨 + 메모
#   결과를 한 줄 record 로 만들어 outputs/evaluation_report.csv 에 누적 저장한다.
#   (Streamlit 에 의존하지 않는 순수 로직만 둔다 — 화면은 app.py 담당)
#
# 설계:
#   질문 로드/검색 적중/Citation 추출은 이미 있는 함수를 재사용한다(중복 구현 금지).
#   - 질문 로드     : rag.retriever.load_questions
#   - 검색 적중 판정 : rag.retriever.expected_in_results
#   - Citation 추출  : rag.answer.extract_citation_numbers

import csv
import os

from rag.answer import extract_citation_numbers
from rag.retriever import expected_in_results, load_questions

# outputs/evaluation_report.csv 컬럼 (요구 순서 그대로).
EVAL_FIELDS = [
    "timestamp", "question_id", "question", "expected_source",
    "retrieved_sources", "retrieval_hit", "answer",
    "citation_present", "citation_refs",
    "grounding_label", "evaluator_note", "top_k", "model",
]


def load_eval_questions(path="eval/questions.yaml") -> list:
    """평가 질문(eval/questions.yaml)을 불러온다. (retriever.load_questions 재사용)"""
    return load_questions(path)


def evaluate_retrieval(rows, expected_source) -> bool:
    """expected_source 가 Top-K 검색 결과에 포함됐는지 확인한다.

    (retriever.expected_in_results 재사용)
    """
    return expected_in_results(rows, expected_source)


def evaluate_citation(answer) -> dict:
    """답변에 [#n] Citation 이 있는지 확인한다.

    반환: {"citation_present": bool, "citation_refs": [int, ...]}
    """
    refs = extract_citation_numbers(answer)
    return {"citation_present": len(refs) > 0, "citation_refs": refs}


def build_evaluation_record(question, rows, answer,
                            grounding_label, note,
                            top_k, model, timestamp) -> dict:
    """평가 결과 한 줄(record)을 구성한다.

    question: eval/questions.yaml 의 질문 dict (id/question/expected_source).
    rows: 검색 결과, answer: 생성된 답변 텍스트.
    timestamp 는 호출자(app.py)가 만들어 넘긴다 (이 모듈은 시계를 읽지 않는다).
    """
    expected_source = question.get("expected_source", "")
    # Top-K 의 source 를 중복 없이(순서 유지) 모은다.
    retrieved = list(dict.fromkeys(r["source"] for r in rows))
    citation = evaluate_citation(answer)

    return {
        "timestamp": timestamp,
        "question_id": question.get("id", ""),
        "question": question.get("question", ""),
        "expected_source": expected_source,
        "retrieved_sources": ", ".join(retrieved),
        "retrieval_hit": evaluate_retrieval(rows, expected_source),
        "answer": answer,
        "citation_present": citation["citation_present"],
        "citation_refs": ", ".join(f"#{n}" for n in citation["citation_refs"]),
        "grounding_label": grounding_label,
        "evaluator_note": note,
        "top_k": top_k,
        "model": model,
    }


def save_evaluation_report(record, path="outputs/evaluation_report.csv") -> str:
    """평가 record 를 CSV 에 append 한다 (평가 이력 누적). 파일이 없으면 헤더부터 쓴다.

    반환: 저장한 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    return path
