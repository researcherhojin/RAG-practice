# rag/readiness.py
# RAG Lab — Phase 3 Readiness Gate (RAG 투입 가능 여부 판정)
#
# 이 파일의 역할:
#   Phase 2 진단 결과(outputs/ingestion_report.csv)를 읽어서
#   각 문서/페이지를 Ready / Partial / Blocked 로 판정한다.
#   판정 결과에 needs_ocr/vision/conversion 플래그와 사람이 읽을 수 있는
#   reason 을 붙여 outputs/readiness_report.csv 로 저장한다.
#   (Streamlit 에 의존하지 않는 순수 로직만 둔다 — 화면은 app.py 담당)

import csv
import os

# 출력 CSV 컬럼 (순서 고정)
READINESS_FIELDS = [
    "source", "file_type", "parser_type", "page",
    "text_length", "scanned", "warning",
    "readiness_status", "rag_ready",
    "needs_ocr", "needs_vision", "needs_conversion", "reason",
]

# 판정 임계값.
# Phase 2 의 MIN_TEXT_PER_PAGE(=50) 와 일관되게 "매우 짧음" 기준을 둔다.
READY_MIN_LENGTH = 200   # 이 이상이면 "텍스트 충분"
MIN_USABLE_LENGTH = 50   # 이 미만이면 "매우 짧음" (사실상 추출 실패)


def classify_row(row: dict) -> dict:
    """진단 행 1개를 받아 판정 결과(13개 필드) dict 를 돌려준다."""
    # ingestion CSV 는 모든 값을 문자열로 저장하므로 형 변환을 한다.
    source = row.get("source", "")
    file_type = row.get("file_type", "")
    parser_type = row.get("parser_type", "")
    page = row.get("page", "")
    warning = (row.get("warning") or "").strip()

    try:
        text_length = int(row.get("text_length") or 0)
    except ValueError:
        text_length = 0
    scanned = str(row.get("scanned")).strip().lower() == "true"

    # warning 문구를 기반으로 필요한 후속 처리 플래그를 도출한다.
    needs_ocr = "OCR" in warning
    needs_vision = "Vision" in warning
    needs_conversion = ("변환" in warning) or ("구조 확인" in warning)

    # 상태 결정 (위에서부터 우선순위).
    if scanned or text_length < MIN_USABLE_LENGTH:
        status = "Blocked"
        if scanned:
            reason = "스캔본 의심(scanned=True) — 텍스트 레이어 없음"
        elif warning:
            # 이미지/HWP/인코딩 실패 등은 warning 에 이유가 담겨 있다.
            reason = f"추출 텍스트 부족({text_length}자): {warning}"
        else:
            reason = f"추출 텍스트 부족({text_length}자)"
    elif not warning and text_length >= READY_MIN_LENGTH:
        status = "Ready"
        reason = f"텍스트 충분({text_length}자), 경고 없음"
    else:
        status = "Partial"
        if warning:
            reason = f"텍스트는 있으나({text_length}자) 경고: {warning}"
        else:
            reason = f"텍스트가 다소 짧음({text_length}자) — 검토 권장"

    # Blocked 만 다음 단계로 넘기지 않는다 (Ready · Partial 은 통과).
    rag_ready = status != "Blocked"

    return {
        "source": source,
        "file_type": file_type,
        "parser_type": parser_type,
        "page": page,
        "text_length": text_length,
        "scanned": scanned,
        "warning": warning,
        "readiness_status": status,
        "rag_ready": rag_ready,
        "needs_ocr": needs_ocr,
        "needs_vision": needs_vision,
        "needs_conversion": needs_conversion,
        "reason": reason,
    }


def evaluate_report(input_path="outputs/ingestion_report.csv") -> list:
    """진단 CSV 를 읽어 각 행을 판정한 결과 리스트를 돌려준다.

    ingestion CSV 는 업로드마다 append 되므로, 같은 (source, page) 는
    가장 마지막(최신) 행만 사용한다.
    """
    if not os.path.exists(input_path):
        return []

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # (source, page) 기준으로 최신 행만 남긴다.
    latest = {}
    for row in rows:
        key = (row.get("source", ""), row.get("page", ""))
        latest[key] = row

    return [classify_row(row) for row in latest.values()]


def summarize(records: list) -> dict:
    """상태별 개수를 센다."""
    counts = {"Ready": 0, "Partial": 0, "Blocked": 0}
    for r in records:
        status = r.get("readiness_status")
        if status in counts:
            counts[status] += 1
    return counts


def save_readiness(records: list, path="outputs/readiness_report.csv") -> str:
    """판정 결과를 CSV 로 덮어쓴다 (현재 진단 스냅샷 기준 재생성).

    반환: 저장한 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=READINESS_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path
