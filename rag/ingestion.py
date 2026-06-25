# rag/ingestion.py
# RAG Lab — Phase 2 문서 Ingestion / 형식 진단 모듈
#
# 이 파일의 역할:
#   업로드된 파일의 형식을 진단하고, 형식별로 텍스트 추출을 시도한다.
#   각 파일(또는 PDF의 각 페이지)마다 진단 결과(record)를 만들어 돌려주고,
#   결과를 outputs/ingestion_report.csv 에 누적 저장한다.
#   (Streamlit 에 의존하지 않는 순수 로직만 둔다 — 화면은 app.py 담당)

import csv
import io
import json
import os
import re
import zipfile

import fitz  # PyMuPDF
import pymupdf4llm
from docx import Document

# 살짝 손상된 PDF(예: dict 키 오류)에서 MuPDF 가 stderr 로 쏟아내는 경고를 끈다.
# 추출은 그대로 진행되며, 콘솔 노이즈("MuPDF error: ...")만 사라진다.
fitz.TOOLS.mupdf_display_errors(False)

# 업로드 허용 확장자 (Streamlit file_uploader 의 type= 에 그대로 넘긴다)
SUPPORTED_EXTENSIONS = [
    "pdf", "txt", "docx", "hwp", "hwpx",
    "png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff",
]

# 이미지로 취급할 확장자
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}

# 진단 결과 CSV 의 컬럼 (순서 고정)
REPORT_FIELDS = [
    "source", "file_type", "parser_type", "page",
    "text_length", "scanned", "warning", "content_preview",
]

# PDF 한 페이지에서 이보다 적은 글자가 나오면 스캔본(이미지)으로 의심한다.
MIN_TEXT_PER_PAGE = 50

# content_preview 에 담을 앞부분 글자 수
PREVIEW_CHARS = 300

# HWPX(zip) 압축 해제 시 누적 허용 크기 — 압축 폭탄(zip bomb) 방어용 상한.
MAX_HWPX_DECOMPRESSED = 50 * 1024 * 1024  # 50MB

# 추출된 본문 전체를 (source, page) 단위로 저장하는 사이드카 경로.
# 진단 CSV 에는 preview 만 남기고, 다음 단계(Chunking)가 본문을 읽도록 별도 저장한다.
TEXT_STORE_PATH = "outputs/extracted_text.json"


def _preview(text: str) -> str:
    """텍스트 앞부분을 잘라 미리보기 문자열로 만든다."""
    text = (text or "").strip()
    return text[:PREVIEW_CHARS]


def _make_record(source, file_type, parser_type, page, text,
                 scanned=False, warning=""):
    """진단 record 한 개를 만든다.

    CSV 에는 8개 항목만 저장하지만(REPORT_FIELDS), record 자체에는 본문 전체를
    "text" 로 담아 둔다 (Chunking 단계가 본문을 쓸 수 있게). save_report 는
    extrasaction="ignore" 로 이 "text" 키를 CSV 에서 무시한다.
    """
    text = text or ""
    return {
        "source": source,
        "file_type": file_type,
        "parser_type": parser_type,
        "page": page,
        "text_length": len(text),
        "scanned": scanned,
        "warning": warning,
        "content_preview": _preview(text),
        "text": text,
    }


# ---------------------------------------------------------------------------
# 형식별 추출 함수
# 각 함수는 (records, text) 튜플을 돌려준다.
#   records: 진단 record 리스트 (PDF는 페이지별로 여러 개)
#   text   : Q&A 에 넘길 추출 텍스트 전체
# ---------------------------------------------------------------------------


def _ingest_pdf(data: bytes, source: str):
    """PDF: PyMuPDF4LLM 으로 페이지별 Markdown 추출을 시도한다."""
    records = []
    texts = []
    # 메모리상의 bytes 를 그대로 열어 임시 파일을 만들지 않는다.
    doc = fitz.open(stream=data, filetype="pdf")
    pages = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    for i, chunk in enumerate(pages, start=1):
        page_text = chunk.get("text", "") or ""
        texts.append(page_text)
        # 페이지 글자 수가 너무 적으면 스캔본(OCR 필요)으로 의심한다.
        scanned = len(page_text.strip()) < MIN_TEXT_PER_PAGE
        warning = "OCR 또는 Vision 필요 가능성" if scanned else ""
        records.append(_make_record(
            source, "PDF", "pymupdf4llm", i, page_text,
            scanned=scanned, warning=warning,
        ))
    return records, "\n\n".join(texts)


def _ingest_txt(data: bytes, source: str):
    """TXT: 파일 내용을 그대로(UTF-8) 읽는다."""
    text = data.decode("utf-8")
    return [_make_record(source, "TXT", "plain-text", 1, text)], text


def _ingest_docx(data: bytes, source: str):
    """DOCX: python-docx 로 문단 텍스트를 추출한다."""
    document = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs)
    return [_make_record(source, "DOCX", "python-docx", 1, text)], text


def _ingest_hwp(data: bytes, source: str):
    """HWP: 직접 파싱하지 않고 변환을 권장한다."""
    record = _make_record(
        source, "HWP", "none", 1, "",
        warning="HWPX/PDF/DOCX 변환 권장",
    )
    return [record], ""


def _ingest_hwpx(data: bytes, source: str):
    """HWPX: zip 구조를 열어 본문 XML 의 텍스트를 best-effort 로 추출한다.

    HWPX 는 OWPML(zip) 포맷이다. Contents/section*.xml 안의 태그를 제거해
    대략적인 텍스트를 얻는다. 구조가 예상과 다르거나 실패하면 경고만 남긴다.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            section_files = sorted(
                name for name in zf.namelist()
                if re.match(r"Contents/section\d+\.xml$", name)
            )
            if not section_files:
                # 예상한 본문 구조가 없으면 추출하지 않고 경고만 남긴다.
                return [_make_record(
                    source, "HWPX", "none", 1, "",
                    warning="HWPX 구조 확인 필요",
                )], ""

            # 압축 폭탄 방어: 압축 해제 누적 크기가 상한을 넘으면 거기서 멈춘다.
            total = sum(zf.getinfo(name).file_size for name in section_files)
            if total > MAX_HWPX_DECOMPRESSED:
                return [_make_record(
                    source, "HWPX", "none", 1, "",
                    warning="HWPX 가 너무 큽니다 (압축 해제 상한 초과)",
                )], ""

            parts = []
            for name in section_files:
                xml = zf.read(name).decode("utf-8", errors="ignore")
                # XML 태그를 제거해 텍스트만 남긴다 (대략적 추출).
                parts.append(re.sub(r"<[^>]+>", " ", xml))
            text = re.sub(r"\s+", " ", " ".join(parts)).strip()

        # 태그 제거 후에도 의미 있는 텍스트가 거의 없으면 구조 확인이 필요하다.
        if len(text) < MIN_TEXT_PER_PAGE:
            return [_make_record(
                source, "HWPX", "hwpx-xml", 1, text,
                warning="HWPX 구조 확인 필요",
            )], text
        return [_make_record(source, "HWPX", "hwpx-xml", 1, text)], text
    except Exception:
        # zip 이 아니거나 구조가 복잡해 실패한 경우.
        return [_make_record(
            source, "HWPX", "none", 1, "",
            warning="HWPX 구조 확인 필요",
        )], ""


def _ingest_image(data: bytes, source: str, ext: str):
    """이미지: 텍스트 추출을 시도하지 않고 OCR/Vision 필요만 알린다."""
    record = _make_record(
        source, ext.upper(), "none", 1, "",
        warning="OCR 또는 Vision 필요",
    )
    return [record], ""


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def ingest_file(file) -> dict:
    """업로드 파일을 진단·추출한다.

    반환: {"records": [record, ...], "text": "추출된 전체 텍스트"}
    """
    source = file.name
    ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
    data = file.read()  # 업로드 파일 전체를 bytes 로 읽는다.

    try:
        if ext == "pdf":
            records, text = _ingest_pdf(data, source)
        elif ext == "txt":
            records, text = _ingest_txt(data, source)
        elif ext == "docx":
            records, text = _ingest_docx(data, source)
        elif ext == "hwp":
            records, text = _ingest_hwp(data, source)
        elif ext == "hwpx":
            records, text = _ingest_hwpx(data, source)
        elif ext in IMAGE_EXTENSIONS:
            records, text = _ingest_image(data, source, ext)
        else:
            records = [_make_record(
                source, ext.upper() or "UNKNOWN", "none", 1, "",
                warning="지원하지 않는 파일 형식",
            )]
            text = ""
    except UnicodeDecodeError:
        # 주로 TXT 인코딩 문제.
        records = [_make_record(
            source, ext.upper(), "plain-text", 1, "",
            warning="텍스트 인코딩 확인 필요 (UTF-8 권장)",
        )]
        text = ""
    except Exception:
        # 형식별 추출 중 예기치 못한 오류.
        records = [_make_record(
            source, ext.upper() or "UNKNOWN", "none", 1, "",
            warning="텍스트 추출 실패 (파일 손상 여부 확인)",
        )]
        text = ""

    return {"records": records, "text": text}


def save_report(records, path="outputs/ingestion_report.csv") -> str:
    """진단 record 들을 CSV 에 append 한다. 파일이 없으면 헤더를 먼저 쓴다.

    반환: 저장한 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        # extrasaction="ignore": record 의 "text" 키 등 REPORT_FIELDS 외 항목은 무시.
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


def _store_key(source, page) -> str:
    """본문 저장소의 키 — (source, page) 를 하나의 문자열로 만든다."""
    return f"{source}\t{page}"


def save_text_store(records, path=TEXT_STORE_PATH) -> str:
    """추출 본문을 (source, page) 단위로 JSON 저장소에 upsert 한다 (최신 우선).

    같은 (source, page) 를 다시 업로드하면 최신 본문으로 덮어쓴다.
    반환: 저장소 파일 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store = load_text_store(path)
    for r in records:
        store[_store_key(r["source"], r["page"])] = r.get("text", "")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    return path


def load_text_store(path=TEXT_STORE_PATH) -> dict:
    """본문 저장소(JSON)를 로드한다. 없으면 빈 dict 를 돌려준다."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_text(store: dict, source, page) -> str:
    """저장소에서 (source, page) 의 본문을 꺼낸다."""
    return store.get(_store_key(source, page), "")
