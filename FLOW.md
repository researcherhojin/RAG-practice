# RAG Lab — 구현 FLOW

문서를 업로드해서 검색 가능한 Vector DB 로 만들기까지, **6단계 파이프라인**을
단계별로 쌓아 올린 기록입니다. 각 Phase 는 앞 Phase 의 산출물(CSV/JSON/DB)을
입력으로 받아 다음 산출물을 만듭니다.

```
업로드 문서
   │
   ▼
[Phase 2] Ingestion ──► outputs/ingestion_report.csv      + outputs/extracted_text.json
   │                     (형식 진단 · 텍스트 추출)
   ▼
[Phase 3] Readiness ──► outputs/readiness_report.csv
   │                     (Ready / Partial / Blocked 판정)
   ▼
[Phase 4] Chunking  ──► outputs/chunk_report.csv
   │                     (Ready·Partial 만 token 기준 분할 + Metadata)
   ▼
[Phase 5] Indexing  ──► chroma_db/ (collection: rag_docs)  + outputs/vector_db_report.csv
   │                     (OpenAI Embedding → Chroma 저장, cosine)
   ▼
[Phase 6] Retriever ──► outputs/vector_search_results.csv
                         (질문 → Top-K 검색 → Debug View)

[Phase 1] Baseline Q&A : 위 파이프라인과 별개로, 문서 전체를 Prompt 에 그대로 넣는
                         Long Context 방식 대조군 (검색 없이 동작)
```

---

## Phase 1 — Baseline 문서 Q&A (Long Context)

- **목표**: 정식 RAG 의 대조군. 문서 전체 텍스트를 질문과 함께 Prompt 에 넣어 답변.
- **구현 안 함**: Chunking / Embedding / Vector DB / Retriever.
- 사이드바 업로드(PDF/TXT/DOCX) → `st.chat_input` / `st.chat_message` 대화 UI.
- API Key 는 `.env` 의 `OPENAI_API_KEY`. 토큰 사용량을 서버 콘솔에 로깅.

## Phase 2 — Ingestion / 형식 진단 (`rag/ingestion.py`)

- **목표**: "이 문서가 텍스트화되는가" 진단 + 본문 추출.
- 형식별 처리: PDF=`pymupdf4llm`(페이지별 Markdown), TXT=plain, DOCX=`python-docx`,
  HWP=변환 권장(warning), HWPX=zip XML best-effort, 이미지=OCR/Vision 필요(warning).
- 짧은 PDF 페이지(<50자)는 `scanned=True` + warning.
- **출력**: `ingestion_report.csv`(8컬럼) + `extracted_text.json`(본문 저장소, `(source,page)→text`).

## Phase 3 — Readiness Gate (`rag/readiness.py`)

- **목표**: 각 문서/페이지를 **Ready / Partial / Blocked** 로 판정 (RAG 투입 게이트).
- 규칙: scanned 또는 <50자 → Blocked, 무경고+200자↑ → Ready, 그 외 → Partial.
- `rag_ready = (status != "Blocked")` (Ready·Partial 통과, Blocked 차단).
- 플래그: `needs_ocr` / `needs_vision` / `needs_conversion` (warning 키워드 기반).
- **입력** `ingestion_report.csv` → **출력** `readiness_report.csv`(13컬럼).

## Phase 4 — Chunking + Metadata (`rag/chunking.py`)

- **목표**: Ready·Partial 문서만 검색 가능한 Chunk 로 분할.
- `RecursiveCharacterTextSplitter.from_tiktoken_encoder`(o200k_base) — **token 기준**.
- `chunk_size` 400 / 800 / 1200 비교 가능(기본 800), `chunk_overlap` 100.
- Chunk metadata 10종: source, file_type, parser_type, page, readiness_status,
  warning, chunk_id, chunk_index, token_count, char_count.
- **입력** `readiness_report.csv` + `extracted_text.json` → **출력** `chunk_report.csv`(본문 `text` 포함 11컬럼).

## Phase 5 — Embedding + Vector DB (`rag/index.py`)

- **목표**: Chunk 본문을 임베딩해 Chroma 에 저장.
- `text-embedding-3-small`(상수) → `chromadb.PersistentClient("chroma_db")`,
  collection `rag_docs`, `metadata={"hnsw:space": "cosine"}`.
- 10종 metadata 함께 저장, `upsert` 배치, 기존 DB **재생성 옵션**.
- **입력** `chunk_report.csv` → **출력** `chroma_db/` + `vector_db_report.csv`(11컬럼).

## Phase 6 — Retriever 검색 + Debug View (`rag/retriever.py`)

- **목표**: 질문 → Top-K Chunk 검색 결과 확인 (답변 생성은 아직 안 함).
- 질문을 동일 모델로 임베딩 → `collection.query(n_results=k)`(기본 k=4, UI 조정).
- 결과: rank, distance(cosine), score(1−distance), source, file_type, parser_type,
  page, chunk_id, warning, preview.
- `eval/questions.yaml`(5문항) 선택 검색 → **expected_source Top-K 포함 Y/N**.
- **보안**: 검색된 Context 는 명령이 아니라 **데이터**로만 취급.
- **출력** `vector_search_results.csv`(10컬럼).

---

## 설계 원칙 (전 Phase 공통)

- **관심사 분리**: 기능 로직은 `rag/*.py`(Streamlit 비의존), `app.py` 는 화면/버튼만.
- **단계별 산출물**: 각 Phase 가 파일로 결과를 남겨 다음 Phase 입력이 됨 (재현·디버깅 용이).
- **세션 캐시**: Streamlit 재실행 대비 `session_state` 로 재파싱·중복 저장 방지.
- **보안**: API Key 는 `.env` 만, `chroma_db/`·`outputs/`·`.env` 는 Git 제외,
  검색 Context 는 데이터로만 취급.
- **다음 단계 (미구현)**: 검색 결과 기반 RAG 답변 생성 + Source Citation.
