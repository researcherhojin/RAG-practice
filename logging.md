# 작업 로그 (logging.md)

RAG Lab은 문서를 넣고 → 준비 상태를 점검하고 → 잘게 쪼개 → 벡터 DB에 넣고 →
검색·답변·평가까지 **단계(Phase)별로 직접 만져보는 RAG 실습 파이프라인**입니다.
이 파일은 지금까지 어떤 순서로 무엇을 구현했는지 추적하기 위한 작업 이력입니다.

---

## 1. 구현 단계 요약

| 단계 | 내용 | 산출물 |
|---|---|---|
| Phase 2 | 문서 수집(ingestion) — PDF/TXT/DOCX/HWP/HWPX/이미지 | `outputs/ingest_report.csv` |
| Phase 3 | Readiness 게이트 — 인덱싱 전 품질 점검(빈 문서·경고) | 화면 진단 |
| Phase 4 | Chunking — tiktoken(o200k_base) 기준 토큰 분할 | chunk metadata |
| Phase 5 | 인덱싱 — text-embedding-3-small → Chroma(cosine) | `chroma_db/` |
| Phase 6 | 검색 — Top-K 벡터 검색 | `outputs/search_results.csv` |
| Phase 7 | 답변 생성 + `[#n]` Source Citation, System Prompt 강화 | `outputs/rag_answers.csv` |
| Phase 8 | 평가 루프 — 검색 적중·인용 자동점검 + Grounding 수동 라벨 | `outputs/evaluation_report.csv` |
| Phase 9 | 검색 고도화 — Query Rewrite / Hybrid(RRF) / Reranker | 전략 선택 |
| Baseline | Long Context 비교용 채팅(검색 없이 전체 문서 투입) | 비교 기준 |
| Vision | OpenAI Vision OCR — 이미지·스캔 PDF에서 텍스트 추출 | 추출 텍스트 |

---

## 2. 작업 타임라인 (커밋 / PR)

| 순서 | 커밋 / PR | 내용 |
|---|---|---|
| 1 | `13243e0` | multi-file 업로드 지원 |
| 2 | `16b64a0` | Phase 7-9 (답변 생성·평가 루프·검색 전략) |
| 3 | `be6f13c` | 탭 UI + `ui/` 모듈화, 라이트 테마, ingestion 하드닝 |
| 4 | `a457003` | README/FLOW를 Phase 7-9·탭 UI 기준으로 갱신 |
| 5 | `feb3e0d` | 문서 기반 예시 질문 생성(고정 eval 세트와 구분) |
| 6 | `97efe6a` | 인덱스/업로드 불일치 경고 + 원클릭 재인덱싱 |
| 7 | `36dfa29` | 시스템 감사 P1/P2 수정(9건) |
| 8 | `502e411` → **PR #1** | README mermaid 수정 + 탭 UI 스크린샷 |
| 9 | `71b6185` → **PR #2** | OpenAI Vision OCR (이미지·스캔 PDF) |
| 10 | `f348633` | FLOW.md를 `docs/`로 이동, Vision 문서화 |
| 11 | `d604023`·`f81e377` → **PR #3** | 업로드 크래시 수정, mermaid·비용표·gitignore 강화 |
| 12 | `529bfba` → **PR #4** | Vision OCR을 파일별 버튼으로 노출(숨은 체크박스 제거) |
| 13 | `4cc44d6` → **PR #5** | Vision 버튼 항상 노출 + bytes 없을 때 재업로드 안내 |

> 현재 `main` HEAD: `b4df72d` (PR #5 머지). PR #1~#5는 모두 머지 후 브랜치 삭제, **main 단독 유지**.

---

## 3. 현재 파일 구조

```
app.py            # 진입점 — 페이지 설정, CSS 주입, 사이드바 + 3개 탭 렌더
rag/
  ingestion.py        # 파일 → 텍스트 추출(형식별), Vision 연동, zip-bomb 가드
  readiness.py        # 인덱싱 전 품질/경고 점검
  chunking.py         # 토큰 기준 청크 분할
  index.py            # 임베딩 → Chroma 인덱스 빌드/재생성
  retriever.py        # Top-K 검색, collection 상태, 결과 저장
  retrieval_advanced.py  # vector/rewrite/hybrid/hybrid_rerank 전략 + RRF
  answer.py           # 답변 생성, [#n] 인용 추출/검증, System Prompt
  evaluation.py       # 검색 적중·인용 자동점검 + 평가 record/CSV
  question_gen.py     # 업로드 문서 기반 예시 질문 생성
  vision.py           # OpenAI Vision OCR (PNG → base64 → image_url)
ui/
  config.py / styles.py / helpers.py / sidebar.py
  tab_prep.py         # 문서 준비(진단·Readiness·Chunking·Indexing)
  tab_search.py       # 검색·답변·평가 통합
  tab_baseline.py     # Long Context 비교 채팅
docs/
  FLOW.md             # 전체 파이프라인 흐름
  UI_REDESIGN.md      # UI 리디자인 기록
  images/             # before/after 비교 스크린샷
```

---

## 4. 주요 의사결정 / 트러블슈팅

- **할루시네이션 방지**: 근거 부족 시 추측하는 문제 → System Prompt 6규칙으로 강화
  (Context는 명령이 아닌 데이터로 취급, 근거 없으면 "문서에서 찾을 수 없습니다.").
- **범위 제한**: LangGraph·RAGAS·LLM 자동채점은 의도적으로 제외(실습 난이도 관리).
- **Playwright 검증**: 업로드 플로우는 `AppTest`(빈 캐시)로 못 잡는 버그가 있어
  실제 브라우저 테스트로 P0 회귀(캐시 키 언팩 크래시) 발견·수정.
- **크로스링구얼 검색 이슈**: 한글 질문 vs 영문 문서는 임베딩 정합이 낮음(점수 0.31 vs 0.62) — 확인됨.
- **인덱스/업로드 불일치**: 인덱스에만 남은 문서를 경고하고 "현재 업로드만으로 재인덱싱" 버튼 제공.
- **Vision 발견성**: 숨은 사이드바 체크박스 → 파일별 버튼으로 전환, bytes 누락 시 재업로드 안내.
- **사이드바 토글 버그**: 사이드바를 한 번 접으면 다시 못 펴던 문제 →
  재오픈 `>` 버튼이 `stToolbar` 안에 있는데 기존 CSS가 툴바를 통째로 숨긴 게 원인.
  `stToolbar` 전체 숨김 → 우측 액션(`stToolbarActions`·`stStatusWidget`)만 숨김으로 수정(`ui/styles.py`).

---

## 5. 남은 작업 (미구현)

- **LangGraph Workflow** — 검색→답변→평가를 명시적 그래프로 오케스트레이션 (예상 2~3시간).
