# ui/styles.py
# 라이트 미니멀 스타일(CSS) + 공용 화면 헬퍼(섹션 헤더 · 미리보기 박스).
#
# 색은 .streamlit/config.toml 테마를 따르고, 여기선 타이포 스케일·여백·카드·
# Streamlit 기본 크롬 정리만 한다.

import html

import streamlit as st

_CSS = """
<style>
  /* Streamlit 기본 크롬 정리 (헤더 자체는 남겨 사이드바 토글 유지) */
  header[data-testid="stHeader"] {background: transparent;}
  #MainMenu, footer {visibility: hidden;}
  [data-testid="stToolbar"] {display: none;}

  /* 본문 폭/여백 — 읽기 좋은 한 칼럼 */
  .block-container {max-width: 860px; padding-top: 2.2rem; padding-bottom: 5rem;}

  /* 타이포 — 기본 헤더가 너무 굵고 큼 → 차분하게 */
  h1 {font-size: 1.65rem !important; font-weight: 650 !important; letter-spacing: -0.02em;}

  /* 섹션 헤더 (커스텀) */
  .rl-eyebrow {font-size: .72rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #B0875F;}
  .rl-title {font-size: 1.08rem; font-weight: 600; letter-spacing: -.01em;
    margin: .1rem 0 .15rem; color: #1A1A18;}
  .rl-desc {font-size: .85rem; color: #78736B; margin: 0 0 .2rem; line-height: 1.45;}

  /* 카드(테두리 컨테이너) */
  div[data-testid="stVerticalBlockBorderWrapper"] {background: #FFFFFF;}

  /* 탭 — 가볍게 */
  button[data-baseweb="tab"] {font-size: .92rem; font-weight: 500;}
  button[data-baseweb="tab"][aria-selected="true"] {font-weight: 650;}

  /* 버튼 — 둥글기·두께 정리 */
  .stButton button {border-radius: 8px; font-weight: 500;}

  /* 입력 위젯 라벨 톤 */
  label p {font-size: .85rem !important; font-weight: 500;}

  /* 추출 본문 미리보기 박스 — 원시 마크다운을 단정한 스크롤 박스로 */
  .rl-pre {max-height: 260px; overflow-y: auto; background: #FAFAF9;
    border: 1px solid #E5E2DC; border-radius: 8px; padding: .7rem .85rem;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .76rem;
    line-height: 1.5; color: #4A463F; white-space: pre-wrap; word-break: break-word;}

  /* 답변 본문 카드 */
  .rl-answer {background: #FFFCF8; border: 1px solid #ECE6DC; border-radius: 10px;
    padding: 1rem 1.15rem; font-size: .95rem; line-height: 1.6;}
</style>
"""


def inject_styles():
    """페이지 상단에 스타일을 주입한다 (set_page_config 직후 1회)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def section(title: str, desc: str = "", eyebrow: str = ""):
    """카드 안에서 쓰는 가벼운 섹션 헤더 (사용자 입력은 escape)."""
    eb = f"<div class='rl-eyebrow'>{html.escape(eyebrow)}</div>" if eyebrow else ""
    ds = f"<div class='rl-desc'>{html.escape(desc)}</div>" if desc else ""
    st.markdown(f"{eb}<div class='rl-title'>{html.escape(title)}</div>{ds}",
                unsafe_allow_html=True)


def preview_box(text: str):
    """추출 본문/Chunk 원문을 단정한 스크롤 박스로 보여준다 (원시 마크다운 escape)."""
    st.markdown(f"<div class='rl-pre'>{html.escape(text or '(없음)')}</div>",
                unsafe_allow_html=True)


def answer_box(text: str):
    """RAG 답변 본문을 카드 스타일로 보여준다 (LLM 출력 escape)."""
    st.markdown(f"<div class='rl-answer'>{html.escape(text or '')}</div>",
                unsafe_allow_html=True)
