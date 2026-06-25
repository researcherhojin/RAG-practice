# rag/vision.py
# RAG Lab — OpenAI Vision 으로 이미지/스캔 페이지에서 텍스트 추출.
#
# 이 파일의 역할:
#   이미지 bytes 를 OpenAI 멀티모달 모델에 보내 본문 텍스트를 추출한다.
#   이미지 파일과 스캔 PDF 페이지(래스터화한 이미지)에 공통으로 쓴다.
#   (Streamlit 에 의존하지 않는 순수 로직 — 화면은 ui/ 담당)
#
# 보안/비용:
#   이미지 1장당 API 호출 1회(비용 발생). 스캔 PDF 는 페이지 수만큼 호출되므로
#   호출 측(ingestion)에서 MAX_VISION_PAGES 로 페이지 수를 제한한다.

import base64
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("rag-lab")

# 이미지 입력이 가능한 멀티모달 모델 (답변 모델과 동일하게 둔다).
VISION_MODEL = "gpt-5.4-mini"

# 스캔 PDF 한 문서에서 Vision 을 적용할 최대 페이지 수 (비용 상한).
MAX_VISION_PAGES = 20

_PROMPT = (
    "이 이미지에 보이는 모든 텍스트를 그대로 추출해 주세요. "
    "설명이나 해석 없이 본문 텍스트만, 원래 줄바꿈을 유지해 출력하세요. "
    "텍스트가 없으면 아무것도 출력하지 마세요."
)


def extract_text(image_png_bytes: bytes, model: str = VISION_MODEL) -> str:
    """PNG 이미지 bytes 를 Vision 모델에 보내 텍스트를 추출한다. 실패하면 빈 문자열.

    호출 측에서 어떤 이미지든 PNG 로 변환해 넘긴다 (형식 호환성).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 가 설정되어 있지 않습니다. .env 를 확인하세요.")

    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    try:
        response = OpenAI(api_key=api_key).chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Vision 추출 실패: %s", e)
        return ""
