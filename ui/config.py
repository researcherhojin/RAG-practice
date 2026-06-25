# ui/config.py
# 앱 전역 설정/상수 + 공용 로거 + OpenAI 클라이언트 팩토리.

import logging
import os

from openai import OpenAI

# 답변에 사용할 OpenAI 모델. 바꾸고 싶으면 이 값만 수정하면 된다.
MODEL = "gpt-5.4-mini"

# 전 모듈이 공유하는 로거 (basicConfig 는 app.py 에서 1회 설정).
logger = logging.getLogger("rag-lab")


def get_client() -> OpenAI:
    """.env 의 OPENAI_API_KEY 로 OpenAI 클라이언트를 만든다 (Baseline 답변용)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 가 설정되어 있지 않습니다. .env 를 확인하세요.")
    return OpenAI(api_key=api_key)
