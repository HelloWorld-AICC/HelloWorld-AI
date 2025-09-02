"""
텍스트 정리 및 전처리 모듈
"""

import re
from typing import List, Dict, Any
from prefect import task, get_run_logger


@task(name="clean_text")
def clean_text_batch(texts: List[str], remove_patterns: List[str] = None) -> List[str]:
    """텍스트 배치 정리"""
    logger = get_run_logger()

    if remove_patterns is None:
        remove_patterns = [
            r"^\s*[A-Z]\.\s*",  # "A." 제거
            r"^\s*[0-9]+\.\s*",  # "1." 제거
            r"^\s*[-•]\s*",  # "-" 또는 "•" 제거
        ]

    cleaned_texts = []
    for i, text in enumerate(texts):
        cleaned = text

        # HTML 엔티티 제거
        cleaned = cleaned.replace("&nbsp;", " ").replace("\xa0", " ")
        cleaned = (
            cleaned.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        )

        # 패턴 제거
        for pattern in remove_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # 공백 정리
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*", "\n\n", cleaned)
        cleaned = cleaned.strip()

        cleaned_texts.append(cleaned)

        if i % 100 == 0:
            logger.info(f"Cleaned {i+1}/{len(texts)} texts")

    logger.info(f"Completed text cleaning for {len(texts)} texts")
    return cleaned_texts


@task(name="normalize_text")
def normalize_text(text: str) -> str:
    """단일 텍스트 정규화"""
    # 소문자 변환 (영어의 경우)
    text = text.lower()

    # 특수문자 정리
    text = re.sub(r"[^\w\s가-힣]", " ", text)

    # 연속된 공백을 하나로
    text = re.sub(r"\s+", " ", text)

    return text.strip()


@task(name="extract_keywords")
def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """텍스트에서 키워드 추출"""
    # 한글, 영어, 숫자만 추출
    words = re.findall(r"[가-힣a-zA-Z0-9]+", text)

    # 길이 필터링
    keywords = [word for word in words if len(word) >= min_length]

    return keywords
