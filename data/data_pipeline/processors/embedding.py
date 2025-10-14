"""OpenAI 임베딩 생성 태스크."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI
from prefect import task, get_run_logger


def _build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for embedding generation")

    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


@task(name="generate_openai_embeddings")
def generate_embeddings(
    records: List[Dict[str, Any]],
    title_field: str = "title",
    contents_field: str = "contents",
    embedding_field: str = "Embedding",
    model: str = "text-embedding-3-large",
) -> int:
    """주어진 레코드에 임베딩을 생성하고 적용."""

    logger = get_run_logger()

    if not records:
        logger.info("No records provided for embedding generation")
        return 0

    client = _build_client()
    generated = 0

    for record in records:
        title = record.get(title_field, "") or ""
        contents = record.get(contents_field, "") or ""
        text = f"{title}\n{contents}".strip()

        if not text:
            logger.warning("Skipping empty record during embedding generation")
            continue

        response = client.embeddings.create(model=model, input=text)
        record[embedding_field] = response.data[0].embedding
        generated += 1

    logger.info("Generated embeddings for %d records", generated)
    return generated
