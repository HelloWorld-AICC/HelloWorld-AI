"""MongoDB 저장 태스크 (교체 방식)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from prefect import task, get_run_logger


_CLIENT_CACHE: Dict[str, MongoClient] = {}


def _get_client(uri: str) -> MongoClient:
    client = _CLIENT_CACHE.get(uri)
    if client is not None:
        return client

    client = MongoClient(uri, appname="prefect-data-pipeline")
    _CLIENT_CACHE[uri] = client
    return client


@task(name="mongo_delete_by_urls")
def delete_existing_by_urls(
    urls: List[str], uri: str, db_name: str, collection_name: str
) -> int:
    """URL 목록과 일치하는 문서를 일괄 삭제."""

    logger = get_run_logger()
    urls = [url for url in urls if url]
    if not urls:
        logger.info("No URLs provided for deletion")
        return 0

    client = _get_client(uri)
    collection = client[db_name][collection_name]

    try:
        result = collection.delete_many({"url": {"$in": urls}})
        logger.info("Deleted %d existing MongoDB documents", result.deleted_count)
        return result.deleted_count
    except PyMongoError as exc:
        logger.error(f"Failed to delete existing records: {exc}")
        raise


def _to_db_document(record: Dict[str, Any], embedding_field: str) -> Dict[str, Any]:
    return {
        "title": record.get("title"),
        "contents": record.get("contents"),
        "url": record.get("url"),
        "Embedding": record.get(embedding_field) or [],
    }


@task(name="mongo_insert_records")
def insert_records(
    records: List[Dict[str, Any]],
    uri: str,
    db_name: str,
    collection_name: str,
    embedding_field: str = "Embedding",
) -> int:
    """MongoDB에 문서들을 삽입."""

    logger = get_run_logger()

    if not records:
        logger.info("No records provided for MongoDB insert")
        return 0

    client = _get_client(uri)
    collection = client[db_name][collection_name]

    documents = [_to_db_document(record, embedding_field) for record in records]

    try:
        result = collection.insert_many(documents, ordered=False)
        inserted = len(result.inserted_ids)
        logger.info("Inserted %d documents into MongoDB", inserted)
        return inserted
    except PyMongoError as exc:
        logger.error(f"Failed to insert records: {exc}")
        raise
