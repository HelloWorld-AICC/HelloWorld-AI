"""
중복 제거 모듈
"""

from typing import List, Dict, Any, Set
from prefect import task, get_run_logger
import hashlib


@task(name="remove_duplicates")
def remove_duplicates_by_fingerprint(
    records: List[Dict[str, Any]], fingerprint_field: str = "fingerprint"
) -> List[Dict[str, Any]]:
    """지문 기반 중복 제거"""
    logger = get_run_logger()

    seen_fingerprints: Set[str] = set()
    unique_records = []

    for record in records:
        fingerprint = record.get(fingerprint_field)
        if not fingerprint:
            logger.warning(
                f"Record missing fingerprint: {record.get('title', 'Unknown')[:50]}"
            )
            continue

        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_records.append(record)

    removed_count = len(records) - len(unique_records)
    logger.info(
        f"Removed {removed_count} duplicate records. Kept {len(unique_records)} unique records."
    )

    return unique_records


@task(name="remove_similar_content")
def remove_similar_content(
    records: List[Dict[str, Any]], similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """내용 유사도 기반 중복 제거 (간단한 구현)"""
    logger = get_run_logger()

    # 간단한 유사도 계산 (더 정교한 구현은 필요시 추가)
    def calculate_similarity(text1: str, text2: str) -> float:
        """간단한 Jaccard 유사도"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    unique_records = []

    for i, record in enumerate(records):
        is_duplicate = False
        current_content = record.get("contents", "")

        # 이미 추가된 레코드와 비교
        for existing_record in unique_records:
            existing_content = existing_record.get("contents", "")

            similarity = calculate_similarity(current_content, existing_content)
            if similarity > similarity_threshold:
                is_duplicate = True
                logger.debug(f"Found similar content (similarity: {similarity:.2f})")
                break

        if not is_duplicate:
            unique_records.append(record)

    removed_count = len(records) - len(unique_records)
    logger.info(
        f"Removed {removed_count} similar records. Kept {len(unique_records)} unique records."
    )

    return unique_records


@task(name="generate_fingerprints")
def generate_fingerprints(
    records: List[Dict[str, Any]], fields: List[str] = None
) -> List[Dict[str, Any]]:
    """레코드에 지문 생성"""
    logger = get_run_logger()

    if fields is None:
        fields = ["title", "contents"]

    for record in records:
        # 지정된 필드들을 결합하여 지문 생성
        content_parts = []
        for field in fields:
            if field in record and record[field]:
                content_parts.append(str(record[field]))

        if content_parts:
            combined_content = "".join(content_parts)
            fingerprint = hashlib.sha256(combined_content.encode("utf-8")).hexdigest()
            record["fingerprint"] = fingerprint
        else:
            logger.warning(f"Record missing required fields for fingerprint: {record}")
            record["fingerprint"] = None

    logger.info(f"Generated fingerprints for {len(records)} records")
    return records
