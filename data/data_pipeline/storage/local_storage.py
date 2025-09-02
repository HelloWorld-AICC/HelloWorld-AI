"""
로컬 파일 저장 모듈
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from prefect import task, get_run_logger
import pandas as pd


@task(name="save_json")
def save_to_json(
    records: List[Dict[str, Any]],
    output_path: str,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> str:
    """JSON 파일로 저장"""
    logger = get_run_logger()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=ensure_ascii, indent=indent)

        logger.info(f"Saved {len(records)} records to {path.resolve()}")
        return str(path.resolve())

    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise


@task(name="save_csv")
def save_to_csv(records: List[Dict[str, Any]], output_path: str) -> str:
    """CSV 파일로 저장"""
    logger = get_run_logger()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not records:
            logger.warning("No records to save")
            return str(path.resolve())

        # DataFrame으로 변환하여 저장
        df = pd.DataFrame(records)
        df.to_csv(path, index=False, encoding="utf-8-sig")

        logger.info(f"Saved {len(records)} records to {path.resolve()}")
        return str(path.resolve())

    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise


@task(name="save_jsonl")
def save_to_jsonl(records: List[Dict[str, Any]], output_path: str) -> str:
    """JSONL 파일로 저장 (한 줄에 하나의 JSON)"""
    logger = get_run_logger()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                json_line = json.dumps(record, ensure_ascii=False)
                f.write(json_line + "\n")

        logger.info(f"Saved {len(records)} records to {path.resolve()}")
        return str(path.resolve())

    except Exception as e:
        logger.error(f"Failed to save JSONL: {e}")
        raise


@task(name="save_multiple_formats")
def save_multiple_formats(
    records: List[Dict[str, Any]], base_path: str, formats: List[str] = None
) -> Dict[str, str]:
    """여러 형식으로 동시 저장"""
    logger = get_run_logger()

    if formats is None:
        formats = ["json", "csv", "jsonl"]

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for fmt in formats:
        if fmt == "json":
            output_path = base_path.with_suffix(".json")
            saved_paths["json"] = save_to_json(records, str(output_path))
        elif fmt == "csv":
            output_path = base_path.with_suffix(".csv")
            saved_paths["csv"] = save_to_csv(records, str(output_path))
        elif fmt == "jsonl":
            output_path = base_path.with_suffix(".jsonl")
            saved_paths["jsonl"] = save_to_jsonl(records, str(output_path))

    logger.info(f"Saved data in {len(saved_paths)} formats")
    return saved_paths
