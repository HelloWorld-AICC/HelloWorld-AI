"""
단일 사이트 크롤링을 위한 Prefect 플로우
"""

from typing import List, Dict, Any
from prefect import flow, get_run_logger
from datetime import datetime

from ..crawlers.hug_faq_crawler import HugFaqCrawler
from ..processors.text_cleaner import clean_text_batch
from ..processors.deduplicator import (
    remove_duplicates_by_fingerprint,
    generate_fingerprints,
)
from ..storage.local_storage import save_multiple_formats


@flow(name="single-site-crawl", description="단일 사이트 크롤링 및 데이터 처리")
def crawl_single_site(
    site_name: str = "hug_faq",
    output_dir: str = "crawled_data",
    save_formats: List[str] = None,
) -> Dict[str, Any]:
    """
    단일 사이트 크롤링 실행

    Args:
        site_name: 크롤링할 사이트 이름
        output_dir: 결과 저장 디렉토리
        save_formats: 저장할 파일 형식들

    Returns:
        크롤링 결과 및 저장 경로 정보
    """
    logger = get_run_logger()

    if save_formats is None:
        save_formats = ["json", "csv"]

    start_time = datetime.now()
    logger.info(f"Starting single site crawl for {site_name} at {start_time}")

    try:
        # 1. 크롤러 초기화 및 실행
        if site_name == "hug_faq":
            crawler = HugFaqCrawler()
        else:
            raise ValueError(f"Unsupported site: {site_name}")

        # 2. 크롤링 실행
        logger.info("Starting crawling...")
        raw_data = crawler.crawl()
        logger.info(f"Crawled {len(raw_data)} raw items")

        if not raw_data:
            logger.warning("No data crawled")
            return {
                "site_name": site_name,
                "crawled_count": 0,
                "processed_count": 0,
                "saved_paths": {},
                "status": "completed_no_data",
            }

        # 3. 데이터 전처리
        logger.info("Processing data...")

        # 텍스트 정리
        titles = [item.title for item in raw_data]
        contents = [item.contents for item in raw_data]

        cleaned_titles = clean_text_batch(titles)
        cleaned_contents = clean_text_batch(contents)

        # 전처리된 데이터로 레코드 업데이트
        processed_records = []
        for i, item in enumerate(raw_data):
            record = {
                "title": cleaned_titles[i],
                "contents": cleaned_contents[i],
                "url": item.url,
                "source_site": item.source_site,
                "crawled_at": item.crawled_at,
                "fingerprint": item.fingerprint,
                "metadata": item.metadata,
            }
            processed_records.append(record)

        # 4. 중복 제거
        logger.info("Removing duplicates...")
        unique_records = remove_duplicates_by_fingerprint(processed_records)
        logger.info(f"After deduplication: {len(unique_records)} unique records")

        # 5. 데이터 저장
        logger.info("Saving data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{output_dir}/{site_name}_{timestamp}"

        saved_paths = save_multiple_formats(unique_records, base_filename, save_formats)

        # 6. 결과 요약
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = {
            "site_name": site_name,
            "crawled_count": len(raw_data),
            "processed_count": len(unique_records),
            "duplicates_removed": len(raw_data) - len(unique_records),
            "saved_paths": saved_paths,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "status": "completed_success",
        }

        logger.info(f"Crawl completed successfully in {duration:.2f} seconds")
        logger.info(
            f"Results: {len(raw_data)} crawled -> {len(unique_records)} processed"
        )

        return result

    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        return {"site_name": site_name, "error": str(e), "status": "failed"}


if __name__ == "__main__":
    # 로컬 테스트 실행
    result = crawl_single_site()
    print(f"Crawl result: {result}")
