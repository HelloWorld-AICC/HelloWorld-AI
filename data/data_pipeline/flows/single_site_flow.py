"""
단일 사이트 크롤링을 위한 Prefect 플로우
"""

from typing import List, Dict, Any, Optional
from prefect import flow, get_run_logger
from datetime import datetime

from crawlers.hug_faq_crawler import HugFaqCrawler
from crawlers.industrial_accident_act_crawler import IndustrialAccidentActCrawler
from crawlers.labor_standards_act_crawler import LaborStandardsActCrawler
from crawlers.minimum_wage_act_crawler import MinimumWageActCrawler
from crawlers.wage_claim_guarantee_act_crawler import WageClaimGuaranteeActCrawler
from processors.text_cleaner import clean_text_batch
from processors.deduplicator import remove_duplicates_by_fingerprint
from storage.local_storage import save_multiple_formats
from storage.mongo_storage import delete_existing_by_urls as mongo_delete_by_urls
from storage.mongo_storage import insert_records as mongo_insert_records
from processors.embedding import generate_embeddings


@flow(name="single-site-crawl", description="단일 사이트 크롤링 및 데이터 처리")
def crawl_single_site(
    site_name: str = "hug_faq",
    output_dir: str = "crawled_data",
    save_formats: List[str] = None,
    enable_mongo: bool = False,
    mongo_uri: Optional[str] = None,
    mongo_db: str = "HelloWorld-AI",
    mongo_collection: Optional[str] = "foreigner_legalQA_v3",
    mongo_embedding_model: Optional[str] = "text-embedding-3-large",
    mongo_embedding_field: str = "Embedding",
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

    if mongo_collection is None:
        mongo_collection = site_name

    start_time = datetime.now()
    logger.info(f"Starting single site crawl for {site_name} at {start_time}")

    try:
        # 1. 크롤러 초기화 및 실행
        if site_name == "hug_faq":
            crawler = HugFaqCrawler()
        elif site_name == "law_industrial_accident":
            crawler = IndustrialAccidentActCrawler()
        elif site_name == "law_labor_standards":
            crawler = LaborStandardsActCrawler()
        elif site_name == "law_minimum_wage":
            crawler = MinimumWageActCrawler()
        elif site_name == "law_wage_claim_guarantee":
            crawler = WageClaimGuaranteeActCrawler()
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

        # 5. MongoDB 업서트 (옵션)
        mongo_result = None
        embedded_new_records = 0
        if enable_mongo:
            if not mongo_uri:
                raise ValueError("MongoDB URI is required when enable_mongo=True")
            target_urls = [
                record["url"] for record in unique_records if record.get("url")
            ]

            # 신규 레코드 임베딩
            if mongo_embedding_model and unique_records:
                embedded_new_records = generate_embeddings(
                    records=unique_records,
                    title_field="title",
                    contents_field="contents",
                    embedding_field=mongo_embedding_field,
                    model=mongo_embedding_model,
                )

            deleted_count = mongo_delete_by_urls(
                urls=target_urls,
                uri=mongo_uri,
                db_name=mongo_db,
                collection_name=mongo_collection,
            )

            inserted_count = mongo_insert_records(
                records=unique_records,
                uri=mongo_uri,
                db_name=mongo_db,
                collection_name=mongo_collection,
                embedding_field=mongo_embedding_field,
            )

            mongo_result = {
                "deleted": deleted_count,
                "inserted": inserted_count,
                "embedded_new_records": embedded_new_records,
            }

        # 6. 데이터 저장
        logger.info("Saving data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{output_dir}/{site_name}_{timestamp}"

        saved_paths = save_multiple_formats(unique_records, base_filename, save_formats)

        # 7. 결과 요약
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
            "mongo_result": mongo_result,
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
