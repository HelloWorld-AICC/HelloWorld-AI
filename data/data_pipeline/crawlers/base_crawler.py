"""
기본 크롤러 클래스
모든 사이트별 크롤러가 상속받아야 하는 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import hashlib
from dataclasses import dataclass
from prefect import task, get_run_logger


@dataclass
class CrawlResult:
    """크롤링 결과를 담는 데이터 클래스"""

    title: str
    contents: str
    url: str
    source_site: str
    crawled_at: int
    fingerprint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseCrawler(ABC):
    """모든 크롤러의 기본 클래스"""

    def __init__(self, site_name: str, base_url: str):
        self.site_name = site_name
        self.base_url = base_url
        self.logger = get_run_logger()

    @abstractmethod
    def parse_content(self, html: str, page_url: str) -> List[CrawlResult]:
        """HTML을 파싱하여 구조화된 데이터로 변환"""
        pass

    @abstractmethod
    def get_crawl_urls(self) -> List[str]:
        """크롤링할 URL 목록 반환"""
        pass

    def fetch_html(self, url: str, retries: int = 3, delay: float = 1.0) -> str:
        """웹페이지 HTML 가져오기"""
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; DataCrawler/1.0; +https://github.com/your-repo)"
        }

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or "utf-8"
                return response.text
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (2**attempt))  # 지수 백오프
                else:
                    raise

        raise Exception(f"Failed to fetch {url} after {retries} attempts")

    def clean_text(self, text: str) -> str:
        """텍스트 정리 (공백, 특수문자 등)"""
        import re

        # HTML 엔티티 제거
        text = text.replace("&nbsp;", " ").replace("\xa0", " ")
        # 연속된 공백 제거
        text = re.sub(r"\s+", " ", text)
        # 앞뒤 공백 제거
        return text.strip()

    def generate_fingerprint(self, title: str, contents: str) -> str:
        """중복 제거를 위한 지문 생성"""
        content = f"{title}{contents}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @task(name="crawl_site")
    def crawl(self) -> List[CrawlResult]:
        """사이트 크롤링 실행"""
        self.logger.info(f"Starting crawl for {self.site_name}")

        urls = self.get_crawl_urls()
        all_results = []

        for url in urls:
            try:
                self.logger.info(f"Crawling {url}")
                html = self.fetch_html(url)
                results = self.parse_content(html, url)

                # 지문 생성 및 메타데이터 추가
                for result in results:
                    result.fingerprint = self.generate_fingerprint(
                        result.title, result.contents
                    )
                    result.source_site = self.site_name
                    result.crawled_at = int(time.time())

                all_results.extend(results)
                self.logger.info(f"Found {len(results)} items from {url}")

                # 크롤링 간격 조절
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error crawling {url}: {e}")
                continue

        self.logger.info(
            f"Completed crawl for {self.site_name}. Total: {len(all_results)} items"
        )
        return all_results
