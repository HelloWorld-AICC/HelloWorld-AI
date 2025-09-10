"""
산업재해보상보험법 페이지 전용 크롤러

법령 사이트(law.go.kr)는 페이지 구조가 자주 변동될 수 있으므로, 본 크롤러는
여러 후보 셀렉터를 사용해 제목/본문을 추출하는 방어적 파싱을 수행합니다.
"""

from typing import List
from bs4 import BeautifulSoup, Tag

from crawlers.base_crawler import BaseCrawler, CrawlResult


class IndustrialAccidentActCrawler(BaseCrawler):
    """산업재해보상보험법 전용 크롤러"""

    def __init__(self):
        super().__init__(
            site_name="law_industrial_accident",
            base_url="https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%82%B0%EC%97%85%EC%9E%AC%ED%95%B4%EB%B3%B4%EC%83%81%EB%B3%B4%ED%97%98%EB%B2%95",
        )

    def get_crawl_urls(self) -> List[str]:
        """크롤링할 URL 목록 반환"""
        return [self.base_url]

    def fetch_html(self, url: str, retries: int = 3, delay: float = 1.0) -> str:
        """iframe을 통한 실제 법령 내용 접근"""
        from bs4 import BeautifulSoup

        html = super().fetch_html(url, retries=retries, delay=delay)
        self.logger.info(f"Initial HTML length: {len(html)}")

        # BeautifulSoup으로 파싱
        soup = BeautifulSoup(html, "html.parser")

        # iframe에서 실제 콘텐츠 URL 찾기
        iframe = soup.find("iframe", id="lawService")
        if iframe and iframe.get("src"):
            iframe_url = iframe["src"]
            self.logger.info(f"Found iframe URL: {iframe_url}")

            # 상대 URL을 절대 URL로 변환
            if iframe_url.startswith("/"):
                base_domain = "https://www.law.go.kr"
                iframe_url = base_domain + iframe_url

            # iframe 내부 콘텐츠 크롤링
            iframe_html = super().fetch_html(iframe_url, retries=retries, delay=delay)
            self.logger.info(f"Iframe HTML length: {len(iframe_html)}")

            # iframe HTML이 JavaScript 렌더링 필요한지 확인
            if "lawcon" in iframe_html or "제1조" in iframe_html:
                self.logger.info("Found law content in iframe!")
                return iframe_html
            else:
                self.logger.warning(
                    "No law content in static iframe, trying JavaScript rendering"
                )
                # JavaScript 렌더링 시도
                rendered_html = self._render_with_playwright(iframe_url)
                if rendered_html:
                    return rendered_html
                return iframe_html

        self.logger.warning("No iframe found, using original HTML")
        return html

    def _render_with_playwright(self, url: str) -> str:
        """Playwright를 이용한 JavaScript 렌더링"""
        try:
            from playwright.sync_api import sync_playwright

            self.logger.info("Starting Playwright rendering...")
            with sync_playwright() as p:
                # Chromium 브라우저 시작
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-software-rasterizer",
                    ],
                )

                page = browser.new_page()

                # 페이지 로드 대기 시간 증가
                page.set_default_timeout(30000)

                # 페이지 로드
                self.logger.info(f"Loading page: {url}")
                page.goto(url, wait_until="domcontentloaded")

                # JavaScript 실행 대기 (더 긴 시간)
                page.wait_for_timeout(5000)

                # 법령 내용이 로드될 때까지 대기
                try:
                    # .lawcon 클래스나 조문이 나타날 때까지 대기
                    page.wait_for_selector(".lawcon, .pgroup", timeout=20000)
                    self.logger.info("Found law content elements!")
                except:
                    self.logger.warning(
                        "Law content elements not found, proceeding anyway"
                    )

                # 최종 HTML 가져오기
                html = page.content()
                browser.close()

                self.logger.info(f"Playwright rendered HTML length: {len(html)}")

                # 조문 내용 확인
                if "lawcon" in html or "제1조" in html:
                    self.logger.info(
                        "Successfully rendered law content with Playwright!"
                    )
                    return html
                else:
                    self.logger.warning(
                        "No law content found even after Playwright rendering"
                    )
                    return html

        except ImportError:
            self.logger.error("Playwright not available")
            return ""
        except Exception as e:
            self.logger.error(f"Playwright rendering failed: {e}")
            return ""

    def _fetch_with_selenium(self, url: str) -> str:
        """JavaScript 렌더링 폴백"""
        # Playwright를 먼저 시도 (WSL 환경에서 더 안정적)
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=30000)
                # 핵심 컨테이너 로딩 대기
                page.wait_for_selector("#conScroll", timeout=15000)
                html = page.content()
                browser.close()
                return html
        except ImportError:
            self.logger.warning("playwright not available, falling back to Selenium")
        except Exception as e:
            self.logger.warning(f"playwright failed ({e}), falling back to Selenium")

        # Selenium 폴백 - 단순화된 headless Chrome
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(options=options)
            try:
                driver.get(url)
                # 핵심 컨테이너 로딩 대기
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#conScroll"))
                )
                page = driver.page_source
                return page
            finally:
                driver.quit()
        except Exception as e:
            self.logger.error(f"Selenium fallback failed: {e}")
            # 마지막 폴백: 오류 전파
            raise

    def parse_content(self, html: str, page_url: str) -> List[CrawlResult]:
        """법령 페이지의 조문을 추출하여 단일 결과로 반환"""
        soup = BeautifulSoup(html, "html.parser")

        # 제목 추출 - 장 제목이나 법령명
        title_text = ""

        # 1) 장 제목 찾기 (.gtit 클래스)
        gtit = soup.select_one(".gtit")
        if gtit and gtit.get_text(strip=True):
            title_text = self.clean_text(gtit.get_text())

        # 2) 폴백 제목
        if not title_text:
            title_text = "산업재해보상보험법"

        # 3) 각 조문을 개별 CrawlResult로 생성 (장 정보 포함)
        results = []

        # pgroup 구조에서 장 제목과 조문의 관계 파악
        pgroups = soup.find_all("div", class_="pgroup")
        self.logger.info(f"Found {len(pgroups)} pgroup divs")

        current_chapter = ""  # 현재 장 제목 추적

        for pgroup in pgroups:
            # 장 제목 확인 (.gtit 클래스)
            chapter_title = pgroup.find("p", class_="gtit")
            if chapter_title:
                chapter_text = self.clean_text(chapter_title.get_text())
                # 개정 정보 제거 (예: "<개정 2007. 12. 27.>" 부분)
                chapter_text = chapter_text.split("<")[0].strip()
                if chapter_text:
                    current_chapter = chapter_text
                    self.logger.info(f"Found chapter: {current_chapter}")
                continue

            # 조문 내용 확인 (.lawcon 클래스)
            lawcon = pgroup.find("div", class_="lawcon")
            if lawcon:
                # 각 조문의 제목과 내용 추출
                article_paragraphs = lawcon.find_all("p")

                for p in article_paragraphs:
                    # 조문 제목이 있는 p 태그 찾기 (.bl 클래스의 label)
                    label = p.find("span", class_="bl")
                    if label:
                        # 조문 제목 추출 (예: "제1조(목적)")
                        article_title = self.clean_text(label.get_text())

                        # 장 정보와 조문 제목 결합
                        if current_chapter:
                            full_title = f"{current_chapter} {article_title}"
                        else:
                            full_title = article_title

                        # 전체 텍스트에서 제목 부분 제거하여 내용만 추출
                        full_text = p.get_text()
                        label_text = label.get_text()
                        article_content = full_text.replace(label_text, "", 1).strip()
                        article_content = self.clean_text(article_content)

                        if article_title and article_content:
                            results.append(
                                CrawlResult(
                                    title=full_title,
                                    contents=article_content,
                                    url=page_url,
                                    source_site=self.site_name,
                                    crawled_at=0,
                                    metadata={
                                        "source": "law.go.kr",
                                        "category": "statute",
                                        "article_type": "law_article",
                                        "chapter": current_chapter,
                                        "article": article_title,
                                    },
                                )
                            )

        if results:
            self.logger.info(f"Extracted {len(results)} individual articles")
        else:
            # 폴백: 전체 텍스트를 하나의 결과로 반환
            self.logger.warning("No articles found, using full text as single result")
            contents = self.clean_text(soup.get_text())
            results = [
                CrawlResult(
                    title=title_text,
                    contents=contents,
                    url=page_url,
                    source_site=self.site_name,
                    crawled_at=0,
                    metadata={
                        "source": "law.go.kr",
                        "category": "statute",
                        "article_type": "full_text",
                    },
                )
            ]

        return results
