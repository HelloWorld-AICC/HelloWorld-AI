"""
HUG FAQ 크롤러
기존 prefect_crawl_test.py의 로직을 활용하여 구현
"""

import re
from typing import List
from bs4 import BeautifulSoup, Tag
from .base_crawler import BaseCrawler, CrawlResult


class HugFaqCrawler(BaseCrawler):
    """HUG FAQ 전용 크롤러"""

    def __init__(self):
        super().__init__(
            site_name="hug_faq",
            base_url="http://hugkorea.or.kr/board/bbs_list.php?code=faq",
        )

    def get_crawl_urls(self) -> List[str]:
        """크롤링할 URL 목록"""
        return [self.base_url]

    def parse_content(self, html: str, page_url: str) -> List[CrawlResult]:
        """FAQ HTML 파싱"""
        soup = BeautifulSoup(html, "html.parser")
        faq_root = soup.select_one(".faq") or soup
        items = []

        for idx, trig in enumerate(faq_root.select(".trigger-button")):
            # 질문 추출
            q_span_list = trig.find_all("span")
            if len(q_span_list) >= 2:
                question = self.clean_text(q_span_list[1])
            else:
                question = self.clean_text(trig)

            # 답변 추출
            acc = trig.find_next_sibling(
                lambda tag: tag.name == "div" and "accordion" in tag.get("class", [])
            )
            if not acc:
                continue

            ans_div = acc.find("div") or acc
            answer = self.clean_text(ans_div)

            # CrawlResult 생성
            result = CrawlResult(
                title=question,
                contents=answer,
                url=page_url,
                source_site=self.site_name,
                crawled_at=0,  # crawl() 메서드에서 설정됨
                metadata={
                    "faq_type": "general",
                    "parsed_from": "accordion",
                    "fragment_id": trig.get("id") or acc.get("id") or f"faq-{idx}",
                    "position": idx,
                },
            )

            items.append(result)

        return items

    def clean_text(self, html_fragment) -> str:
        """HUG FAQ 전용 텍스트 정리"""
        if isinstance(html_fragment, Tag):
            # <br> 태그를 줄바꿈으로 변환
            for br in html_fragment.find_all("br"):
                br.replace_with("\n")
            text = html_fragment.get_text(separator=" ", strip=True)
        else:
            text = str(html_fragment)

        # HTML 엔티티 제거
        text = text.replace("\xa0", " ").replace("&nbsp;", " ")

        # 정규식으로 텍스트 정리
        text = re.sub(r"^\s*A\.\s*", "", text)  # "A." 제거
        text = re.sub(r"[ \t]+", " ", text)  # 연속된 공백을 하나로
        text = re.sub(r"\n\s*\n\s*", "\n\n", text)  # 연속된 줄바꿈 정리

        return text.strip()
