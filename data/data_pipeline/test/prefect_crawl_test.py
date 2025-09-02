# filename: src/flow_hug_faq.py
from __future__ import annotations
import json, re, time, hashlib
from pathlib import Path
from typing import List, Dict, Union

import requests
from bs4 import BeautifulSoup, Tag
from prefect import flow, task, get_run_logger

# --------- 기존 코드(핵심) 재사용: clean_text / parse_faq / fetch ---------
TARGET_URL = "http://hugkorea.or.kr/board/bbs_list.php?code=faq"


def clean_text(html_fragment: Union[Tag, str]) -> str:
    """<br> -> \n, &nbsp; 제거, 공백 정리."""
    if isinstance(html_fragment, Tag):
        for br in html_fragment.find_all("br"):
            br.replace_with("\n")
        text = html_fragment.get_text(separator=" ", strip=True)
    else:
        text = str(html_fragment)
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = re.sub(r"^\s*A\.\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
    return text


def parse_faq(html: str, page_url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    faq_root = soup.select_one(".faq") or soup
    items = []
    for trig in faq_root.select(".trigger-button"):
        q_span_list = trig.find_all("span")
        if len(q_span_list) >= 2:
            question = clean_text(q_span_list[1])
        else:
            question = clean_text(trig)
        acc = trig.find_next_sibling(
            lambda tag: tag.name == "div" and "accordion" in tag.get("class", [])
        )
        if not acc:
            continue
        ans_div = acc.find("div") or acc
        answer = clean_text(ans_div)
        items.append({"title": question, "contents": answer, "url": page_url})
    return items


def fetch(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FAQCrawler/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


# ----------------- Prefect 태스크 -----------------
@task(retries=2, retry_delay_seconds=10)
def crawl(url: str) -> List[Dict]:
    html = fetch(url)
    data = parse_faq(html, url)
    return data


def _fingerprint(title: str, contents: str) -> str:
    return hashlib.sha256((title + contents).encode("utf-8")).hexdigest()


@task
def clean(records: List[Dict]) -> List[Dict]:
    # 간단 중복제거(지문 기반)
    seen = set()
    out = []
    for r in records:
        fp = _fingerprint(r.get("title", ""), r.get("contents", ""))
        if fp in seen:
            continue
        seen.add(fp)
        # 여기에 필요하면 추가 전처리(PII 마스킹, 공백 정리 등) 삽입
        out.append({**r, "fingerprint": fp, "fetched_at": int(time.time())})
    return out


@task
def save_json(records: List[Dict], output_path: str) -> str:
    path = Path(output_path)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())


# ----------------- Prefect 플로우 -----------------
@flow(name="hug-faq-crawl-weekly")
def main(url: str = TARGET_URL, output_path: str = "hug_faq.json") -> str:
    log = get_run_logger()
    raw = crawl(url)
    cleaned = clean(raw)
    out_path = save_json(cleaned, output_path)
    log.info(f"Saved {len(cleaned)} items -> {out_path}")
    return out_path


if __name__ == "__main__":
    # 로컬 단독 실행 테스트
    main()
