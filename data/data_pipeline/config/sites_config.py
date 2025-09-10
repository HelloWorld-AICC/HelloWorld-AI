"""
크롤링 대상 사이트 설정

이 파일은 데이터 파이프라인에서 크롤링할 대상 사이트들의 설정을 관리합니다.
각 사이트별로 크롤러 클래스, URL, 크롤링 옵션 등을 설정할 수 있습니다.

사용법:
- get_enabled_sites(): 활성화된 사이트 목록 반환
- get_site_config(site_name): 특정 사이트 설정 반환
- update_site_config(site_name, **kwargs): 사이트 설정 업데이트
- add_new_site(name, base_url, crawler_class, **kwargs): 새 사이트 추가
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class SiteConfig:
    """
    개별 사이트 크롤링 설정

    Attributes:
        name: 사이트 식별자 (예: "hug_faq", "site_a")
        base_url: 크롤링할 사이트의 기본 URL
        crawler_class: 해당 사이트를 크롤링할 크롤러 클래스명
        enabled: 크롤링 활성화 여부 (True/False)
        crawl_interval_days: 크롤링 주기 (일 단위, 기본값: 30일 = 한달)
        max_pages: 최대 크롤링할 페이지 수 (무한 크롤링 방지)
        user_agent: HTTP 요청 시 사용할 User-Agent 문자열
        headers: HTTP 요청 헤더 (인증, 쿠키 등)
        delay_seconds: 페이지 간 요청 지연 시간 (서버 부하 방지)
        timeout_seconds: HTTP 요청 타임아웃 시간
        retry_count: 요청 실패 시 재시도 횟수
    """

    name: str
    base_url: str
    crawler_class: str
    enabled: bool = True
    crawl_interval_days: int = 30  # 한달에 한번 크롤링
    user_agent: str = None
    headers: Dict[str, str] = None
    delay_seconds: float = 1.0
    timeout_seconds: int = 30
    retry_count: int = 3


# 크롤링 대상 사이트들 설정
SITES_CONFIG = {
    "hug_faq": SiteConfig(
        name="hug_faq",
        base_url="http://hugkorea.or.kr/board/bbs_list.php?code=faq",
        crawler_class="HugFaqCrawler",
        enabled=True,
        crawl_interval_days=30,  # 한달에 한번 크롤링
        delay_seconds=1.0,  # 서버 부하 방지를 위한 1초 지연
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ),
    # 추후 추가될 사이트들 (현재는 비활성화)
    "site_a": SiteConfig(
        name="site_a",
        base_url="https://example-site-a.com",
        crawler_class="SiteACrawler",
        enabled=False,  # 아직 구현되지 않음
        crawl_interval_days=30,  # 한달에 한번 크롤링
    ),
    "site_b": SiteConfig(
        name="site_b",
        base_url="https://example-site-b.com",
        crawler_class="SiteBCrawler",
        enabled=False,  # 아직 구현되지 않음
        crawl_interval_days=30,  # 한달에 한번 크롤링
    ),
    # 법령 사이트들 (크롤러는 미구현 상태로 비활성화 등록)
    "law_industrial_accident": SiteConfig(
        name="law_industrial_accident",
        base_url="https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%82%B0%EC%97%85%EC%9E%AC%ED%95%B4%EB%B3%B4%EC%83%81%EB%B3%B4%ED%97%98%EB%B2%95",
        crawler_class="IndustrialAccidentActCrawler",
        enabled=True,
        crawl_interval_days=30,
    ),
    "law_labor_standards": SiteConfig(
        name="law_labor_standards",
        base_url="https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EA%B7%BC%EB%A1%9C%EA%B8%B0%EC%A4%80%EB%B2%95",
        crawler_class="LaborStandardsActCrawler",
        enabled=True,
        crawl_interval_days=30,
    ),
    "law_minimum_wage": SiteConfig(
        name="law_minimum_wage",
        base_url="https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%B5%9C%EC%A0%80%EC%9E%84%EA%B8%88%EB%B2%95",
        crawler_class="MinimumWageActCrawler",
        enabled=True,
        crawl_interval_days=30,
    ),
    "law_wage_claim_guarantee": SiteConfig(
        name="law_wage_claim_guarantee",
        base_url="https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%9E%84%EA%B8%88%EC%B1%84%EA%B6%8C%EB%B3%B4%EC%9E%A5%EB%B2%95",
        crawler_class="WageClaimGuaranteeActCrawler",
        enabled=True,
        crawl_interval_days=30,
    ),
}


def get_enabled_sites() -> List[str]:
    """
    활성화된 사이트 목록 반환

    Returns:
        활성화된 사이트 이름들의 리스트
    """
    return [name for name, config in SITES_CONFIG.items() if config.enabled]


def get_site_config(site_name: str) -> SiteConfig:
    """
    특정 사이트 설정 반환

    Args:
        site_name: 사이트 이름

    Returns:
        해당 사이트의 SiteConfig 객체

    Raises:
        ValueError: 존재하지 않는 사이트 이름인 경우
    """
    if site_name not in SITES_CONFIG:
        raise ValueError(f"Unknown site: {site_name}")
    return SITES_CONFIG[site_name]


def get_all_site_configs() -> Dict[str, SiteConfig]:
    """
    모든 사이트 설정 반환 (활성화 여부와 관계없이)

    Returns:
        모든 사이트 설정의 딕셔너리
    """
    return SITES_CONFIG.copy()


def update_site_config(site_name: str, **kwargs) -> None:
    """
    사이트 설정 업데이트

    Args:
        site_name: 업데이트할 사이트 이름
        **kwargs: 업데이트할 설정 키-값 쌍들

    Raises:
        ValueError: 존재하지 않는 사이트이거나 잘못된 설정 키인 경우

    Example:
        update_site_config("hug_faq", max_pages=200, delay_seconds=2.0)
    """
    if site_name not in SITES_CONFIG:
        raise ValueError(f"Unknown site: {site_name}")

    current_config = SITES_CONFIG[site_name]
    for key, value in kwargs.items():
        if hasattr(current_config, key):
            setattr(current_config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")


def add_new_site(name: str, base_url: str, crawler_class: str, **kwargs) -> None:
    """
    새로운 사이트 추가

    Args:
        name: 새 사이트 이름
        base_url: 사이트 기본 URL
        crawler_class: 크롤러 클래스명
        **kwargs: 추가 설정 옵션들

    Raises:
        ValueError: 이미 존재하는 사이트 이름인 경우

    Example:
        add_new_site(
            name="new_site",
            base_url="https://example.com",
            crawler_class="NewSiteCrawler",
            max_pages=50,
            delay_seconds=1.5
        )
    """
    if name in SITES_CONFIG:
        raise ValueError(f"Site {name} already exists")

    config = SiteConfig(
        name=name, base_url=base_url, crawler_class=crawler_class, **kwargs
    )

    SITES_CONFIG[name] = config


def get_sites_by_crawl_interval(days: int) -> List[str]:
    """
    특정 크롤링 주기를 가진 사이트들 반환

    Args:
        days: 크롤링 주기 (일 단위)

    Returns:
        해당 주기를 가진 사이트 이름들의 리스트

    Example:
        monthly_sites = get_sites_by_crawl_interval(30)  # 한달 주기 사이트들
    """
    return [
        name
        for name, config in SITES_CONFIG.items()
        if config.enabled and config.crawl_interval_days == days
    ]


def get_next_crawl_sites() -> List[str]:
    """
    다음 크롤링 대상 사이트들 반환 (활성화된 모든 사이트)

    Returns:
        활성화된 사이트 이름들의 리스트

    Note:
        실제 스케줄링 로직에서는 마지막 크롤링 시간과 비교하여
        주기가 도래한 사이트들만 반환하도록 구현할 수 있습니다.
    """
    return get_enabled_sites()
