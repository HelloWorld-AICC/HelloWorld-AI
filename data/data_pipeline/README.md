# 🚀 Prefect 기반 데이터 자동 크롤링 시스템

이 프로젝트는 Prefect를 활용하여 다양한 웹사이트에서 데이터를 자동으로 수집하고 처리하는 시스템입니다.

## 🏗️ 프로젝트 구조

```
data_pipeline/
├── crawlers/           # 크롤러별 모듈
│   ├── base_crawler.py      # 기본 크롤러 클래스
│   └── hug_faq_crawler.py   # HUG FAQ 전용 크롤러
├── flows/              # Prefect 플로우 정의
│   └── single_site_flow.py  # 단일 사이트 크롤링 플로우
├── processors/         # 데이터 처리 모듈
│   ├── text_cleaner.py      # 텍스트 정리
│   └── deduplicator.py      # 중복 제거
├── storage/            # 데이터 저장 모듈
│   ├── local_storage.py     # 로컬 파일 저장
│   └── mongo_storage.py     # MongoDB 업서트 태스크
├── config/             # 설정 파일들
│   └── sites_config.py      # 크롤링 대상 사이트 설정
├── scripts/            # 실행 스크립트
│   └── run_single_crawl.py  # 단일 크롤링 실행
└── tests/              # 테스트 코드
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
cd HelloWorld-AI/data/data_pipeline
pip install -r requirements.txt
```

### 2. 단일 사이트 크롤링 실행

```bash
# 대화형 모드 (사이트 선택 프롬프트)
python scripts/run_single_crawl.py

# 바로 HUG FAQ 크롤링
python scripts/run_single_crawl.py --site hug_faq

# 산업재해보상보험법 등 다른 사이트 예시
python scripts/run_single_crawl.py --site law_industrial_accident

# MongoDB까지 적재하고 싶을 때 (HelloWorld-AI/foreigner_legalQA_v3)
python scripts/run_single_crawl.py --site hug_faq --enable-mongo \
  --mongo-uri "<your-mongodb-uri>" --mongo-db "HelloWorld-AI" \
  --mongo-collection "foreigner_legalQA_v3"
```

> 환경변수 `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`을 미리 설정하면 CLI 인자를 생략할 수 있습니다.

### 3. Prefect UI에서 모니터링

```bash
# Prefect 서버 시작
prefect server start

# 브라우저에서 http://localhost:4200 접속
```

## 📋 현재 지원하는 사이트

- **hug_faq**: HUG 한국 FAQ (활성화됨)
- **site_a**: 예시 사이트 A (비활성화)
- **site_b**: 예시 사이트 B (비활성화)

## 🔧 새로운 사이트 추가하기

### 1. 크롤러 클래스 생성

```python
# crawlers/my_site_crawler.py
from .base_crawler import BaseCrawler, CrawlResult

class MySiteCrawler(BaseCrawler):
    def __init__(self):
        super().__init__(
            site_name="my_site",
            base_url="https://example.com"
        )

    def parse_content(self, html: str, page_url: str) -> List[CrawlResult]:
        # HTML 파싱 로직 구현
        pass

    def get_crawl_urls(self) -> List[str]:
        # 크롤링할 URL 목록 반환
        return [self.base_url]
```

### 2. 설정에 사이트 추가

```python
# config/sites_config.py
from crawlers.my_site_crawler import MySiteCrawler

SITES_CONFIG["my_site"] = SiteConfig(
    name="my_site",
    base_url="https://example.com",
    crawler_class="MySiteCrawler",
    enabled=True
)
```

### 3. 플로우에 크롤러 추가

```python
# flows/single_site_flow.py
from ..crawlers.my_site_crawler import MySiteCrawler

# crawl_single_site 함수 내부에 추가
if site_name == "my_site":
    crawler = MySiteCrawler()
```

## 📊 데이터 처리 파이프라인

1. **크롤링**: 웹사이트에서 원시 데이터 수집
2. **텍스트 정리**: HTML 태그 제거, 공백 정리
3. **중복 제거**: 지문 기반 중복 항목 제거
4. **데이터 저장**: JSON/CSV 로컬 저장 + 선택적 MongoDB 업서트

### MongoDB 적재 옵션

- `--enable-mongo` 플래그를 주면 MongoDB에 **같은 URL 문서를 모두 삭제한 뒤 새 데이터로 교체**
- 기본 저장 위치는 **Database: `HelloWorld-AI` / Collection: `foreigner_legalQA_v3`**
- MongoDB에는 기존 스키마(`title`, `contents`, `url`, `Embedding`)만 저장되며, `Embedding` 값이 없으면 빈 리스트가 들어갑니다.
- OpenAI API를 통해 각 문서에 임베딩을 생성 후 `Embedding` 필드로 저장 (기본 모델: `text-embedding-3-large`)
- 법령 사이트(law.go.kr)는 간헐적으로 접속이 제한될 수 있으므로, 404 등 오류가 발생하면 복구 후 재실행하세요.

## 🔑 환경 변수

- `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`: MongoDB 연결 정보 (기본: `HelloWorld-AI` / `foreigner_legalQA_v3`)
- `MONGODB_EMBEDDING_MODEL`, `MONGODB_EMBEDDING_FIELD`: 임베딩 관련 기본값
- `OPENAI_API_KEY` (필수), `OPENAI_API_BASE`(선택): 신규 문서 임베딩 생성을 위한 OpenAI 인증

## 🔍 모니터링 및 로깅

- Prefect UI를 통한 실시간 모니터링
- 구조화된 로깅 (크롤링 진행상황, 에러 등)
- 성능 메트릭 (처리 시간, 항목 수 등)

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/

# 특정 모듈 테스트
pytest tests/test_crawlers/
pytest tests/test_flows/
```

## 📝 설정 옵션

`sites_config.py`에서 각 사이트별로 다음 옵션을 설정할 수 있습니다:

- `crawl_interval_days`: 크롤링 간격 (일 단위, 기본값: 30일 = 한달)
- `max_pages`: 최대 크롤링 페이지 수 (무한 크롤링 방지)
- `delay_seconds`: 요청 간 지연 시간 (서버 부하 방지)
- `timeout_seconds`: HTTP 요청 타임아웃
- `retry_count`: 실패 시 재시도 횟수
- `user_agent`: HTTP 요청 시 사용할 User-Agent 문자열
- `headers`: HTTP 요청 헤더 (인증, 쿠키 등)

### 크롤링 주기 설정

현재 시스템은 **한달에 한번** 크롤링하도록 설계되어 있습니다:
- `crawl_interval_days=30`: 30일마다 크롤링 실행
- 필요에 따라 주간(7일), 월간(30일), 분기별(90일) 등으로 조정 가능
