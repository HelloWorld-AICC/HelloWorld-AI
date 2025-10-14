#!/usr/bin/env python3
"""Prefect 단일 사이트 크롤링 실행 스크립트."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(env_path: Path) -> None:
    """간단한 .env 로더 (기존 환경변수는 유지)."""

    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()

        os.environ.setdefault(key, value)


load_env_file(PROJECT_ROOT / ".env")

from config.sites_config import get_enabled_sites, get_site_config
from flows.single_site_flow import crawl_single_site


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """명령행 인자 파싱."""

    parser = argparse.ArgumentParser(description="Prefect 기반 단일 사이트 크롤링 실행")
    parser.add_argument("--site", help="크롤링할 사이트 식별자")
    parser.add_argument(
        "--enable-mongo",
        action="store_true",
        help="MongoDB 업서트를 활성화",
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI"),
        help="MongoDB 접속 URI (기본: 환경변수 MONGODB_URI)",
    )
    parser.add_argument(
        "--mongo-db",
        default=os.getenv("MONGODB_DB", "HelloWorld-AI"),
        help="MongoDB 데이터베이스 이름",
    )
    parser.add_argument(
        "--mongo-collection",
        default=os.getenv("MONGODB_COLLECTION", "foreigner_legalQA_v3"),
        help="MongoDB 컬렉션 이름",
    )
    parser.add_argument(
        "--mongo-conflict-strategy",
        default=os.getenv("MONGODB_CONFLICT_STRATEGY"),
        help="[Deprecated] 구 방식 옵션 (무시됨)",
    )
    parser.add_argument(
        "--mongo-embedding-model",
        default=os.getenv("MONGODB_EMBEDDING_MODEL", "text-embedding-3-large"),
        help="신규 문서 임베딩에 사용할 OpenAI 모델 (비활성화하려면 빈 문자열)",
    )
    parser.add_argument(
        "--mongo-embedding-field",
        default=os.getenv("MONGODB_EMBEDDING_FIELD", "Embedding"),
        help="임베딩이 저장될 필드명",
    )
    parser.add_argument(
        "--mongo-upsert-key",
        default=os.getenv("MONGODB_UPSERT_KEY"),
        help="[Deprecated] 구 방식 옵션 (무시됨)",
    )

    return parser.parse_args(argv)


def select_site_interactively(enabled_sites: List[str]) -> Optional[str]:
    """사용자 인터랙션으로 사이트 선택."""

    print("\n크롤링할 사이트를 선택하세요:")
    for idx, site in enumerate(enabled_sites, 1):
        config = get_site_config(site)
        print(f"{idx}. {site} ({config.base_url})")

    try:
        choice = int(input("\n선택 (번호): ")) - 1
    except ValueError:
        print("❌ 숫자를 입력해주세요.")
        return None

    if 0 <= choice < len(enabled_sites):
        return enabled_sites[choice]

    print("❌ 잘못된 선택입니다.")
    return None


def main(argv: Optional[List[str]] = None) -> int:
    """메인 실행 함수."""

    args = parse_args(argv)

    if args.enable_mongo and not args.mongo_uri:
        print(
            "❌ enable-mongo 플래그를 사용할 때는 MongoDB URI가 필요합니다.\n"
            "   --mongo-uri 또는 환경변수 MONGODB_URI를 설정해주세요."
        )
        return 1

    print("🚀 Prefect 기반 데이터 크롤링 시스템")
    print("=" * 50)

    enabled_sites = get_enabled_sites()
    if not enabled_sites:
        print("❌ 활성화된 사이트가 없습니다.")
        return 1

    print(f"활성화된 사이트: {', '.join(enabled_sites)}")

    site_name = args.site
    if not site_name:
        site_name = select_site_interactively(enabled_sites)
        if not site_name:
            return 1

    try:
        site_config = get_site_config(site_name)
        print(f"\n📋 선택된 사이트: {site_name}")
        print(f"   URL: {site_config.base_url}")
        print(f"   크롤러: {site_config.crawler_class}")
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    output_dir = "crawled_data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 결과 저장 디렉토리: {output_dir}")

    if args.enable_mongo:
        target_collection = args.mongo_collection or site_name
        print("🗄️  MongoDB 업서트 활성화")
        print(f"   URI: {args.mongo_uri or '<<undefined>>'}")
        print(f"   DB: {args.mongo_db}")
        print(f"   Collection: {target_collection}")
        if args.mongo_embedding_model:
            print(f"   Embedding model: {args.mongo_embedding_model}")
        else:
            print("   Embedding model: disabled")

    print("\n🔄 크롤링 시작...")

    try:
        result = crawl_single_site(
            site_name=site_name,
            output_dir=output_dir,
            save_formats=["json", "csv"],
            enable_mongo=args.enable_mongo,
            mongo_uri=args.mongo_uri,
            mongo_db=args.mongo_db,
            mongo_collection=args.mongo_collection,
            mongo_embedding_model=args.mongo_embedding_model or None,
            mongo_embedding_field=args.mongo_embedding_field,
        )
    except Exception as exc:  # noqa: BLE001 - 사용자 피드백을 위한 광범위 예외 처리
        print(f"\n❌ 크롤링 실패: {exc}")
        return 1

    status = result.get("status", "unknown")

    if status != "completed_success":
        print("\n⚠️  크롤링이 완전히 성공하지 않았습니다.")
        print("=" * 50)
        print(f"사이트: {result.get('site_name', site_name)}")
        if "error" in result:
            print(f"에러: {result['error']}")
        print(f"상태: {status}")
        return 1

    print("\n✅ 크롤링 완료!")
    print("=" * 50)
    print(f"사이트: {result['site_name']}")
    print(f"크롤링된 항목: {result['crawled_count']}")
    print(f"처리된 항목: {result['processed_count']}")
    print(f"제거된 중복: {result.get('duplicates_removed', 0)}")
    print(f"소요 시간: {result.get('duration_seconds', 0):.2f}초")

    if result.get("saved_paths"):
        print("\n📁 저장된 파일:")
        for fmt, path in result["saved_paths"].items():
            print(f"  {fmt.upper()}: {path}")

    if result.get("mongo_result"):
        stats = result["mongo_result"]
        print("\n🗄️  MongoDB 요약:")
        print(
            f"  deleted={stats.get('deleted', 0)} inserted={stats.get('inserted', 0)}"
        )
        print(f"  embedded_new_records={stats.get('embedded_new_records', 0)}")

    print(f"\n상태: {result['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
