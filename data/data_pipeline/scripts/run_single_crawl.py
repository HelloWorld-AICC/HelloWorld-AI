#!/usr/bin/env python3
"""
단일 사이트 크롤링 실행 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flows.single_site_flow import crawl_single_site
from config.sites_config import get_enabled_sites, get_site_config


def main():
    """메인 실행 함수"""
    print("🚀 Prefect 기반 데이터 크롤링 시스템")
    print("=" * 50)

    # 활성화된 사이트 목록 표시
    enabled_sites = get_enabled_sites()
    print(f"활성화된 사이트: {', '.join(enabled_sites)}")

    if not enabled_sites:
        print("❌ 활성화된 사이트가 없습니다.")
        return

    # 사용자 입력 받기
    if len(sys.argv) > 1:
        site_name = sys.argv[1]
    else:
        print("\n크롤링할 사이트를 선택하세요:")
        for i, site in enumerate(enabled_sites, 1):
            config = get_site_config(site)
            print(f"{i}. {site} ({config.base_url})")

        try:
            choice = int(input("\n선택 (번호): ")) - 1
            if 0 <= choice < len(enabled_sites):
                site_name = enabled_sites[choice]
            else:
                print("❌ 잘못된 선택입니다.")
                return
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
            return

    # 사이트 설정 확인
    try:
        site_config = get_site_config(site_name)
        print(f"\n📋 선택된 사이트: {site_name}")
        print(f"   URL: {site_config.base_url}")
        print(f"   크롤러: {site_config.crawler_class}")
    except ValueError as e:
        print(f"❌ {e}")
        return

    # 출력 디렉토리 설정
    output_dir = "crawled_data"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📁 결과 저장 디렉토리: {output_dir}")
    print("\n🔄 크롤링 시작...")

    try:
        # 크롤링 실행
        result = crawl_single_site(
            site_name=site_name, output_dir=output_dir, save_formats=["json", "csv"]
        )

        # 결과 출력
        print("\n✅ 크롤링 완료!")
        print("=" * 50)
        print(f"사이트: {result['site_name']}")
        print(f"크롤링된 항목: {result['crawled_count']}")
        print(f"처리된 항목: {result['processed_count']}")
        print(f"제거된 중복: {result.get('duplicates_removed', 0)}")
        print(f"소요 시간: {result.get('duration_seconds', 0):.2f}초")

        if "saved_paths" in result:
            print("\n📁 저장된 파일:")
            for fmt, path in result["saved_paths"].items():
                print(f"  {fmt.upper()}: {path}")

        print(f"\n상태: {result['status']}")

    except Exception as e:
        print(f"\n❌ 크롤링 실패: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
