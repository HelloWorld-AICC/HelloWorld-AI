import json
import logging
import os

# 로깅 설정
logger = logging.getLogger(__name__)

def initialize_config():
    """
    이 함수는 애플리케이션의 설정을 초기화하고, 환경 변수를 설정하는 함수입니다.

    Process:
        1. '.env' 파일에서 환경변수를 로드
        2. 'shared_code/configs/mongo_config.json' 파일에서 MongoDB 및 기타 설정을 로드
        3. MongoDB URI와 OpenAI API키를 환경변수로 설정

    Returns:
        dict: 설정 정보가 담긴 딕셔너리
    """

    # 환경 변수 로드
    logger.info("Starting application initialization...")

    # Config 파일 로드
    CONFIG_NAME = "mongo_config.json"
    logger.info(f"## config_name : {CONFIG_NAME}")

    # 현재 파일(config.py)의 디렉토리 경로를 기준으로 configs 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs', CONFIG_NAME)
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f'## db : {config["db"]}')
    logger.info(f'## db_name : {config["path"]["db_name"]}')

    return config