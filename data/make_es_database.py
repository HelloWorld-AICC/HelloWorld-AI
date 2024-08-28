import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore # ElasticSearch vectorstore in langchain style

import argparse
from typing import Dict
import json
from utils import data_preprocess

load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")

CONFIG_FILE_NAME = "config.json"

def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--model_id", type=str, default=config["config"]["model_id"], help="model id")
    g.add_argument("--chunk_size", type=str, default=config["config"]["chunk_size"], help="data chunk size")
    g.add_argument("--overlap_size", type=str, default=config["config"]["overlap_size"], help="chunk overlapping size")
    g.add_argument("--data_file_name", type=str, default=config["path"]["data_file_name"], help="data path")
    g.add_argument("--data_name", type=str, default=config["path"]["data_name"], help="data index name to save")
    g.add_argument("--cache_dir", type=str, default="./cache", help="cache directory path")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## data_file_name : ", args.data_file_name)
    print("## data_name : ", args.data_name)
    print("## chunk_size : ", args.chunk_size)

    hf = HuggingFaceEmbeddings(model_name=args.model_id)
    db = ElasticsearchStore(
        es_cloud_id=ES_CLOUD_ID,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        es_api_key=ES_API_KEY,
        index_name=args.data_name,
        embedding=hf
    )

    # 전처리에 동일한 config 전달
    # list[Document] 형태로 반환됨
    # 예시 : [Document(page_content='중도 퇴사 후 ...'), ...]
    batchtext = data_preprocess(CONFIG_FILE_NAME)

    print("## Preprocess Completed!! ##")
    print("## Add Data to ElasticSearch DB.. ##")

    # DB에 텍스트 데이터 추가
    db.from_documents(batchtext, 
                  embedding=hf,
                  es_cloud_id=ES_CLOUD_ID,
                  es_user=ES_USER,
                  es_password=ES_PASSWORD,
                  es_api_key=ES_API_KEY,
                  index_name=args.data_name)

    print("## Data saved to DB!! ##")


if __name__ == "__main__":
    with open(f'configs/{CONFIG_FILE_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)