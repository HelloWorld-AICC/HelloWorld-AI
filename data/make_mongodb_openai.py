import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

import argparse
from typing import Dict
import json

from uuid import uuid4

from utils import data_preprocess

load_dotenv(verbose=True)

os.environ["MONGODB_ATLAS_CLUSTER_URI"] = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

CONFIG_FILE_NAME = "config.json"

def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--model_id", type=str, default=config["config"]["model_id"], help="model id")
    g.add_argument("--chunk_size", type=str, default=config["config"]["chunk_size"], help="data chunk size")
    g.add_argument("--overlap_size", type=str, default=config["config"]["overlap_size"], help="chunk overlapping size")
    g.add_argument("--data_file_name", type=str, default=config["path"]["data_file_name"], help="data path")
    g.add_argument("--db_name", type=str, default=config["path"]["db_name"], help="data index name to save")
    g.add_argument("--collection_name", type=str, default=config["path"]["collection_name"], help="vectorstore name to save")
    g.add_argument("--index_name", type=str, default=config["path"]["collection_name"], help="index name to save")
    g.add_argument("--cache_dir", type=str, default="./cache", help="cache directory path")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## data_file_name : ", args.data_file_name)
    print("## db_name : ", args.db_name)
    print("## chunk_size : ", args.chunk_size)
    print("## overlap_size : ", args.overlap_size)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # model?

    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])

    MONGODB_COLLECTION = client[args.db_name][args.collection_name]

    vector_store = MongoDBAtlasVectorSearch(
        collection = MONGODB_COLLECTION,
        embedding = embeddings,
        index_name = args.index_name,
        relevance_score_fn = "cosine" # [cosine, euclidean, dotProduct]
    )


    # 전처리에 동일한 config 전달
    # list[Document] 형태로 반환됨
    # 예시 : [Document(page_content='중도 퇴사 후 ...'), ...]
    batchtext = data_preprocess(CONFIG_FILE_NAME)

    print("## Preprocess Completed!! ##")
    print("## Add Data to MongoDB.. ##")

    # DB에 텍스트 데이터 추가
    uuids = [str(uuid4()) for _ in range(len(batchtext))]

    vector_store.add_documents(documents=batchtext, ids=uuids)

    print("## Data saved to DB!! ##")


if __name__ == "__main__":
    with open(f'configs/{CONFIG_FILE_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)