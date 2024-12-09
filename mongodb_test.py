from pymongo import MongoClient
import sys, os
from dotenv import load_dotenv

load_dotenv(verbose=True)

client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"), ssl=True)
print(client.server_info())  # 연결 성공 시 서버 정보 출력