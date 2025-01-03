import sys, os
# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from pymongo import MongoClient
import argparse
import json

import ast


load_dotenv(verbose=True)

os.environ["MONGODB_ATLAS_CLUSTER_URI"] = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
CONFIG_FILE_NAME = "config.json"


def main(config):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--file_path", type=str, default=config["path"]["data_file_name"], help="data path")
    g.add_argument("--db_name", type=str, default=config["path"]["db_name"], help="database name")
    g.add_argument("--collection_name", type=str, default=config["path"]["collection_name"], help="collection name")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## file_path : ", args.file_path)
    print("## db_name : ", args.db_name)
    print("## collection_name : ", args.collection_name)

    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
    collection = client[args.db_name][args.collection_name]
    
    # jsonl일 때
    if args.file_path.split(".")[-1] == "jsonl":
        with open(args.file_path, 'r', encoding='utf-8') as f:
            print(f"## Upload {args.file_path} file : {len(f)} rows")
            for document in f:
                document_dict = ast.literal_eval(document)
                result = collection.insert_one(document_dict)
                print("## Document inserted with ID:", result.inserted_id)
    
    # json일 때
    elif args.file_path.split(".")[-1] == "json":
        with open(args.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"## Upload {args.file_path} file : {len(data)} rows")

            for document in data:
                result = collection.insert_one(document)
                print("## Document inserted with ID:", result.inserted_id)

    else:
        raise ValueError("올바르지 않은 확장자 : json / jsonl 필요")

    print('## Document uploaded!')
    client.close()

if __name__ == "__main__":
    with open(f'configs/{CONFIG_FILE_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)
