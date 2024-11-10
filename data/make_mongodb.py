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
    g.add_argument("--jsonl_file_path", type=str, default=config["path"]["data_file_name"], help="data path")
    g.add_argument("--db_name", type=str, default=config["path"]["db_name"], help="database name")
    g.add_argument("--collection_name", type=str, default=config["path"]["collection_name"], help="collection name")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## jsonl_file_path : ", args.jsonl_file_path)
    print("## db_name : ", args.db_name)
    print("## chunk_size : ", args.chunk_size)
    print("## overlap_size : ", args.overlap_size)

    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
    collection = client[args.db_name][args.collection_name]
    
    with open(args.jsonl_file_path, 'r', encoding='utf-8') as f:
        for document in f:
            document_dict = ast.literal_eval(document)
            result = collection.insert_one(document_dict)
            print("## Document inserted with ID:", result.inserted_id)

    print('## Document uploaded!')
    client.close()

if __name__ == "__main__":
    with open(f'configs/{CONFIG_FILE_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)
