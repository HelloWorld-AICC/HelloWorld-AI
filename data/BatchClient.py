from dotenv import load_dotenv
import sys, os
import json
import pandas as pd

import re
from tqdm import tqdm
import ast
import copy

from openai import OpenAI


class BatchClient:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.config = config
        self.embedding_columns = ["상담제목", "상담내용요약"]

        load_dotenv(verbose=True)
        print('## config loaded ##')

        self.client = OpenAI(api_key = os.getenv('OPENAI_KEY'))
        print('## client loaded ##')


    # embedding batch 처리
    def make_embedding(self, df: pd.DataFrame, column_name: str, batch_jsonl_path : str, meta_path : str):
        id_list = []
        metadata_list = []

        if column_name in self.embedding_columns:
            url = "/v1/embeddings"
            
            print('## Embedding model loaded')
        
        else:
            raise ValueError("## Wrong column name!")
        

        ## df column rename
        df.rename(columns={"제목" : "상담제목", "상담내용" : "상담내용요약",
                   "진행 과정 및 결과" : "진행_과정_및_결과", "작성일" : "상담일시"
                   }, inplace = True)

        with open(batch_jsonl_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                # print('## row : ', row)
                json_line = json.dumps(
                                        {"custom_id" : f"request-{idx}",
                                        "method" : "POST",
                                        "url" : url,
                                        "body" : {
                                            "input" : row[column_name],
                                            "model": self.config['embedding_model'],
                                            "encoding_format" : "float",
                                            "dimensions" : 1024},
                                        },
                                        ensure_ascii=False
                                    )
                
                f.write(json_line + '\n')

                id_list.append(f"request-{idx}")
            
                metadata_list.append({
                    "상담제목": {
                        "raw_text": row["상담제목"],
                        "Embedding": []
                    },
                    "상담내용요약": {
                        "raw_text": row["상담내용요약"],
                        "Embedding": []
                    },
                    "진행_과정_및_결과": row["진행_과정_및_결과"],
                    "내담자_정보": {
                        "거주지역": row["거주지역"],
                        "국적": row["국적"],
                        "체류자격": row["체류자격"],
                        "추가정보": []  # LLM 추출 부분 빈 리스트로 초기화
                    },
                    "해결방법": "",  # LLM 추출 부분 빈 문자열로 초기화
                    "metadata": {
                        "상담유형": row["상담유형"],
                        "상담일시": row["상담일시"].strftime('%Y-%m-%d %H:%M:%S')
                    }
                })
        
        
        print(f"## jsonl file saved in {batch_jsonl_path}")

        if os.path.exists(meta_path):
            print('## meta file already exists!')

        else:
            metadata_csv = pd.DataFrame(columns=['request_id', 'metadata'])
            metadata_csv["request_id"] = id_list
            metadata_csv["metadata"] = metadata_list

            metadata_csv.to_csv(meta_path, index=False)
            print(f"## meta file saved in {meta_path}")
            

        batch_input_file = self.client.files.create(
            file=open(batch_jsonl_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        api_meta = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint = url,
            completion_window="24h",
            metadata={
                "description": "make embedding"
            }
        )

        return api_meta


    # generate batch 처리 - 내담자정보, 해결방안
    def make_generate(self, df: pd.DataFrame, batch_jsonl_path : str, meta_path : str,  prompt_path=None):
        id_list = []
        metadata_list = []

        with open(prompt_path, 'r', encoding='utf-8') as file:
            PROMPT = file.read()
        
        PROMPT = ast.literal_eval(PROMPT)
        url = "/v1/chat/completions"
        print('## Generate model loaded')
        
        ## df column rename
        df.rename(columns={"제목" : "상담제목", "상담내용" : "상담내용요약",
                   "진행 과정 및 결과" : "진행_과정_및_결과", "작성일" : "상담일시"
                   }, inplace = True)

        ## string validation test ##

        with open(batch_jsonl_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():

                user_input = f"""<제목>{row['상담제목']}</제목>
                <전체상담내용>{row['상담내용요약']}</전체상담내용>
                <상담기록>\n{row['진행_과정_및_결과']}</상담기록>"""

                json_line = json.dumps(
                                        {"custom_id" : f"request-{idx}",
                                        "method" : "POST",
                                        "url" : url,
                                        "body" : {"model": self.config['generate_model'],
                                                    "messages": [
                                                            {"role": "system", "content": PROMPT["system_prompt"]},
                                                            {"role": "user", "content": PROMPT["user_prompt"] + "\n" + user_input + "\n\n[output]\n"}
                                                        ],
                                                    "max_tokens" : self.config['max_tokens']},
                                        }, 
                                        ensure_ascii=False
                                    )
                f.write(json_line + '\n')

                id_list.append(f"request-{idx}")
            
                metadata_list.append({
                    "상담제목": {
                        "raw_text": row["상담제목"],
                        "Embedding": []
                    },
                    "상담내용요약": {
                        "raw_text": row["상담내용요약"],
                        "Embedding": []
                    },
                    "진행_과정_및_결과": row["진행_과정_및_결과"],
                    "내담자_정보": {
                        "거주지역": row["거주지역"],
                        "국적": row["국적"],
                        "체류자격": row["체류자격"],
                        "추가정보": []  # LLM 추출 부분 빈 리스트로 초기화
                    },
                    "해결방법": "",  # LLM 추출 부분 빈 문자열로 초기화
                    "metadata": {
                        "상담유형": row["상담유형"],
                        "상담일시": row["상담일시"].strftime('%Y-%m-%d %H:%M:%S')
                    }
                })
        
        print(f"## jsonl file saved in {batch_jsonl_path}")

        # metafile 없을 때
        if os.path.exists(meta_path):
            print('## meta file already exists!')

        # metafile 있을 때
        else:
            metadata_csv = pd.DataFrame(columns=['request_id', 'metadata'])
            metadata_csv["request_id"] = id_list
            metadata_csv["metadata"] = metadata_list

            metadata_csv.to_csv(meta_path, index=False)
            print(f"## meta file saved in {meta_path}")


        batch_input_file = self.client.files.create(
            file=open(batch_jsonl_path, "rb"),
            purpose="batch"
        )


        batch_input_file_id = batch_input_file.id

        api_meta = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=url,
            completion_window="24h",
            metadata={
            "description": "make generation"
            }
        )

        return api_meta

    # triplet extractor 배치 취소
    def cancel_batch(self, id_list, custom_id=None):
        # custom id 입력 있을 경우 -> 특정 request id만 cancel
        if custom_id != None:
            self.client.batches.cancel(custom_id)
        
        else:
            for request_id in id_list:
                self.client.batches.cancel(request_id)
        return
    

    # batch id 받아서 status 확인
    def get_batch_status(self, batch_id):
        return self.client.batches.retrieve(batch_id)

    def get_batch_answer(self, output_file_id, result_path):
        answer = self.client.files.content(output_file_id).content
        with open(result_path, 'wb') as file:
            file.write(answer)
        
        print(f'## file saved in {result_path}')
        return
