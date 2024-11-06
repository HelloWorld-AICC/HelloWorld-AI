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
        self.generate_columns = ["추가정보", "해결방법"]

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
                        "상담일시": row["상담일시"]
                    }
                })
        
        metadata = pd.DataFrame(columns=['request_id', 'metadata'])
        metadata["request_id"] = id_list
        metadata["metadata"] = metadata_list

        metadata.to_csv(meta_path, index=False)

        print(f"## jsonl file saved in {batch_jsonl_path}")
        print(f"## meta file saved in {meta_path}")

        batch_input_file = self.client.files.create(
            file=open(batch_jsonl_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        metadata = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint = url,
            completion_window="24h",
            metadata={
                "description": "make embedding"
            }
        )

        return metadata


    # generate batch 처리
    def make_generate(self, df: pd.DataFrame, column_name: str, batch_jsonl_path : str, meta_path : str,  prompt_path=None):
        id_list = []
        metadata_list = []

        if column_name in self.generate_columns:
            # 프롬프트 불러오기
            with open(prompt_path, 'r', encoding='utf-8') as file:
                PROMPT = file.read()
            
            url = "/v1/chat/completions"
            print('## Generate model loaded')
        
        else:
            raise ValueError("## Wrong column name!")

        ## df column rename
        df.rename(columns={"제목" : "상담제목", "상담내용" : "상담내용요약",
                   "진행 과정 및 결과" : "진행_과정_및_결과", "작성일" : "상담일시"
                   }, inplace = True)

        with open(batch_jsonl_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                json_line = json.dumps(
                                        {"custom_id" : f"request-{idx}",
                                        "method" : "POST",
                                        "url" : url,
                                        "body" : {"model": self.config['generate_model'],
                                                    "messages": [
                                                            {"role": "system", "content": PROMPT},
                                                            {"role": "user", "content": row[column_name]}
                                                        ]
                                                    },
                                                    "max_tokens" : self.config['max_tokens']
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
                        "raw_text": row["상담내용"],
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
                        "상담일시": row["상담일시"]
                    }
                })
        
        metadata = pd.DataFrame(columns=['request_id', 'metadata'])
        metadata["request_id"] = id_list
        metadata["metadata"] = metadata_list

        metadata.to_csv(meta_path, index=False)

        print(f"## jsonl file saved in {batch_jsonl_path}")
        print(f"## meta file saved in {meta_path}")

        # batch_input_file = self.client.files.create(
        #     file=open(batch_jsonl_path, "rb"),
        #     purpose="batch"
        # )


        # batch_input_file_id = batch_input_file.id

        # metadata = self.client.batches.create(
        #     input_file_id=batch_input_file_id,
        #     endpoint=url,
        #     completion_window="24h",
        #     metadata={
        #     "description": "make generation"
        #     }
        # )

        # return metadata
        return


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


    # batch request 받기 "(S | edge | O)"
    # return_type : embedding or generate
    def get_data(self, gpt_answer_path:str, meta_path: str, result_jsonl_path:str, return_type: str, column_name:str) -> list:
        """
        gpt batch request를 받아, Document format에 맞추기
        -> metadata와 매치하여 노드 전체 property + edge 관계를 담은 triplet list 반환
        
        output : jsonl 형태
        """
        
        # jsonl result path
        # UTF-8 인코딩으로 파일 열기
        with open(gpt_answer_path, 'r', encoding='utf-8') as file:
            data = []
            for line in file:
                # 각 줄을 JSON 객체로 파싱
                json_obj = json.loads(line)
                data.append(json_obj)

        print(f"## {len(data)} 개의 데이터 로드됨. 데이터 예시 : {data[0]}")

        # metadata csv
        metadata_csv = pd.read_csv(meta_path)
        

    
        ## generate 경우
        if return_type == "generate" and column_name in self.generate_columns:
            for idx, request in enumerate(data):
                answer = request["response"]["body"]["choices"][0]["message"]["content"]

                # 동일한 request_id 가진 metadata 불러와서 매칭하기
                metadata = metadata_csv.loc[metadata_csv["request_id"] == request["custom_id"], "metadata"]
                metadata = ast.literal_eval(metadata.values[0])
                
                if column_name == "추가정보":
                    data[idx]["내담자_정보"]["추가정보"] = answer
                elif column_name == "해결방법":
                    data[idx][column_name] = answer
                else:
                    raise ValueError("## Wrong column name : ", column_name)
                
                json_line = json.dumps({})
                f.write(json_line + '\n')
        
        
        elif return_type == "embedding" and column_name in self.embedding_columns:
            for idx, request in enumerate(data):
                embedding = data['response']['body']['data'][0]['embedding']
            
                # 동일한 request_id 가진 metadata 불러와서 매칭하기
                metadata = metadata_csv.loc[metadata_csv["request_id"] == request["custom_id"], "metadata"]
                metadata = ast.literal_eval(metadata.values[0])

                try:
                    data[idx][column_name]["Embedding"] = embedding
                except:
                    raise ValueError("## Wrong column name : ", column_name)
                
        
        else:
            raise ValueError(f"## Wrong return type {return_type} or column name {column_name}!")


        with open(result_jsonl_path, 'w', encoding='utf-8') as f:
            for row in data:
                json_line = json.dumps(row, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"## jsonl file saved in {result_jsonl_path}")

        return