from dotenv import load_dotenv
import sys, os
import json
import pandas as pd

import re
import ast
embedding_columns = ["상담제목", "상담내용요약"]


# return_type : embedding or generate
# jsonl to metacsv
def get_result_csv(gpt_answer_path:str, meta_path: str, return_type: str, column_name:str) -> list:
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

    # print(f"## {len(data)} 개의 데이터 로드됨. 데이터 예시 : {data[0]}")


    # 이전 meta csv 불러옴
    if os.path.exists(meta_path):
        result_csv = pd.read_csv(meta_path)
        
        result_csv['metadata'] = result_csv['metadata'].apply(ast.literal_eval)

    else:
        raise ValueError("No result meta csv")

    ## generate 경우
    if return_type == "generate":
        for request in data:
            answer = request["response"]["body"]["choices"][0]["message"]["content"]

            # answer 후처리
            try:
                info, solution = answer.split("</내담자정보>")
            except:
                print("## answer : {answer} format is not aligned")

            info = re.sub(r"<\/?\w+>", "", info)
            solution = re.sub(r"<\/?\w+>", "", solution)

            # # 후처리
            # info = [x for x in info.split('\n') if x != ""]
            # solution = [x for x in solution.split('\n') if x != ""]

            # 동일한 request_id 가진 metadata 불러와서 매칭하기

            result_csv.loc[result_csv["request_id"] == request["custom_id"], "metadata"].values[0]["내담자_정보"]["추가정보"] = info
            result_csv.loc[result_csv["request_id"] == request["custom_id"], "metadata"].values[0]["해결방법"] = solution

    
    elif return_type == "embedding" and column_name in ["상담제목", "상담내용요약"]:
        for request in data:
            embedding = request['response']['body']['data'][0]['embedding']
            result_csv.loc[result_csv["request_id"] == request["custom_id"], "metadata"].values[0][column_name]["Embedding"] = embedding
    
    else:
        raise ValueError(f"## Wrong return type {return_type} or column name {column_name}!")
    

    # 덮어 씌우기
    result_csv.to_csv(meta_path, index=False)
    print('#' * 10)
    print(result_csv.loc[0, 'metadata']["상담제목"]['Embedding'])
    print(result_csv.loc[0, 'metadata']["상담내용요약"]['Embedding'])



def get_result_jsonl(meta_path: str, result_jsonl_path: str):
    # metadata csv
    metadata_csv = pd.read_csv(meta_path)
    metadata_csv['metadata'] = metadata_csv['metadata'].apply(ast.literal_eval)


    with open(result_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in metadata_csv.iterrows():
            json_line = json.dumps(row['metadata'], ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"## jsonl file saved in {result_jsonl_path}")

    return


