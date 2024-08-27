import json
import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import os
from langchain.docstore.document import Document

# 전처리 및 청킹 -> 리스트 형태의 텍스트 반환
def data_preprocess(CONFIG_FILE_NAME: str) -> Document:
    with open(f'configs/{CONFIG_FILE_NAME}', 'r') as f:
        config = json.load(f)

    MODEL_ID = config['config']['model_id']
    CHUNK_SIZE = config['config']['chunk_size']
    OVERLAP_SIZE = config['config']['overlap_size']
    DATA_FILE_NAME = config['path']['data_file_name']

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print('## config file loaded ##')

    with open(os.path.join('data/', DATA_FILE_NAME), 'r') as f:
        data = json.load(f)
    
    titles = []; contents = []; sources = []

    for row in data:
        titles.append(row['title'])
        contents.append(row['content'])
        sources.append({'source': row['source']})

    print(f'## raw data (length : {len(data)}) loaded ##')
    
    # 특수 문자를 제거하고 연속된 공백을 하나로 줄인다.
    def remove_escape(raw_text: str) -> str:
        pattern = r"\t|\n|\xa0"
        processed_text = re.sub(pattern, " ", raw_text)
        processed_text_stripped = " ".join(processed_text.split())
        return processed_text_stripped

    def remove_hanja(text):
        # Unicode 범위를 사용하여 한자 제거
        return re.sub(r'[\u4e00-\u9fff]+', '', text)
    
    #하이퍼링크 제거
    def remove_hyperlink(raw_text: str) -> str:
        pattern = (
            r":*\s*\(*:*\s*https?://[\w\dㄱ-ㅎㅏ-ㅣ가-힣!@#$%^&*(),.?/:;\"'<>{}|+=~_-]+\s*\)*"
        )
        processed_text = re.sub(pattern, "", raw_text)
        return processed_text
    
    #텍스트 시작 부분 헤더 제거
    def remove_header(raw_text: str) -> str:
        header_pattern = "안녕하십니까. 대한법률구조공단 사이버상담을 이용해 주셔서 감사합니다."
        header_end_idx = re.search(header_pattern, raw_text)
        if header_end_idx != None:
            processed_text = raw_text[header_end_idx.end() :]
            return processed_text
        else:
            return raw_text
        
    #텍스트 끝 부분 푸터 제거
    def remove_footer(raw_text: str) -> str:
        footer_pattern = "※ 주의 : 사례에 대한 답변은 법령이나 판례 등의 변경으로 내용이 바뀔 수 있으므로 구체적인 사안에 대해서는 반드시 대한법률구조공단 상담(전화상담은 국번없이 ☎ 132) 등을 통해 다시 한 번 확인하시기 바랍니다."
        footer_start_idx = re.search(footer_pattern, raw_text)
        if footer_start_idx != None:
            processed_text = raw_text[: footer_start_idx.start()]
            return processed_text
        else:
            return raw_text
    
    def remove_author_and_url(text):
        # 작성자 정보 제거
        text = re.sub(r'작성자:\s*[\w\s]+', '', text)
        
        # URL 제거
        text = re.sub(r'URL:\s*https?://\S+', '', text)
        
        # 마지막 줄바꿈 제거
        text = text.strip()
        
        return text
    
    #특정 키워드가 포함된 문장 제거
    def remove_page_word(raw_text: str) -> str:
        pattern = '사이버상담|사이버 상담|공단|방문|국번없이 132|132번'
        if re.findall(pattern, raw_text) == []:
            return raw_text
        
        split_text = raw_text.split('.')
        remove_text = [i for i in split_text if re.findall(pattern, i) == []]        

        return '.'.join(remove_text)

    def remove_phone_number(raw_text: str) -> str:
        pattern = r'\b(\d{2,3}-\d{3,4}-\d{4}|\d{2}-\d{3}-\d{4})\b'
        processed_text = re.sub(pattern, "", raw_text)
        return processed_text

    # RAG style : title + [SEP] + contents
    # seperation 지정되어 있지 않음 -> 공백으로 처리
    def make_chunk_data(titles: list, contents: list) -> list:
        if tokenizer.sep_token != None:
            print('## seperated by : ', tokenizer.sep_token)
            chunk_data = [title + tokenizer.sep_token + content for title, content in zip(titles, contents)]
        else:
            chunk_data = [title + " " + content for title, content in zip(titles, contents)]
        
        return chunk_data

    # 토크나이저 기준 분할
    def data_chunking(titles: list, contents: list, sources: list) -> Document:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE
        )
        chunk_data = make_chunk_data(titles, contents)
        chunks = text_splitter.create_documents(chunk_data, sources)

        return chunks
        
    
    preprocess_functions = [
        remove_hanja,
        remove_header,
        remove_footer,
        #remove_escape,
        remove_phone_number,
        #remove_page_word,
        remove_hyperlink,
        remove_author_and_url,
        #remove_link,
    ]

    for preprocess_function in preprocess_functions:
        contents = list(map(preprocess_function, contents))

    chunks = data_chunking(titles, contents, sources)

    return chunks