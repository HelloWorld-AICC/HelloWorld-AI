import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

import argparse
from typing import Dict
import json
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, PretrainedConfig
import torch
from langchain_elasticsearch import ElasticsearchStore
from langchain.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
# llama-cpp-python 설치 필요
from typing import List

load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")

CONFIG_NAME = "config.json"
print("## config_name : ", CONFIG_NAME)

def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--model_id", type=str, default=config["config"]["model_id"], help="model id")
    g.add_argument("--chunk_size", type=int, default=config["config"]["chunk_size"], help="data chunk size")
    g.add_argument("--top_k", type=int, default=config["config"]["top_k"], help="How many documents are retrieved")
    g.add_argument("--db_name", type=str, default=config["path"]["db_name"], help="data index name to save")
    g.add_argument("--cache_dir", type=str, default="./cache", help="cache directory path")
    g.add_argument("--template_name", type=str, default=config["path"]["template_name"], help="What template to load")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## db_name : ", args.db_name)
    print("## top_k : ", args.top_k)
    print("## chunk_size : ", args.chunk_size)
    print("## template_name : ", args.template_name)
    

    def load_db():
        return ElasticsearchStore(
            es_cloud_id=ES_CLOUD_ID,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            es_api_key=ES_API_KEY,
            index_name=args.db_name,
            embedding=LlamaCppEmbeddings(model_path=config['config']['quantized_path'],
                                        n_gpu_layers=-1,
                                        n_batch=config['config']['n_batch'],
                                        n_ctx=config['config']['n_ctx'])
        )
    
    def load_model():
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        # terminators = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        # # Embeddings by llama.cpp quantized model


        model = LlamaCpp(
            model_path=config['config']['quantized_path'],
            n_gpu_layers=-1,
            n_batch=config['config']['n_batch'],
            n_ctx = config['config']['n_ctx'],
            temperature=config['inference']['temperature'],
            top_p=config['inference']['top_p'],
            top_k= config['inference']['top_k'],
            max_tokens=config['inference']['max_length'],
            repeat_penalty=config['inference']['repeat_penalty']
        )
        model.eval()

        pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer, 
                max_length=config['inference']['max_length'],
                device=config['device']
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"## Get {args.model_id} ready to go ##")

        with open(f'templates/{args.template_name}.txt', 'r') as f:
            template = f

        print('## template : ', template)
        prompt_informed = PromptTemplate(template=template, input_variables=["context", "question"])
        return LLMChain(prompt=prompt_informed, llm=llm)
    
    print('## Loading DB... ##')
    db = load_db()
    print('## Loading Model... ##')
    chatbot_model = load_model()

    print("## Conversation Start !! ##")
    print(f'## We will retrieve top-{args.top_k} relevant documents!')
    while True:
        query = input("질문 >> ")
        print("질문 : ", query)
        similar_docs = db.similarity_search(query)
        print(f"## 가장 유사도 높은 top-{args.top_k} passage:\n")
        for i in range(args.top_k):
            document = similar_docs[i].page_content
            print(f"- top{i+1} : {document}")
        
        ## Ask Local LLM context informed prompt
        informed_context= ' '.join([x.page_content for x in similar_docs[:args.top_k]])
        informed_response = chatbot_model.run(context=informed_context,question=query)

        print(f"\t답변  : {informed_response}")

        

if __name__ == "__main__":
    with open(f'configs/{CONFIG_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)