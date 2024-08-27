import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

import argparse
from typing import Dict
import json
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, PretrainedConfig
import torch
from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

CONFIG_NAME = "chat_config.json"
print("## config_name : ", CONFIG_NAME)

def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--model_id", type=str, default=config["config"]["model_id"], help="model id")
    g.add_argument("--chunk_size", type=int, default=config["config"]["chunk_size"], help="data chunk size")
    g.add_argument("--top_k", type=int, default=config["config"]["top_k"], help="How many documents are retrieved")
    g.add_argument("--cache_dir", type=str, default="./cache", help="cache directory path")
    g.add_argument("--template_name", type=str, default=config["path"]["template_name"], help="What template to load")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## model_id : ", args.model_id)
    print("## top_k : ", args.top_k)
    print("## chunk_size : ", args.chunk_size)
    print("## template_name : ", args.template_name)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print('## EOS token : ',tokenizer.eos_token)

    def load_db():
        torch.cuda.empty_cache()
        return ElasticsearchStore(
            es_cloud_id=ES_CLOUD_ID,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            es_api_key=ES_API_KEY,
            index_name=args.db_name,
            # embedding=HuggingFaceEmbeddings(model_name=args.model_id,
            #                                 cache_folder=args.cache_dir)
            embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
        )
    
    def load_model():
        torch.cuda.empty_cache()

        

        model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                     cache_dir=args.cache_dir,
                                                     device_map=config['device'],
                                                     low_cpu_mem_usage=True) 
        model.eval()

        model_config = PretrainedConfig(
            max_length = config['inference']['max_length'],
            do_sample = config['inference']['do_sample'],
            num_beams = config['inference']['num_beams'],
            temperature = config['inference']['temperature'],
            top_k = config['inference']['top_k'],
            top_p = config['inference']['top_p'],
            no_repeat_ngram_size = config['inference']['no_repeat_ngram_size'],
        )

        pipe = pipeline(
                "text2text-generation",
                model=model,
                config=model_config,
                tokenizer=tokenizer, 
                max_length=config['inference']['max_length']
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"## Get {args.model_id} ready to go ##")

        # with open(f'templates/{args.template_name}.txt', 'r') as f:
        #     template = f.readlines()
        #     template = ''.join(template)

        template_informed = """
        당신은 유능한 AI 어시스턴트입니다.
        [관련 문서]를 참조하여 [질문]에 대한 적절한 답변을 생성해주세요.
        [관련 문서] : {context}
        [질문] : {question}
        [답변] : """
        prompt_informed = PromptTemplate(template=template_informed, input_variables=["context", "question"])
        
        
        return LLMChain(prompt=prompt_informed, llm=llm)
    
    print('## Loading DB... ##')
    db = load_db()
    print('## Loading Model... ##')
    chatbot_model = load_model()

    print("## Conversation Start !! ##")
    print(f'## We will retrieve top-{args.top_k} relevant documents!')
    while True:
        query = input("질문 >> ")
        similar_docs = db.similarity_search(query)
        print(f"\n\n## 가장 유사도 높은 top-{args.top_k} passage:\n")
        for i in range(args.top_k):
            document = similar_docs[i].page_content
            print(f"- top{i+1} : {document}")
        
        ## Ask Local LLM context informed prompt
        informed_context= ' '.join([x.page_content for x in similar_docs[:args.top_k]])

        informed_response = chatbot_model.run(context=informed_context,question=query, stop=tokenizer.eos_token)

        print(f"\n\n답변  : {informed_response}")

        
if __name__ == "__main__":
    with open(f'configs/{CONFIG_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)