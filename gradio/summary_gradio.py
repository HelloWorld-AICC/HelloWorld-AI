import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

import argparse
from typing import Dict
import json
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from langchain_elasticsearch import ElasticsearchStore
import gradio as gr
from langchain_openai import OpenAIEmbeddings
from model_integrated import BllossomModel

load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

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
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## db_name : ", args.db_name)
    print("## top_k : ", args.top_k)
    print("## chunk_size : ", args.chunk_size)
    
    model = BllossomModel()

    # conversational loop
    def inference(dialog):
        print(f'## {len(dialog)} turns summary ##')
        response = model.get_summary(dialog=dialog)
        return dialog, response

    gr.Interface(
        fn=inference,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="사장님이 월급을 세 달째 안줘요.."),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Original dialog",
            ),
            gr.components.Textbox(
                lines=5,
                label="Retrieved summarization",
            )
        ],
        title="상담 챗봇 프로토타입",
        description="Hello! I am a QA chat bot for Foreigners, ask me any question about it. (Model : MLP-KTLim/llama-3-Korean-Bllossom-8B)",
    ).queue().launch(share=True, debug=True)

        

if __name__ == "__main__":
    with open(f'configs/{CONFIG_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)