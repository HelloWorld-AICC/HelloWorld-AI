import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import LlamaCpp

# dotenv setting
from dotenv import load_dotenv
load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

CONFIG_NAME = "config.json"
print("## config_name : ", CONFIG_NAME)

with open(f'configs/{CONFIG_NAME}', 'r') as f:
    config = json.load(f)


REPO_PATH = "ywhwang/llama-3-Korean-Bllossom-8B-Q5_K_M"

# socket 설정 : 모델 다운로드 중 연결 끊김 방지
import socket
socket.setdefaulttimeout(500)

def load_db():
    torch.cuda.empty_cache()
    return ElasticsearchStore(
        es_cloud_id=ES_CLOUD_ID,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        es_api_key=ES_API_KEY,
        index_name=config['path']['db_name'],
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    )

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config['config']['model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    return tokenizer, terminators

def load_model(terminators):
    torch.cuda.empty_cache()
    # model = AutoModelForCausalLM.from_pretrained(config['config']['model_id'], 
    #                                                 device_map=config['device'],
    #                                                 low_cpu_mem_usage=True) 
    # quantized model download

    if os.path.isfile(config['config']['quantized_path']) == False:
        model = AutoModelForCausalLM.from_pretrained(REPO_PATH, model_file="", model_type="llama")
        model.save_pretrained(config['config']['quantized_path'])

        print(f"## Model saved to {config['config']['quantized_path']}")


    model = LlamaCpp(
            model_path=config['config']['quantized_path'],
            n_gpu_layers=-1,
            n_batch=config['config']['n_batch'],
            n_ctx = config['config']['n_ctx'],
            temperature=config['chat_inference']['temperature'],
            top_p=config['chat_inference']['top_p'],
            top_k= config['chat_inference']['top_k'],
            max_tokens=config['summary_inference']['max_length'],
            repeat_penalty=config['chat_inference']['repeat_penalty'],
            stop=terminators,
            verbose=True
    )
    
    return model


# 대화 전처리
def preprocess_dialog(dialog: list[dict], current_query=None) -> str:
    chat = ["[대화]"]

    if len(dialog) != 0:
        for cvt in dialog:
            chat.append(f"{cvt['sender']}: {cvt['content']}")
    
    if current_query != None:
        chat.append(f"user: {current_query}")

    chat = "\n".join(chat)
    return chat

class BllossomModel:
  def __init__(self):
    self.initialize()


  def initialize(self):
    print('## Loading Tokenizer... ##')
    self.tokenizer, self.terminators = load_tokenizer()

    print('## Loading Model... ##')
    self.chatbot_model = load_model(self.terminators)

    self.db = None

  def get_answer(self, query:str, prev_turn: list[dict]) -> str:
    print('## Chat mode ##')
    if self.db == None:
        print('## Loading DB... ##')
        self.db = load_db()

    print(f"## We will retrieve top-{config['config']['top_k']} relevant documents and Answer ##")
    similar_docs = self.db.similarity_search(query)
    informed_context= ' '.join([x.page_content for x in similar_docs[:config['config']['top_k']]])

    PROMPT = f"""당신은 유능한 AI 어시스턴트입니다. [관련 문서]를 참조하여, [대화]에 대한 적절한 [답변]을 생성해주세요.\n\n[관련 문서]\n{informed_context}"""
    QUERY_PROMPT = f"""{preprocess_dialog(prev_turn, query)}\nbot: """
    message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": QUERY_PROMPT},
    ]

    # Expected to string
    source = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False
    )

    print('## source test : ', source)

    with torch.no_grad():
        outputs = self.chatbot_model.invoke(
                source,
                max_tokens=config['chat_inference']['max_new_tokens']
            )
    # inference = self.tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)

    return outputs

  def get_summary(self, dialog: list[dict]) -> str:
    print('## We will summary dialog ##')

    PROMPT = f"""당신은 유능한 AI 어시스턴트입니다. [대화]를 보고, [요약문]을 생성해주세요.\n"""
    message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": preprocess_dialog(dialog) + "\n\n[요약문]\n"},
    ]

    # Expected to string
    source = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False
    )
    print('## source test : ', source)


    with torch.no_grad():
        outputs = self.chatbot_model.invoke(
                source,
                max_new_tokens=config['summary_inference']['max_new_tokens']
            )
    
    # inference = self.tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)

    return outputs