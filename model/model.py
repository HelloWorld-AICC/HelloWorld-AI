import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# dotenv setting
from dotenv import load_dotenv
load_dotenv(verbose=True)

CONFIG_NAME = "mongo_config.json"

print("## config_name : ", CONFIG_NAME)

with open(f'configs/{CONFIG_NAME}', 'r') as f:
    config = json.load(f)

if config['db'] == 'elasticsearch':
    os.environ["ES_CLOUD_ID"] = os.getenv("ES_CLOUD_ID")
    os.environ["ES_USER"] = os.getenv("ES_USER")
    os.environ['ES_PASSWOR'] = os.getenv("ES_PASSWORD")
    os.environ["ES_API_KEY"] = os.getenv("ES_API_KEY")

elif config['db'] == 'mongo':
   os.environ["MONGODB_ATLAS_CLUSTER_URI"] = os.getenv("MONGODB_ATLAS_CLUSTER_URI")


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")



# socket 설정 : 모델 다운로드 중 연결 끊김 방지
import socket
socket.setdefaulttimeout(500)

def load_db():
    torch.cuda.empty_cache()
    if config['db'] == 'elasticsearch':
        return ElasticsearchStore(
            index_name=config['path']['db_name'],
            embedding = OpenAIEmbeddings()
        )
    elif config['db'] == 'mongo':
       client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
       MONGODB_COLLECTION = client[config['db_name']][config['collection_name']]
       return MongoDBAtlasVectorSearch(
        collection = MONGODB_COLLECTION,
        embedding = OpenAIEmbeddings(model="text-embedding-3-large"),
        index_name = config['collection_name'],
        relevance_score_fn = "cosine" # [cosine, euclidean, dotProduct]
    )
       
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config['config']['model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(tokenizer.eos_token_id)
    ]
    return tokenizer, terminators

def load_model():
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(config['config']['model_id'], 
                                                    device_map=config['device'],
                                                    low_cpu_mem_usage=True) 
    model.eval()
    return model


# 대화 전처리
def preprocess_dialog(dialog: list[dict], current_query=None) -> str:
    chat = ["[대화]"]

    if len(dialog) != 0:
        for cvt in dialog:
            chat.append(f"{cvt['sender']}: {cvt['contents']}")
    
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
    self.chatbot_model = load_model()

    self.db = None

  def get_answer(self, query:str, prev_turn: list[dict], top_k=config['config']['top_k']) -> str:
    print('## Chat mode ##')
    if self.db == None:
        print('## Loading DB... ##')
        self.db = load_db()

    print(f"## We will retrieve top-{top_k} relevant documents and Answer ##")
    similar_docs = self.db.similarity_search(query)
    informed_context= ' '.join([x.page_content for x in similar_docs[:top_k]])

    PROMPT = f"""당신은 유능한 AI 어시스턴트입니다. [관련 문서]를 참조하여, [대화]에 대한 적절한 [답변]을 생성해주세요.\n\n[관련 문서]\n{informed_context}"""
    QUERY_PROMPT = f"""{preprocess_dialog(prev_turn, query)}\nbot: """
    message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": QUERY_PROMPT},
    ]

    source = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
    )

    with torch.no_grad():
        outputs = self.chatbot_model.generate(
                source.to(config['device']),
                max_new_tokens=config['chat_inference']['max_new_tokens'],
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=config['chat_inference']['do_sample'],
                num_beams=config['chat_inference']['num_beams'],
                temperature=config['chat_inference']['temperature'],
                top_k=top_k,
                top_p=config['chat_inference']['top_p'],
                no_repeat_ngram_size=config['chat_inference']['no_repeat_ngram_size'],
            )
    
    inference = self.tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)

    return inference

  def get_summary(self, dialog: list[dict]) -> str:
    print('## We will summary dialog ##')

    PROMPT = f"""당신은 유능한 AI 어시스턴트입니다. [대화]를 보고, [요약문]을 생성해주세요.\n"""
    message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": preprocess_dialog(dialog) + "\n\n[요약문]\n"},
    ]

    source = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
    )

    with torch.no_grad():
        outputs = self.chatbot_model.generate(
                source.to(config['device']),
                max_new_tokens=config['summary_inference']['max_new_tokens'],
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=config['summary_inference']['do_sample'],
                num_beams=config['summary_inference']['num_beams'],
                temperature=config['summary_inference']['temperature'],
                top_k=config['summary_inference']['top_k'],
                top_p=config['summary_inference']['top_p'],
                no_repeat_ngram_size=config['summary_inference']['no_repeat_ngram_size'],
            )
    
    inference = self.tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)

    return inference