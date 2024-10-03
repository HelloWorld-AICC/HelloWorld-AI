from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sys
import json
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from utils import gpt_tokens, model_tokens

from dotenv import load_dotenv

load_dotenv(verbose=True)

CONFIG_NAME = "gpt_config.json"
INCLUDE_RAG = True
THRESHOLD = 0.25


print("## config_name : ", CONFIG_NAME)

with open(f'configs/{CONFIG_NAME}', 'r') as f:
    config = json.load(f)

if config['db'] == 'elasticsearch':
    os.environ["ES_CLOUD_ID"] = os.getenv("ES_CLOUD_ID")
    os.environ["ES_USER"] = os.getenv("ES_USER")
    os.environ['ES_PASSWORD'] = os.getenv("ES_PASSWORD")
    os.environ["ES_API_KEY"] = os.getenv("ES_API_KEY")

elif config['db'] == 'mongo':
   os.environ["MONGODB_ATLAS_CLUSTER_URI"] = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
   api_key = os.getenv('MONGODB_API_KEY')
   cluster_url = "helloworld-ai.fpdjl.mongodb.net" 
   uri = f"mongodb+srv://{api_key}@{cluster_url}/?authMechanism=MONGODB-AWS&authSource=$external"
   print('## db : ',config['db'])
   print('## db_name : ',config['path']['db_name'])


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")


# 로깅 설정
logging.basicConfig(filename='./model/log_file.txt', level=logging.INFO, 
                    format="[ %(asctime)s | %(levelname)s ] %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

if INCLUDE_RAG == False:
    print("## No RAG system ##")
    logger.info("## No RAG system ##")

app = Flask(__name__)
CORS(app)


# 대화 기록을 저장할 리스트
conversations = []; conversations_length = 0


# Load DB once at startup
db = None

try:
    if config['db'] == 'elasticsearch':
        db = ElasticsearchStore(
            index_name='helloworld',
            embedding=OpenAIEmbeddings()
        )
    elif config['db'] == 'mongo':
        client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"], ssl=True,
                             tlsCAFile='/etc/ssl/certs/ca-certificates.crt')
        #client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"],tls=True, tlsInsecure=False)
        
        MONGODB_COLLECTION = client[config['path']['db_name']][config['path']['collection_name']]
        db = MongoDBAtlasVectorSearch(
            collection = MONGODB_COLLECTION,
            embedding = OpenAIEmbeddings(model="text-embedding-3-large"),
            index_name = config['path']['index_name'],
            relevance_score_fn = "cosine" # [cosine, euclidean, dotProduct]
        )
    else:
        raise ValueError("Wrong db value setted in config file")
    
except Exception as e:
    print(f"Error loading database: {str(e)}")


# local model test
def generate_bllossom_response(query, db):
    global conversations
    from transformers import AutoTokenizer
    import torch
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(config["quantized_path"], trust_remote_code=True)

    llm = ChatOpenAI(
        model=config['openai_chat_inference']['model'],
        frequency_penalty=config['openai_chat_inference']['frequency_penalty'],
        logprobs=config['openai_chat_inference']['logprobs'],
        top_logprobs=config['openai_chat_inference']['top_logprobs'],
        max_tokens=config['chat_inference']['max_new_tokens'],  # 최대 토큰수
        temperature=config['chat_inference']['temperature'],  # 창의성 (0.0 ~ 2.0)
    )

    template_text = """
    당신은 한국의 외국인 근로자를 위한 법률 및 비자 전문 AI 어시스턴트입니다. 다음 지침을 따라 응답해 주세요:

    1. 관련 문서의 정보를 바탕으로 정확하고 최신의 법률 및 비자 정보를 제공하세요.
    2. 복잡한 법률 용어나 절차를 쉽게 설명하여 외국인 근로자가 이해하기 쉽게 답변하세요.
    3. 불확실한 정보에 대해서는 명확히 언급하고, 공식 기관에 문의할 것을 권장하세요.
    4. 문화적 차이를 고려하여 정중하고 친절한 태도로 응대하세요.
    5. 필요한 경우 관련 정부 기관이나 지원 센터의 연락처를 제공하세요.
    6. 개인정보 보호를 위해 구체적인 개인 정보를 요구하지 마세요.
    7. 대화 기록을 참고하여 문맥에 맞는 자연스러운 응답을 제공하세요.
    8. 사용자의 이전 질문이나 concerns를 기억하고 연관된 정보를 제공하세요.

    관련 문서: 
    {context}

    이전 대화:
    {conversation_history}
    """

    # history 기록 넣기?
    similar_docs = db.similarity_search_with_relevance_scores(query, k=config['config']['top_k'])
    similar_docs = similar_docs[::-1]

    # for i, doc in enumerate(similar_docs):
    #     logger.info(f"Top-{i+1} document : {doc.page_content}\n\n")


    # 검색된 문서의 내용을 하나의 문자열로 결합
    context = " ".join([doc[0].page_content for doc in similar_docs if doc[1] > THRESHOLD])

    # 템플릿 설정
    prompt_template = PromptTemplate.from_template(template_text)

    # 템플릿에 값을 채워서 프롬프트를 완성
    filled_prompt = prompt_template.format(context = context, conversation_history= conversations, user_query=query)

    # truncation - context length 넘을 경우 이전 대화 기록부터 삭제
    if config["config"]['quantized_path'] == "ywhwang/llama-3-Korean-Bllossom-8B-awq" and model_tokens(config["config"]['quantized_path'], filled_prompt) >= config["config"]["context_length"]:
        logger.info("Max length exceeded!")
        conversations.pop(0)
        filled_prompt = prompt_template.format(context = context, conversation_history= conversations, user_query=query)
    
    messages = [
        {'role': 'system', 'content': f"{filled_prompt}"},
        {'role': 'user', 'content': f"사용자 질문\n{query}"}
    ]

    chat_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=config["chat_inference"]["temperature"], 
                                     top_p=config["chat_inference"]["top_p"], max_tokens=config["chat_inference"]["max_tokens"])
    
    llm = LLM(model=config["config"]["quantized_path"],
          tokenizer=config["config"]["quantized_path"],
          trust_remote_code = True,
          dtype="float16",
          quantization="AWQ")
    
    with torch.inference_mode():
        output = llm.generate([chat_messages],
                                sampling_params=sampling_params)

    return output[0].outputs[0].text


# GPT 연동
def generate_gpt_response(query, db):
    global conversations
    llm = ChatOpenAI(
        model=config['openai_chat_inference']['model'],
        frequency_penalty=config['openai_chat_inference']['frequency_penalty'],
        logprobs=config['openai_chat_inference']['logprobs'],
        top_logprobs=config['openai_chat_inference']['top_logprobs'],
        max_tokens=config['chat_inference']['max_new_tokens'],  # 최대 토큰수
        temperature=config['chat_inference']['temperature'],  # 창의성 (0.0 ~ 2.0)
    )

    template_text = """
    당신은 한국의 외국인 근로자를 위한 법률 및 비자 전문 AI 어시스턴트입니다. 다음 지침을 따라 응답해 주세요:

    1. 관련 문서의 정보를 바탕으로 정확하고 최신의 법률 및 비자 정보를 제공하세요.
    2. 복잡한 법률 용어나 절차를 쉽게 설명하여 외국인 근로자가 이해하기 쉽게 답변하세요.
    3. 불확실한 정보에 대해서는 명확히 언급하고, 공식 기관에 문의할 것을 권장하세요.
    4. 문화적 차이를 고려하여 정중하고 친절한 태도로 응대하세요.
    5. 필요한 경우 관련 정부 기관이나 지원 센터의 연락처를 제공하세요.
    6. 개인정보 보호를 위해 구체적인 개인 정보를 요구하지 마세요.
    7. 이전 대화 내용을 참고하여 문맥에 맞는 자연스러운 응답을 제공하세요.
    8. 사용자의 이전 질문이나 concerns를 기억하고 연관된 정보를 제공하세요.

    관련 문서: 
    {context}

    이전 대화:
    {conversation_history}

    사용자 질문:
    {user_query}
    """
    
    similar_docs = db.similarity_search_with_relevance_scores(query, k=config['config']['top_k'])
    similar_docs = similar_docs[::-1]

    for i, doc in enumerate(similar_docs):
        print(f"## Top-{i+1} document : {doc}\n\n")

    # 검색된 문서의 내용을 하나의 문자열로 결합
    context = " ".join([doc[0].page_content for doc in similar_docs if doc[1] > THRESHOLD])

    # 템플릿 설정
    prompt_template = PromptTemplate.from_template(template_text)

    # 템플릿에 값을 채워서 프롬프트를 완성
    filled_prompt = prompt_template.format(context = context, conversation_history= conversations, user_query=query)
    
    # truncation - context length 넘을 경우 이전 대화 기록부터 삭제
    if config["openai_chat_inference"]['model'] == "gpt-3.5-turbo-0125" and gpt_tokens(model_name=config["openai_chat_inference"]['model'], string=filled_prompt) >= config["openai_chat_inference"]["context_length"]:
        logger.info("## Max length exceeded!")
        conversations.pop(0)
        filled_prompt = prompt_template.format(context = context, conversation_history= conversations, user_query=query)

    logger.info(f"## total prompt : {filled_prompt}")
    output = llm.invoke(input = filled_prompt)
    
    return output.content

## no rag
def generate_no_rag_gpt_response(query, db):
    global conversations
    llm = ChatOpenAI(
        model=config['openai_chat_inference']['model'],
        frequency_penalty=config['openai_chat_inference']['frequency_penalty'],
        logprobs=config['openai_chat_inference']['logprobs'],
        top_logprobs=config['openai_chat_inference']['top_logprobs'],
        max_tokens=config['chat_inference']['max_new_tokens'],  # 최대 토큰수
        temperature=config['chat_inference']['temperature'],  # 창의성 (0.0 ~ 2.0)
    )

    template_text = """
    당신은 한국의 외국인 근로자를 위한 법률 및 비자 전문 AI 어시스턴트입니다. 다음 지침을 따라 응답해 주세요:

    1. 정확한 최신의 법률 및 비자 정보를 제공하세요.
    2. 복잡한 법률 용어나 절차를 쉽게 설명하여 외국인 근로자가 이해하기 쉽게 답변하세요.
    3. 불확실한 정보에 대해서는 명확히 언급하고, 공식 기관에 문의할 것을 권장하세요.
    4. 문화적 차이를 고려하여 정중하고 친절한 태도로 응대하세요.
    5. 필요한 경우 관련 정부 기관이나 지원 센터의 연락처를 제공하세요.
    6. 개인정보 보호를 위해 구체적인 개인 정보를 요구하지 마세요.
    7. 이전 대화 내용을 참고하여 문맥에 맞는 자연스러운 응답을 제공하세요.
    8. 사용자의 이전 질문이나 concerns를 기억하고 연관된 정보를 제공하세요.

    이전 대화:
    {conversation_history}

    사용자 질문:
    {user_query}
    """
    # 템플릿 설정
    prompt_template = PromptTemplate.from_template(template_text)

    # 템플릿에 값을 채워서 프롬프트를 완성
    filled_prompt = prompt_template.format(conversation_history= conversations, user_query=query)
    
    # truncation - context length 넘을 경우 이전 대화 기록부터 삭제
    if config["openai_chat_inference"]['model'] == "gpt-3.5-turbo-0125" and gpt_tokens(model_name=config["openai_chat_inference"]['model'], string=filled_prompt) >= config["openai_chat_inference"]["context_length"]:
        logger.info("## Max length exceeded!")
        conversations.pop(0)
        filled_prompt = prompt_template.format(conversation_history= conversations, user_query=query)

    logger.info(f"## total prompt : {filled_prompt}")
    output = llm.invoke(input = filled_prompt)
    
    return output.content



@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/get_test/<param>', methods=['GET'])
def get_echo_call(param):
    return jsonify({"param": param})


@app.route('/question', methods=['POST'])
def question():
    global conversations, conversations_length
    if db is None:
        return jsonify({"error": "Database not initialized"}), 500
    try:
        data = request.get_json()
        # logger.info(f"Received data: {data}")  # 받은 데이터 로깅

        # 새로운 쿼리 형식 처리
        conversation = data.get('Conversation', [])
        if not conversation:
            return jsonify({"error": "No conversation data provided"}), 400

        print('## history ##')
        print(conversations)

        # 마지막 human 발화 추출
        user_query = next((item['utterance'] for item in reversed(conversation) if item['speaker'] == 'human'), None)

        logger.info(f"## 사용자 쿼리: {user_query}")

        if user_query == "감사합니다" or user_query == "종료" or user_query == "quit":
            # log를 파일에 출력
            file_handler = logging.FileHandler('./model/log_file.txt')
            logger.addHandler(file_handler)

            print("## log completed!")
            return jsonify({"end": "dialog end"}), 200  # 200은 HTTP 성공 상태 코드입니다

        # AI 응답 생성
        if CONFIG_NAME == "gpt_config.json":
            if user_query is None:
                return jsonify({"error": "No user utterance found"}), 400

            # logger.info(f"Extracted user query: {user_query}")  # 추출된 사용자 쿼리 로깅

            if INCLUDE_RAG == True:
                answer = generate_gpt_response(user_query, db)
            else:
                answer = generate_no_rag_gpt_response(user_query, db)


            # 대화 turn 저장
            conversations.append({"human": user_query})
            conversations.append({"system" : answer})
            conversations_length += gpt_tokens(model_name=config["openai_chat_inference"]['model'], string=answer)
        
        elif CONFIG_NAME == "bllossom_config.json":
            if user_query is None:
                return jsonify({"error": "No user utterance found"}), 400

            # logger.info(f"Extracted user query: {user_query}")  # 추출된 사용자 쿼리 로깅
            answer = generate_bllossom_response(user_query, db)
            # 대화 turn 저장
            conversations.append({"human": user_query})
            conversations.append({"system" : answer})

            conversations_length += model_tokens(config["config"]['quantized_path'], answer)

        # logger.info(f"Generated AI response: {answer}")  # 생성된 AI 응답 로깅

        # AI 응답을 텍스트로 직접 반환
        return answer, 200  # 200은 HTTP 성공 상태 코드입니다

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    sys.exit(0)