from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_elasticsearch import ElasticsearchStore
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


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)


# 대화 기록을 저장할 딕셔너리
conversations = {}


# Load DB once at startup
db = None
try:
    db = ElasticsearchStore(
        es_cloud_id=ES_CLOUD_ID,
        es_api_key=ES_API_KEY,
        index_name='helloworld',
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    )
except Exception as e:
    print(f"Error loading database: {str(e)}")


prompt_question = """
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

    대화 기록:
    {conversation_history}
    """

prompt_resume = """
    회사명: DeepSales
    업무명: Inbound Sales Representatives
    구분: 무역·영업·판매·매장관리
    경력사항: 경력무관
    고용 형태: 인턴
    직업설명: This is a full-time on-site role as an Inbound Sales Representative at DeepSales in Daechi-dong. The Inbound Sales Representative will be responsible for inside sales, communication, customer service, order processing, and account management tasks on a daily basis. 
    직업요구사항: Inside Sales and Account Management skills
    Strong Communication and Customer Service abilities
    Experience in Order Processing
    Excellent interpersonal and problem-solving skills
    Knowledge of sales processes and CRM software
    Ability to work well in a team setting
    Previous experience in a sales or customer-facing role
    Bachelor& #39;s degree in Business Administration or related field
    지원지침: {Instructions}
    추가정보:Welcome to DeepSales, a leader in sales intelligence and data innovation, established in 2021 in Seoul& #39;s Gangnam-gu district. DeepSales empowers sales managers and businesses globally with advanced platform leveraging deep learning technologies for actionable insights. Our goal is to maximize sales potential through intelligent data-driven insights and efficiency in sales processes.
"""




#GPT 연동
def generate_ai_response(conversation_history = None,prompt = None,query = None,db = None):

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125",
                        temperature=0,  # 창의성 (0.0 ~ 2.0)
                        max_tokens=4096,  # 최대 토큰수
                        openai_api_key=OPENAI_KEY)

    similar_docs = db.similarity_search(query, k=3)

    # 검색된 문서의 내용을 하나의 문자열로 결합
    context = " ".join([doc.page_content for doc in similar_docs])

    # 템플릿 설정
    prompt_template = PromptTemplate.from_template(prompt)

    # 템플릿에 값을 채워서 프롬프트를 완성
    filled_prompt = prompt_template.format(context = context, conversation_history= conversation_history)
    
    output = llm.invoke(input = filled_prompt)
    
    return output.content

#GPT 연동
def generate_ai_resume(prompt = None,query = None,db = None):

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125",
                        temperature=0,  # 창의성 (0.0 ~ 2.0)
                        max_tokens=4096,  # 최대 토큰수
                        openai_api_key=OPENAI_KEY)

    # similar_docs = db.similarity_search(query, k=3)

    # # 검색된 문서의 내용을 하나의 문자열로 결합
    # context = " ".join([doc.page_content for doc in similar_docs])

    # 템플릿 설정
    prompt_template = PromptTemplate.from_template(prompt)

    # 템플릿에 값을 채워서 프롬프트를 완성
    #filled_prompt = prompt_template.format(context = context, conversation_history= conversation_history)
    
    output = llm.invoke(input = prompt_template)
    
    return output.content

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/get_test/<param>', methods=['GET'])
def get_echo_call(param):
    return jsonify({"param": param})


@app.route('/question', methods=['POST'])
def question():
    if db is None:
        return jsonify({"error": "Database not initialized"}), 500
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")  # 받은 데이터 로깅

        # 새로운 쿼리 형식 처리
        conversation = data.get('Conversation', [])
        if not conversation:
            return jsonify({"error": "No conversation data provided"}), 400

        # 마지막 human 발화 추출
        user_query = next((item['utterance'] for item in reversed(conversation) if item['speaker'] == 'human'), None)
        if user_query is None:
            return jsonify({"error": "No user utterance found"}), 400

        logger.info(f"Extracted user query: {user_query}")  # 추출된 사용자 쿼리 로깅

        # AI 응답 생성
        answer = generate_ai_response(conversation,prompt_question, user_query, db)
        logger.info(f"Generated AI response: {answer}")  # 생성된 AI 응답 로깅

        # AI 응답을 텍스트로 직접 반환
        return answer, 200  # 200은 HTTP 성공 상태 코드입니다

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    



@app.route('/resume', methods=['POST'])
def resume():
    
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")  # 받은 데이터 로깅

        # AI 응답 생성
        answer = generate_ai_response(prompt = prompt_question, db = db)
        logger.info(f"Generated AI response: {answer}")  # 생성된 AI 응답 로깅

        # AI 응답을 텍스트로 직접 반환
        return answer, 200  # 200은 HTTP 성공 상태 코드입니다

    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
