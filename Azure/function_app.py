import azure.functions as func
import logging
import json
import sys,os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from model import ChatModel
import certifi

app = func.FunctionApp()

load_dotenv(verbose=True)
logging.basicConfig(filename='function_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting application initialization...")

with open(f'configs/config.json', 'r') as f:
    config = json.load(f)

logging.info(f"Config file loaded")

os.environ["MONGODB_ATLAS_CLUSTER_URI"] = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

print(certifi.where())
# Load DB once at startup
try:
    logging.info("Starting database initialization...")
    # client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"],
    #                      ssl=True)
    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"],
                         tls=True,
                        tlsCAFile=certifi.where())
    
    # 두 컬렉션 모두 pymongo Collection 객체로 초기화
    MONGODB_COLLECTION = client[config['path']['db_name']][config['path']['collection_name']]
    TEST_COLLECTION = client[config['path']['db_name']][config['path']['test_collection_name']]
    
    chat_model = ChatModel(config)

    logging.info("Database & Model initialized successfully.")

except Exception as e:
    logging.error(f"Error loading database: {str(e)}")


# 사용자 요청 수신
@app.route(route="question", auth_level=func.AuthLevel.ANONYMOUS)
def question(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Question function triggered.")
    
    try:
        req_body = req.get_json()
        conversation = req_body.get('Conversation', [])
        
        
        if not conversation:
            return func.HttpResponse("No conversation data provided", status_code=400)

        user_query = next((item['utterance'] for item in reversed(conversation) 
                         if item['speaker'] == 'human'), None)
                         
        if user_query is None:
            return func.HttpResponse("No user utterance found", status_code=400)

        logging.info(f"Extracted user query: {user_query}")

        # 첫 번째 쿼리인지 확인
        is_first_query = len([item for item in conversation if item['speaker'] == 'human']) == 1

        if is_first_query:
            response = chat_model.generate_ai_response_first_query(conversation_history="",
                                                                query=user_query,
                                                                collection=TEST_COLLECTION)
        else:
            response = chat_model.generate_ai_response(conversation, user_query, MONGODB_COLLECTION)

        # 응답에서 references 제외하고 answer만 반환
        return func.HttpResponse(
            json.dumps({
                "answer": response["answer"]
            }, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)

# for test
@app.route(route="get_test/{param}", auth_level=func.AuthLevel.ANONYMOUS)
def get_echo_call(req: func.HttpRequest) -> func.HttpResponse:
    param = req.route_params.get('param')
    return func.HttpResponse(json.dumps({"param": param}), mimetype="application/json")