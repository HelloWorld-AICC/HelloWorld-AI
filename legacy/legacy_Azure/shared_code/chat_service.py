import logging
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from pymongo import MongoClient

# 로깅 설정
logger = logging.getLogger(__name__)

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

class ChatService:
    """
    핵심 AI 채팅 서비스를 구현한 클래스입니다.
    """

    def __init__(self, config):
        self.config = config
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
        self.initialize_db()
        self.initialize_prompts()

    # DB 초기화
    def initialize_db(self):
        """
        MongoDB 데이터베이스 연결을 초기화하는 함수입니다.
        환경 변수에서 MongoDB URI를 가져와 연결하고, 컬렉션을 초기화합니다.
        """

        try:
            logger.info("Starting database initialization...")
            client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"], ssl=True)
            self.main_collection = client[self.config['path']['db_name']]['foreigner_legalQA_v2']
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            raise

    # 프롬프트 불러오기
    def initialize_prompts(self):
        """
        AI 응답 생성에 사용될 프롬프트 템플릿을 초기화하는 함수입니다.
        """

        self.CHAT_PROMPT_TEMPLATE = PromptTemplate.from_template("""
        당신은 한국의 외국인 근로자를 위한 법률 및 비자 전문 AI 어시스턴트입니다.

        참고 문서:
        {context}

        최근 대화 기록:
        {conversation_history}

        답변 시 주의사항:
        1. 구체적이고 실용적인 해결방안을 제시해주세요
        2. 이전 답변을 반복하지 마세요
        3. 친절하고 이해하기 쉬운 말로 설명해주세요
        """)

    # MongoDB Vector search 유사 문서 검색 로직
    def _perform_vector_search(self, query_embedding):
        """
        유저 쿼리와 유사한 문서를 검색하는 함수입니다.
        
        Args:
            query_embedding (list): 사용자 질문의 벡터 임베딩 값
        
        Returns:
            list: 유사도가 높은 상위 3개 문서 목록을 반환합니다. 
                각 문서는 제목, 내용, URL, 유사도 점수를 포함한다.
        
        """

        # legal_QA 컬렉션용 Vector Search 쿼리
        results = self.main_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "Embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 3
                }
            },
            {
                "$project": {
                    "title": 1, #1은 포함한다는 의미
                    "contents": 1,
                    "url": 1,
                    "score": { "$meta": "vectorSearchScore" },
                    "_id": 0
                }
            }
        ])

        return list(results)

    # Context 생성 로직
    def _build_context(self, results_list):
        """
        검색된 유사 문서들을 AI 응답 생성에 사용할 컨텍스트로 변환하는 함수입니다.

        Args:
            results_list (list): 유사 문서 목록
        
        Returns:
            str: 포맷팅된 컨텍스트 문자열
                유사 문서가 없을 경우 기본 안내 정보 변환
        """

        if not results_list:
            total_context = "일반적인 안내 정보..."
        else:
            total_context = ""
            for idx, result in enumerate(results_list,1):
                context = f"""
                관련 사례{idx} (출처: {result.get('url', 'N/A')}):
                제목: {result.get('title', 'N/A')}
                내용: {result.get('contents', 'N/A')}
                """
                logger.info(f"\n\n{context}\n\n")
                total_context += context

        return total_context

    # Context와 이전 대화 쌍으로 프롬프트 포맷팅하는 로직
    def _format_conversation(self, conversation_history):
        """
        이전 대화 기록을 AI 응답 생성에 적합한 형식으로 변환하는 함수입니다.

        Args: conversation_history (list): 대화 기록 목록
            [{"speaker": "human/ai", "utterance": "대화내용"}, ...]

        
        Returns:
            str: 포맷팅된 대화 기록 문자열
                설정된 최대 대화 쌍 수 만큼만 포함
        """

        # 이전 대화 쌍 불러오기
        max_pairs = self.config['chat_inference']['max_conversation_pairs']
        max_messages = max_pairs*2

        # conversation 텍스트 생성
        formatted_conversation = "\n".join([
            f"{'사용자' if msg['speaker'] == 'human' else 'AI'}: {msg['utterance']}"
            for msg in conversation_history[-max_messages:]
        ])

        return formatted_conversation


    # LLM 응답 생성 로직
    def _get_llm_response(self, prompt):
        """
        OpenAI GPT 모델을 사용하여 응답을 생성하는 함수입니다.

        Args: 
            prompt (str): 컨텍스트와 대화 기록이 포함된 완성된 프롬프트

        Returns:
            dict: {"answer": "생성된 응답 텍스트"}
        """
        
        llm = ChatOpenAI(
            model= self.config['openai_chat_inference']['model'],
            temperature= self.config['chat_inference']['temperature']
        )
        output = llm.invoke(input=prompt)
        
        return {
            "answer": output.content
        }


    # 메인 함수
    def generate_ai_response(self, conversation_history, query):
        """
        사용자 질문에 대한 AI 응답을 생성하는 메인 함수입니다.

        Process:
            1. 사용자 질문을 벡터로 변환
            2. 유사한 법률 문서 검색
            3. 컨텍스트 구성
            4. 이전 대화 기록 포맷팅
            5. AI 응답 생성

        Args:
            conversation_history (list): 이전 대화 기록 목록
            query (str): 사용자의 현재 쿼리

        Returns:
            dict: {"answer": "AI가 생성한 응답"}
        """

        try:
            logger.info("====== AI 응답 생성 시작 ======")
            logger.info(f"입력된 쿼리: {query}")

            # 임베딩 모델 초기화 및 쿼리 임베딩
            logger.info("1. 임베딩 모델 초기화 및 쿼리 벡터화 시작...")
            embedding_model = OpenAIEmbeddings(model=self.config['openai_chat_inference']['embedding'])
            query_embedding = embedding_model.embed_query(query)
            logger.info("   쿼리 벡터화 완료")
            
            # 유사 문서 리스트 초기화
            logger.info("2. MongoDB에서 유사 문서 검색 시작...")
            results_list = self._perform_vector_search(query_embedding)
            logger.info(f"   검색된 유사 문서 수: {len(results_list)}개")
            
            # context 초기화
            logger.info("3. 검색 결과로 컨텍스트 구성 중...")
            context = self._build_context(results_list)
            logger.info("   컨텍스트 구성 완료")
            
            # 이전 대화 포멧 초기화
            logger.info("4. 이전 대화 기록 포맷팅 중...")
            formatted_conversation = self._format_conversation(conversation_history)
            logger.info("   대화 기록 포맷팅 완료")

            # 입력 프롬프트 초기화
            logger.info("5. 최종 프롬프트 구성 중...")
            filled_prompt = self.CHAT_PROMPT_TEMPLATE.format(
                context=context,
                conversation_history=formatted_conversation
            )
            logger.info("   프롬프트 구성 완료")
            
            # llm 응답 반환
            logger.info("6. GPT 모델에 요청하여 응답 생성 중...")
            response = self._get_llm_response(filled_prompt)
            logger.info("   응답 생성 완료")
            logger.info("====== AI 응답 생성 완료 ======\n")

            return response
            
            
        except Exception as e:
            logging.error(f"Error in generate_ai_response: {str(e)}")
            raise