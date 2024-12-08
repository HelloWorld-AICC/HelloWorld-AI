import logging
import json
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

class ChatModel:
    def __init__(self, config):
        self.config = config
        # prompt 불러오기
        try:
            with open(f'../model/templates/chat_template.txt', 'r', encoding='utf-8') as file:
                self.chat_template = file.read()

        except Exception as e:
            logging.error(f"Error loading prompt template: {str(e)}")
        
        self.llm = ChatOpenAI(
            model=config['chat_config']['model'],
            # temperature=config['chat_inference']['temperature']
        )


    # 응답 생성 함수
    def generate_ai_response(self, conversation_history, query, collection):
        try:
            logging.info(f"Generating embedding for query: {query}")
            embedding_model = OpenAIEmbeddings(model=self.config['embedding_model'])
            query_embedding = embedding_model.embed_query(query)

            # legal_QA 컬렉션용 Vector Search 쿼리
            results = collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": self.config['chat_config']['numCandidates'],
                        "limit": config['chat_config']['top_k']
                    }
                },
                {
                    "$project": {
                        "text": 1,           # text 필드만 프로젝션
                        "source": 1,         # 출처 정보도 포함
                        "score": { "$meta": "vectorSearchScore" },
                        "_id": 0
                    }
                }
            ])

            results_list = list(results)
            logging.info("\n=== 유사 문서 검색 결과 ===")
            
            if not results_list:
                logging.info("유사한 문서를 찾을 수 없습니다.")
                context = "일반적인 안내 정보..."
            else:
                context = ""
                for idx, result in enumerate(results_list, 1):
                    # 유사 문서 로깅
                    logging.info(f"\n[유사 문서 {idx}]")
                    logging.info(f"유사도 점수: {result.get('score', 'N/A')}")
                    logging.info(f"출처: {result.get('source', 'N/A')}")
                    logging.info(f"내용: {result.get('text', 'N/A')[:200]}...")  # 앞부분만 로깅

                    # 컨텍스트에 추가
                    context += f"""
                    관련 사례 {idx} (출처: {result.get('source', 'N/A')}):
                    {result.get('text', '')}
                    """

            # config에서 대화 수 가져와서 사용
            max_messages = self.config['chat_inference']['prev_turns']
            
            formatted_conversation = "\n".join([
                f"{'사용자' if msg['speaker'] == 'human' else 'AI'}: {msg['utterance']}"
                for msg in conversation_history[-max_messages:]
            ])

            # 전역 프롬프트 템플릿 사용
            filled_prompt = self.chat_template.format(
                context=context,
                conversation_history=formatted_conversation,
                query=query
            )
            
            output = self.llm.invoke(input=filled_prompt)
            
            return {
                "answer": output.content
            }

        except Exception as e:
            logging.error(f"Error in generate_ai_response: {str(e)}")
            raise ValueError(f"Error in generate_ai_response: {str(e)}")


        
    def generate_ai_response_first_query(self, conversation_history, query, collection):
        try:
            logging.info(f"Generating embedding for query: {query}")
            embedding_model = OpenAIEmbeddings(model=self.config['embedding_model'])
            query_embedding = embedding_model.embed_query(query)

            # MongoDB Vector Search 쿼리
            results = collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "내담자_정보.Embedding",
                        "queryVector": query_embedding,
                        "exact": True,
                        "limit": self.config['chat_config']['top_k']
                    }
                },
                {
                    "$project": {
                        "내담자_정보": 1,
                        "해결방법": 1,
                        "score": { "$meta": "vectorSearchScore" },
                        "_id": 0
                    }
                }
            ])

            # 결과 처리 및 컨텍스트 구성
            results_list = list(results)
            logging.info("\n=== 유사 문서 검색 결과 ===")
            
            if not results_list:
                logging.info("유사한 문서를 찾을 수 없습니다.")
            
            context = ""
            for idx, result in enumerate(results_list, 1):
                # 유사 문서 로깅
                logging.info(f"\n[유사 문서 {idx}]")
                logging.info(f"유사도 점수: {result.get('score', 'N/A')}")
                info = result['내담자_정보']
                logging.info(f"거주지역: {info.get('거주지역', 'N/A')}")
                logging.info(f"국적: {info.get('국적', 'N/A')}")
                logging.info(f"체류자격: {info.get('체류자격', 'N/A')}")
                logging.info(f"추가정보: {info.get('추가정보', 'N/A')}")
                logging.info(f"해결방법: {result.get('해결방법', 'N/A')[:100]}...")  # 해결방법은 앞부분만

                # 컨텍스트 구성
                context += f"""
                    사례 {idx}:
                    - 거주지역: {info.get('거주지역', 'N/A')}
                    - 국적: {info.get('국적', 'N/A')}
                    - 체류자격: {info.get('체류자격', 'N/A')}
                    - 추가정보: {info.get('추가정보', 'N/A')}
                    - 해결방법: {result.get('해결방법', 'N/A')}

                    """

            # 전역 프롬프트 템플릿 사용
            filled_prompt = self.chat_template.format(
                context=context,
                conversation_history="",
                query=query
            )
            
            output = self.llm.invoke(input=filled_prompt)
            
            return {
                "answer": output.content
            }

        except Exception as e:
            logging.error(f"Error in generate_ai_response_first_query: {str(e)}")
            raise ValueError()

