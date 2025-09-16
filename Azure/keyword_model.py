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
            with open(
                f"../model/templates/chat_template.txt", "r", encoding="utf-8"
            ) as file:
                self.chat_template = file.read()

        except Exception as e:
            logging.error(f"Error loading prompt template: {str(e)}")

        self.llm = ChatOpenAI(
            model=config["chat_config"]["model"],
            # temperature=config['chat_config']['temperature']
        )

    # 키워드 기반 검색 함수
    def keyword_search(self, keywords, collection, top_k):
        """
        키워드 리스트를 받아서 최소 2개 이상의 키워드가 포함된 문서를 검색하고 점수를 계산
        """
        if not keywords or len(keywords) == 0:
            return []

        # 키워드가 1개뿐이면 OR 조건으로 검색
        if len(keywords) == 1:
            keyword_query = {"text": {"$regex": keywords[0], "$options": "i"}}
        else:
            # 키워드가 2개 이상이면 최소 2개 이상 포함 조건으로 검색
            keyword_query = {"text": {"$regex": "|".join(keywords), "$options": "i"}}

        # 키워드 검색 수행
        keyword_results = list(
            collection.find(keyword_query, {"text": 1, "source": 1, "_id": 1})
        )

        # 각 문서에 대해 키워드 매칭 점수 계산 및 필터링
        scored_docs = []
        for doc in keyword_results:
            score = 0
            text_content = doc.get("text", "").lower()
            matched_keywords = []

            # 각 키워드가 포함된 횟수만큼 점수 추가
            for keyword in keywords:
                keyword_count = text_content.count(keyword.lower())
                if keyword_count > 0:
                    score += keyword_count
                    matched_keywords.append(keyword)

            # 최소 2개 이상의 키워드가 매칭된 경우만 포함
            if len(matched_keywords) >= 2:
                scored_docs.append(
                    {
                        "_id": doc["_id"],
                        "text": doc.get("text", ""),
                        "source": doc.get("source", ""),
                        "keyword_score": score,
                        "matched_keywords": matched_keywords,
                        "matched_count": len(matched_keywords),
                    }
                )

        # 키워드 점수로 내림차순 정렬
        scored_docs.sort(key=lambda x: x["keyword_score"], reverse=True)

        return scored_docs[:top_k]

    # 하이브리드 검색 함수 (키워드 + ANN)
    def hybrid_search(self, query, keywords, collection, top_k):
        """
        키워드 검색과 ANN 검색을 결합한 하이브리드 검색
        """
        all_results = []
        seen_ids = set()

        # 1. 키워드 검색 수행
        if keywords and len(keywords) > 0:
            keyword_results = self.keyword_search(keywords, collection, top_k)

            # 키워드 결과 추가 (중복 제거)
            for doc in keyword_results:
                if doc["_id"] not in seen_ids:
                    all_results.append(doc)
                    seen_ids.add(doc["_id"])

            logging.info(
                f"키워드 검색 결과: {len(keyword_results)}개 (중복 제거 후: {len(all_results)}개)"
            )

        # 2. ANN 검색으로 부족한 만큼 추가
        remaining_k = top_k - len(all_results)
        if remaining_k > 0:
            logging.info(f"ANN 검색으로 추가 {remaining_k}개 검색")

            # ANN 검색 수행
            embedding_model = OpenAIEmbeddings(
                model=self.config["data_config"]["embedding_model"]
            )
            query_embedding = embedding_model.embed_query(query)

            ann_results = collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "Embedding",
                            "queryVector": query_embedding,
                            "numCandidates": self.config["chat_config"][
                                "numCandidates"
                            ],
                            "limit": remaining_k
                            * 2,  # 중복 제거를 위해 더 많이 가져오기
                        }
                    },
                    {
                        "$project": {
                            "text": 1,
                            "source": 1,
                            "score": {"$meta": "vectorSearchScore"},
                            "_id": 1,
                        }
                    },
                ]
            )

            ann_results_list = list(ann_results)

            # ANN 결과를 키워드 결과와 동일한 형식으로 변환 (중복 제거)
            for doc in ann_results_list:
                if doc["_id"] not in seen_ids and len(all_results) < top_k:
                    all_results.append(
                        {
                            "_id": doc["_id"],
                            "text": doc.get("text", ""),
                            "source": doc.get("source", ""),
                            "vector_score": doc.get("score", 0),
                            "search_type": "vector",
                        }
                    )
                    seen_ids.add(doc["_id"])

        # 3. 최종 결과 반환 (상위 top_k개)
        return all_results[:top_k]

    # 응답 생성 함수 (키워드 지원)
    def generate_ai_response(
        self, conversation_history, query, collection, keywords=None
    ):
        try:
            logging.info(f"Generating response for query: {query}")
            if keywords:
                logging.info(f"Using keywords: {keywords}")

            # 하이브리드 검색 수행 (키워드 + ANN)
            top_k = self.config["chat_config"]["top_k"]
            results_list = self.hybrid_search(query, keywords, collection, top_k)

            logging.info(f"\n=== 하이브리드 검색 결과 (총 {len(results_list)}개) ===")

            # 검색된 문서 ID들 추출
            retrieved_doc_ids = [str(doc["_id"]) for doc in results_list]

            if not results_list:
                logging.info("검색된 문서가 없습니다.")
                context = "일반적인 안내 정보..."
            else:
                context = ""
                for idx, result in enumerate(results_list, 1):
                    # 검색 결과 로깅
                    logging.info(f"[검색 결과 {idx}]")
                    logging.info(f"문서 ID: {result.get('_id', 'N/A')}")

                    # 검색 타입에 따라 다른 점수 표시
                    if "keyword_score" in result:
                        logging.info(
                            f"키워드 점수: {result.get('keyword_score', 'N/A')}"
                        )
                        logging.info(
                            f"매칭된 키워드: {result.get('matched_keywords', [])}"
                        )
                        logging.info(
                            f"매칭된 키워드 수: {result.get('matched_count', len(result.get('matched_keywords', [])))}"
                        )
                    elif "vector_score" in result:
                        logging.info(f"벡터 점수: {result.get('vector_score', 'N/A')}")

                    logging.info(f"출처: {result.get('source', 'N/A')}")
                    logging.info(
                        f"내용: {result.get('text', 'N/A')[:100]}..."
                    )  # 앞부분만 로깅

                    # 컨텍스트에 추가
                    search_type = "키워드" if "keyword_score" in result else "벡터"
                    context += f"""
                    관련 사례 {idx} ({search_type} 검색, 출처: {result.get('source', 'N/A')}):
                    {result.get('text', '')}
                    """

            # config에서 대화 수 가져와서 사용
            max_messages = self.config["chat_config"]["prev_turns"]

            formatted_conversation = "\n".join(
                [
                    f"{'사용자' if msg['speaker'] == 'human' else 'AI'}: {msg['utterance']}"
                    for msg in conversation_history[-max_messages:]
                ]
            )

            # 전역 프롬프트 템플릿 사용
            filled_prompt = self.chat_template.format(
                context=context,
                conversation_history=formatted_conversation,
                query=query,
            )

            output = self.llm.invoke(input=filled_prompt)

            return {
                "answer": output.content,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_docs": results_list,
            }

        except Exception as e:
            logging.error(f"Error in generate_ai_response: {str(e)}")
            raise ValueError(f"Error in generate_ai_response: {str(e)}")

    def generate_ai_response_first_query(self, conversation_history, query, collection):
        try:
            logging.info(f"Generating embedding for query: {query}")
            embedding_model = OpenAIEmbeddings(
                model=self.config["data_config"]["embedding_model"]
            )
            query_embedding = embedding_model.embed_query(query)

            # MongoDB Vector Search 쿼리
            results = collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "내담자_정보.Embedding",
                            "queryVector": query_embedding,
                            "exact": True,
                            "limit": self.config["chat_config"]["top_k"],
                        }
                    },
                    {
                        "$project": {
                            "내담자_정보": 1,
                            "해결방법": 1,
                            "score": {"$meta": "vectorSearchScore"},
                            "_id": 0,
                        }
                    },
                ]
            )

            # 결과 처리 및 컨텍스트 구성
            results_list = list(results)
            logging.info("=== 유사 문서 검색 결과 ===")

            if not results_list:
                logging.info("유사한 문서를 찾을 수 없습니다.")

            context = ""
            for idx, result in enumerate(results_list, 1):
                # 유사 문서 로깅
                logging.info(f"[유사 문서 {idx}]")
                logging.info(f"유사도 점수: {result.get('score', 'N/A')}")
                info = result["내담자_정보"]
                logging.info(f"거주지역: {info.get('거주지역', 'N/A')}")
                logging.info(f"국적: {info.get('국적', 'N/A')}")
                logging.info(f"체류자격: {info.get('체류자격', 'N/A')}")
                logging.info(f"추가정보: {info.get('추가정보', 'N/A')}")
                logging.info(f"해결방법: {result.get('해결방법', 'N/A')}")

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
                context=context, conversation_history="", query=query
            )

            output = self.llm.invoke(input=filled_prompt)

            return {"answer": output.content}

        except Exception as e:
            logging.error(f"Error in generate_ai_response_first_query: {str(e)}")
            raise ValueError()

    # 키워드 검색 테스트 함수
    def test_keyword_search(self, keywords, collection, top_k=5):
        """
        키워드 검색 기능을 테스트하는 함수
        """
        print(f"=== 키워드 검색 테스트 ===")
        print(f"키워드: {keywords}")
        print(f"Top-k: {top_k}")

        results = self.keyword_search(keywords, collection, top_k)

        print(f"\n검색 결과: {len(results)}개")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] 문서 ID: {doc['_id']}")
            print(f"키워드 점수: {doc['keyword_score']}")
            print(f"매칭된 키워드: {doc['matched_keywords']}")
            print(
                f"매칭된 키워드 수: {doc.get('matched_count', len(doc['matched_keywords']))}"
            )
            print(f"출처: {doc['source']}")
            print(f"내용: {doc['text'][:100]}...")

        return results

    # 하이브리드 검색 테스트 함수
    def test_hybrid_search(self, query, keywords, collection, top_k=5):
        """
        하이브리드 검색 기능을 테스트하는 함수
        """
        print(f"=== 하이브리드 검색 테스트 ===")
        print(f"쿼리: {query}")
        print(f"키워드: {keywords}")
        print(f"Top-k: {top_k}")

        results = self.hybrid_search(query, keywords, collection, top_k)

        print(f"\n검색 결과: {len(results)}개")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] 문서 ID: {doc['_id']}")
            if "keyword_score" in doc:
                print(f"검색 타입: 키워드")
                print(f"키워드 점수: {doc['keyword_score']}")
                print(f"매칭된 키워드: {doc['matched_keywords']}")
                print(
                    f"매칭된 키워드 수: {doc.get('matched_count', len(doc['matched_keywords']))}"
                )
            elif "vector_score" in doc:
                print(f"검색 타입: 벡터")
                print(f"벡터 점수: {doc['vector_score']:.4f}")
            print(f"출처: {doc['source']}")
            print(f"내용: {doc['text'][:100]}...")

        return results
