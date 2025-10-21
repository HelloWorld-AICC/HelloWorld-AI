import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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

    # 하이브리드 검색 함수 (mongo_query 파이프라인 + ANN)
    def hybrid_search(self, query, mongo_query, collection, top_k):
        """
        1단계: MongoDB 파이프라인(`mongo_query`) 기반 텍스트 검색
        2단계: ANN(Vector) 검색으로 부족분 보충 (중복 제거)
        """
        all_results = []
        seen_ids = set()

        # 1단계: mongo_query 파이프라인 검색
        try:
            if mongo_query and isinstance(mongo_query, list) and len(mongo_query) > 0:
                logging.info("Mongo 파이프라인 검색 수행")
                stage1_cursor = collection.aggregate(mongo_query)
                stage1_list = list(stage1_cursor)

                for doc in stage1_list:
                    doc_id = doc.get("_id")
                    if doc_id not in seen_ids:
                        all_results.append(
                            {
                                "_id": doc_id,
                                "title": doc.get("title", ""),
                                "contents": doc.get("contents", ""),
                                "url": doc.get("url", ""),
                                # generate_ai_response에서 키워드/텍스트 검색 판별에 사용
                                "keyword_score": doc.get("score", 0),
                                "highlights": doc.get("highlights", []),
                                "search_type": "mongo",
                            }
                        )
                        seen_ids.add(doc_id)

                if len(stage1_list) == 0:
                    logging.info(
                        "Mongo 파이프라인 검색 결과가 없습니다. ANN 단계로 진행합니다."
                    )
                else:
                    logging.info(
                        f"Mongo 파이프라인 검색 결과: {len(stage1_list)}개 (중복 제거 후: {len(all_results)}개)"
                    )
                # 숫자 로그 (Mongo 1단계 원본/중복제거)
                logging.info(f"mongo_results_count_raw: {len(stage1_list)}")
                logging.info(f"mongo_results_count_deduped: {len(all_results)}")
            else:
                logging.info("유효한 mongo_query가 없어 ANN 단계로 진행합니다.")
        except Exception as e:
            logging.error(
                f"Mongo 파이프라인 검색 중 오류: {str(e)}. ANN 단계로 진행합니다."
            )

        # 2단계: ANN 검색으로 부족한 만큼 추가
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
                            # 중복 제거를 고려해 약간 더 많이 가져온 뒤 상한을 적용
                            "limit": max(remaining_k * 2, remaining_k),
                        }
                    },
                    {
                        "$project": {
                            "title": 1,
                            "contents": 1,
                            "url": 1,
                            "score": {"$meta": "vectorSearchScore"},
                            "_id": 1,
                        }
                    },
                ]
            )

            ann_results_list = list(ann_results)

            # 숫자 로그 (ANN 2단계 원본/중복제거 적용 전)
            logging.info(f"ann_results_count_raw: {len(ann_results_list)}")

            for doc in ann_results_list:
                doc_id = doc.get("_id")
                if doc_id not in seen_ids and len(all_results) < top_k:
                    all_results.append(
                        {
                            "_id": doc_id,
                            "title": doc.get("title", ""),
                            "contents": doc.get("contents", ""),
                            "url": doc.get("url", ""),
                            "vector_score": doc.get("score", 0),
                            "search_type": "vector",
                        }
                    )
                    seen_ids.add(doc_id)

        # 3단계: 최종 결과 반환 (상위 top_k)
        final_results = all_results[:top_k]
        # 숫자 로그 (최종 개수)
        logging.info(f"final_results_count: {len(final_results)}")
        return final_results

    # 응답 생성 함수 (키워드 지원)
    def generate_ai_response(
        self, conversation_history, query, collection, mongo_query
    ):
        try:
            logging.info(f"Generating response for query: {query}")
            if mongo_query:
                logging.info(f"Using query: {mongo_query}")

            # 하이브리드 검색 수행 (키워드 + ANN)
            top_k = self.config["chat_config"]["top_k"]
            results_list = self.hybrid_search(query, mongo_query, collection, top_k)

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
                    logging.info(f"출처(URL): {result.get('url', 'N/A')}")
                    content_snippet = (
                        result.get("contents")
                        or result.get("title")
                        or result.get("text")
                        or ""
                    )
                    logging.info(f"내용: {content_snippet[:100]}...")  # 앞부분만 로깅

                    # 컨텍스트에 추가
                    search_type = "키워드" if "keyword_score" in result else "벡터"
                    context += f"""
                    관련 사례 {idx} ({search_type} 검색, 출처: {result.get('url', 'N/A')}):
                    {(result.get('contents') or result.get('title') or '')}
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
