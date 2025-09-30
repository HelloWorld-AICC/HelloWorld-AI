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

    # 키워드 기반 검색 함수
    def keyword_search(self, keywords, collection, top_k):
        """
        키워드 리스트를 받아서 최소 2개 이상의 키워드가 포함된 문서를 검색하고 점수를 계산
        """
        if not keywords or len(keywords) == 0:
            return []

        # 실제 컬렉션 필드명(`title`, `contents`)을 대상으로 정규식 OR 검색
        # 키워드가 1개이든 여러 개이든 우선 후보군은 OR 매칭으로 넓게 가져온 뒤, 아래에서 최소 2개 이상 매칭으로 필터링
        pattern = keywords[0] if len(keywords) == 1 else "|".join(keywords)
        keyword_query = {
            "$or": [
                {"title": {"$regex": pattern, "$options": "i"}},
                {"contents": {"$regex": pattern, "$options": "i"}},
            ]
        }

        # 키워드 검색 수행
        keyword_results = list(
            collection.find(
                keyword_query,
                {
                    "title": 1,
                    "contents": 1,
                    "url": 1,
                    "_id": 1,
                },
            )
        )

        # 각 문서에 대해 키워드 매칭 점수 계산 및 필터링
        scored_docs = []
        for doc in keyword_results:
            score = 0
            # 검색 대상 텍스트 결합
            title_value = doc.get("title", "") or ""
            contents_value = doc.get("contents", "") or ""
            text_content = f"{title_value} {contents_value}".lower()
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
                        "title": title_value,
                        "contents": contents_value,
                        "url": doc.get("url", ""),
                        "keyword_score": score,
                        "matched_keywords": matched_keywords,
                        "matched_count": len(matched_keywords),
                    }
                )

        # 키워드 점수로 내림차순 정렬
        scored_docs.sort(key=lambda x: x["keyword_score"], reverse=True)

        # 로깅: 후보/필터 통과/리턴 수
        try:
            logging.info(
                f"키워드 후보 {len(keyword_results)}개, 필터 통과 {len(scored_docs)}개, 반환 {min(len(scored_docs), top_k)}개"
            )
        except Exception:
            pass

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
            # 숫자 로그 (키워드 단계 원본/중복제거)
            logging.info(f"keyword_results_count_raw: {len(keyword_results)}")
            logging.info(f"keyword_results_count_deduped: {len(all_results)}")

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
            # 숫자 로그 (ANN 2단계 원본)
            logging.info(f"ann_results_count_raw: {len(ann_results_list)}")

            # ANN 결과를 키워드 결과와 동일한 형식으로 변환 (중복 제거)
            for doc in ann_results_list:
                if doc["_id"] not in seen_ids and len(all_results) < top_k:
                    all_results.append(
                        {
                            "_id": doc["_id"],
                            "title": doc.get("title", ""),
                            "contents": doc.get("contents", ""),
                            "url": doc.get("url", ""),
                            "vector_score": doc.get("score", 0),
                            "search_type": "vector",
                        }
                    )
                    seen_ids.add(doc["_id"])

        # 3. 최종 결과 반환 (상위 top_k개)
        final_results = all_results[:top_k]
        logging.info(f"final_results_count: {len(final_results)}")
        return final_results

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
