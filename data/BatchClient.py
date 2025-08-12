"""OpenAI Batch API를 이용해 임베딩과 생성 작업을 대량 처리하기 위한 유틸리티.

- BatchClient: 내부 도메인 스키마(예: `상담제목`, `상담내용요약`) 기반 배치 구성
- BatchClientExternal: 외부 수집 스키마(예: `title`, `contents`) 기반 배치 구성

주요 기능: JSONL 생성, 파일 업로드, 배치 생성/조회/결과 다운로드.
환경 요구: 환경변수 `OPENAI_KEY` 필요.
"""

from dotenv import load_dotenv
import sys, os
import json
import pandas as pd

import re
from tqdm import tqdm
import ast
import copy

from openai import OpenAI


class BatchClient:
    """내부 도메인 데이터프레임용 배치 클라이언트.

    열 표준화, JSONL 직렬화, Batch 생성/조회/결과 다운로드를 제공합니다.

    Attributes:
        config (dict): 설정 딕셔너리.
        embedding_columns (list[str]): 임베딩 대상 허용 컬럼.
        client (OpenAI): OpenAI SDK 클라이언트.
    """

    def __init__(self, config_path: str):
        """클라이언트와 설정을 초기화합니다.

        Args:
            config_path (str): JSON 설정 파일 경로.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        self.config = config
        self.embedding_columns = ["상담제목", "상담내용요약"]

        load_dotenv(verbose=True)
        print("## config loaded ##")

        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        print("## client loaded ##")

    # embedding batch 처리
    def make_embedding(
        self, df: pd.DataFrame, column_name: str, batch_jsonl_path: str, meta_path: str
    ):
        """임베딩 Batch를 생성합니다.

        주어진 데이터프레임에서 `column_name` 텍스트를 입력으로 사용해 Batch Embeddings 요청을
        JSONL로 저장하고 업로드한 뒤 Batch를 생성합니다.

        Args:
            df (pd.DataFrame): 입력 데이터.
            column_name (str): 임베딩 대상 컬럼명. `상담제목` 또는 `상담내용요약`만 허용.
            batch_jsonl_path (str): 요청 JSONL 저장 경로.
            meta_path (str): 요청-메타 매핑 CSV 저장 경로.

        Returns:
            Any: OpenAI Batch 생성 메타 정보.

        Raises:
            ValueError: 허용되지 않은 컬럼명을 지정한 경우.
        """
        id_list = []
        metadata_list = []

        if column_name in self.embedding_columns:
            url = "/v1/embeddings"

            print("## Embedding model loaded")

        else:
            raise ValueError("## Wrong column name!")

        ## df column rename
        df.rename(
            columns={
                "제목": "상담제목",
                "상담내용": "상담내용요약",
                "진행 과정 및 결과": "진행_과정_및_결과",
                "작성일": "상담일시",
            },
            inplace=True,
        )

        with open(batch_jsonl_path, "w", encoding="utf-8") as f:
            for idx, row in df.iterrows():
                # print('## row : ', row)
                json_line = json.dumps(
                    {
                        "custom_id": f"request-{idx}",
                        "method": "POST",
                        "url": url,
                        "body": {
                            "input": row[column_name],
                            "model": self.config["data_config"]["embedding_model"],
                            "encoding_format": "float",
                            "dimensions": 1024,
                        },
                    },
                    ensure_ascii=False,
                )

                f.write(json_line + "\n")

                id_list.append(f"request-{idx}")

                metadata_list.append(
                    {
                        "상담제목": {"raw_text": row["상담제목"], "Embedding": []},
                        "상담내용요약": {
                            "raw_text": row["상담내용요약"],
                            "Embedding": [],
                        },
                        "진행_과정_및_결과": row["진행_과정_및_결과"],
                        "내담자_정보": {
                            "거주지역": row["거주지역"],
                            "국적": row["국적"],
                            "체류자격": row["체류자격"],
                            "추가정보": [],  # LLM 추출 부분 빈 리스트로 초기화
                        },
                        "해결방법": "",  # LLM 추출 부분 빈 문자열로 초기화
                        "metadata": {
                            "상담유형": row["상담유형"],
                            "상담일시": row["상담일시"].strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    }
                )

        print(f"## jsonl file saved in {batch_jsonl_path}")

        if os.path.exists(meta_path):
            print("## meta file already exists!")

        else:
            metadata_csv = pd.DataFrame(columns=["request_id", "metadata"])
            metadata_csv["request_id"] = id_list
            metadata_csv["metadata"] = metadata_list

            metadata_csv.to_csv(meta_path, index=False)
            print(f"## meta file saved in {meta_path}")

        batch_input_file = self.client.files.create(
            file=open(batch_jsonl_path, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        api_meta = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=url,
            completion_window="24h",
            metadata={"description": "make embedding"},
        )

        return api_meta

    # generate batch 처리 - 내담자정보, 해결방안
    def make_generate(
        self, df: pd.DataFrame, batch_jsonl_path: str, meta_path: str, prompt_path=None
    ):
        """생성(챗 컴플리션) Batch를 생성합니다.

        상담 제목/요약/기록을 하나의 사용자 입력으로 구성하여 Chat Completions Batch를 생성합니다.

        Args:
            df (pd.DataFrame): 입력 데이터.
            batch_jsonl_path (str): 요청 JSONL 저장 경로.
            meta_path (str): 요청-메타 매핑 CSV 저장 경로.
            prompt_path (str | None): `{"system_prompt": str, "user_prompt": str}` 문자열 파일 경로.

        Returns:
            Any: OpenAI Batch 생성 메타 정보.
        """
        id_list = []
        metadata_list = []

        with open(prompt_path, "r", encoding="utf-8") as file:
            PROMPT = file.read()

        PROMPT = ast.literal_eval(PROMPT)
        url = "/v1/chat/completions"
        print("## Generate model loaded")

        ## df column rename
        df.rename(
            columns={
                "제목": "상담제목",
                "상담내용": "상담내용요약",
                "진행 과정 및 결과": "진행_과정_및_결과",
                "작성일": "상담일시",
            },
            inplace=True,
        )

        ## string validation test ##

        with open(batch_jsonl_path, "w", encoding="utf-8") as f:
            for idx, row in df.iterrows():

                user_input = f"""<제목>{row['상담제목']}</제목>
                <전체상담내용>{row['상담내용요약']}</전체상담내용>
                <상담기록>\n{row['진행_과정_및_결과']}</상담기록>"""

                json_line = json.dumps(
                    {
                        "custom_id": f"request-{idx}",
                        "method": "POST",
                        "url": url,
                        "body": {
                            "model": self.config["data_config"]["generate_model"],
                            "messages": [
                                {"role": "system", "content": PROMPT["system_prompt"]},
                                {
                                    "role": "user",
                                    "content": PROMPT["user_prompt"]
                                    + "\n"
                                    + user_input
                                    + "\n\n[output]\n",
                                },
                            ],
                            "max_tokens": self.config["data_config"]["max_tokens"],
                        },
                    },
                    ensure_ascii=False,
                )
                f.write(json_line + "\n")

                id_list.append(f"request-{idx}")

                metadata_list.append(
                    {
                        "상담제목": {"raw_text": row["상담제목"], "Embedding": []},
                        "상담내용요약": {
                            "raw_text": row["상담내용요약"],
                            "Embedding": [],
                        },
                        "진행_과정_및_결과": row["진행_과정_및_결과"],
                        "내담자_정보": {
                            "거주지역": row["거주지역"],
                            "국적": row["국적"],
                            "체류자격": row["체류자격"],
                            "추가정보": [],  # LLM 추출 부분 빈 리스트로 초기화
                        },
                        "해결방법": "",  # LLM 추출 부분 빈 문자열로 초기화
                        "metadata": {
                            "상담유형": row["상담유형"],
                            "상담일시": row["상담일시"].strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    }
                )

        print(f"## jsonl file saved in {batch_jsonl_path}")

        # metafile 없을 때
        if os.path.exists(meta_path):
            print("## meta file already exists!")

        # metafile 있을 때
        else:
            metadata_csv = pd.DataFrame(columns=["request_id", "metadata"])
            metadata_csv["request_id"] = id_list
            metadata_csv["metadata"] = metadata_list

            metadata_csv.to_csv(meta_path, index=False)
            print(f"## meta file saved in {meta_path}")

        batch_input_file = self.client.files.create(
            file=open(batch_jsonl_path, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        api_meta = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=url,
            completion_window="24h",
            metadata={"description": "make generation"},
        )

        return api_meta

    # triplet extractor 배치 취소
    def cancel_batch(self, id_list, custom_id=None):
        """진행 중인 배치를 취소합니다.

        Args:
            id_list (list[str]): 요청 ID 목록.
            custom_id (str | None): 특정 배치 ID.
        """
        # custom id 입력 있을 경우 -> 특정 request id만 cancel
        if custom_id != None:
            self.client.batches.cancel(custom_id)

        else:
            for request_id in id_list:
                self.client.batches.cancel(request_id)
        return

    # batch id 받아서 status 확인
    def get_batch_status(self, batch_id):
        """배치 상태를 조회합니다.

        Args:
            batch_id (str): 배치 ID.

        Returns:
            Any: Batch 객체(상태/입출력 파일 ID 포함).
        """
        return self.client.batches.retrieve(batch_id)

    def get_batch_answer(self, output_file_id, result_path):
        """배치 결과 파일을 다운로드합니다.

        Args:
            output_file_id (str): 출력 파일 ID.
            result_path (str): 저장할 로컬 경로.
        """
        answer = self.client.files.content(output_file_id).content
        with open(result_path, "wb") as file:
            file.write(answer)

        print(f"## file saved in {result_path}")
        return


##############################


class BatchClientExternal:
    """외부 수집 데이터용 배치 클라이언트.

    주로 `title`, `contents`, `url`, 선택적으로 `date` 열을 갖는 데이터프레임을 대상으로
    생성 Batch를 구성합니다. 임베딩 메서드 사용 시에는 사전 전처리로 내부 스키마에 맞추세요.
    """

    def __init__(self, config_path: str):
        """클라이언트와 설정을 초기화합니다.

        Args:
            config_path (str): JSON 설정 파일 경로.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        self.config = config

        load_dotenv(verbose=True)
        print("## config loaded ##")

        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        print("## client loaded ##")

    # embedding batch 처리
    def make_embedding(
        self, df: pd.DataFrame, column_name: str, batch_jsonl_path: str, meta_path: str
    ):
        """임베딩 Batch를 생성합니다(외부 데이터).

        외부 데이터는 내부 스키마(`상담제목`, `상담내용요약` 등)에 맞게 열 이름을 정규화한 뒤 사용하세요.

        Args:
            df (pd.DataFrame): 입력 데이터프레임.
            column_name (str): 임베딩 대상 컬럼명.
            batch_jsonl_path (str): 요청 JSONL 저장 경로.
            meta_path (str): 메타데이터 CSV 저장 경로.

        Returns:
            Any: OpenAI Batch 생성 메타 정보.
        """
        id_list = []
        metadata_list = []

        if column_name in self.embedding_columns:
            url = "/v1/embeddings"

            print("## Embedding model loaded")

        else:
            raise ValueError("## Wrong column name!")

        ## df column rename
        df.rename(
            columns={
                "제목": "상담제목",
                "상담내용": "상담내용요약",
                "진행 과정 및 결과": "진행_과정_및_결과",
                "작성일": "상담일시",
            },
            inplace=True,
        )

        with open(batch_jsonl_path, "w", encoding="utf-8") as f:
            for idx, row in df.iterrows():
                # print('## row : ', row)
                json_line = json.dumps(
                    {
                        "custom_id": f"request-{idx}",
                        "method": "POST",
                        "url": url,
                        "body": {
                            "input": row[column_name],
                            "model": self.config["data_config"]["embedding_model"],
                            "encoding_format": "float",
                            "dimensions": 1024,
                        },
                    },
                    ensure_ascii=False,
                )

                f.write(json_line + "\n")

                id_list.append(f"request-{idx}")

                metadata_list.append(
                    {
                        "상담제목": {"raw_text": row["상담제목"], "Embedding": []},
                        "상담내용요약": {
                            "raw_text": row["상담내용요약"],
                            "Embedding": [],
                        },
                        "진행_과정_및_결과": row["진행_과정_및_결과"],
                        "내담자_정보": {
                            "거주지역": row["거주지역"],
                            "국적": row["국적"],
                            "체류자격": row["체류자격"],
                            "추가정보": [],  # LLM 추출 부분 빈 리스트로 초기화
                        },
                        "해결방법": "",  # LLM 추출 부분 빈 문자열로 초기화
                        "metadata": {
                            "상담유형": row["상담유형"],
                            "상담일시": row["상담일시"].strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    }
                )

        print(f"## jsonl file saved in {batch_jsonl_path}")

        if os.path.exists(meta_path):
            print("## meta file already exists!")

        else:
            metadata_csv = pd.DataFrame(columns=["request_id", "metadata"])
            metadata_csv["request_id"] = id_list
            metadata_csv["metadata"] = metadata_list

            metadata_csv.to_csv(meta_path, index=False)
            print(f"## meta file saved in {meta_path}")

        batch_input_file = self.client.files.create(
            file=open(batch_jsonl_path, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        api_meta = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=url,
            completion_window="24h",
            metadata={"description": "make embedding"},
        )

        return api_meta

    # generate batch 처리 - 내담자정보, 해결방안
    def make_generate(
        self, df: pd.DataFrame, batch_jsonl_path: str, meta_path: str, prompt_path=None
    ):
        """생성(챗 컴플리션) Batch를 생성합니다(외부 데이터).

        `title`을 입력으로 사용하고 `<내담자정보>` 블록 생성을 유도하는 프롬프트를 구성합니다.

        Args:
            df (pd.DataFrame): 입력 데이터(`title`, `contents`, `url`, 선택 `date`).
            batch_jsonl_path (str): 요청 JSONL 저장 경로.
            meta_path (str): 메타데이터 CSV 저장 경로.
            prompt_path (str | None): `{"system_prompt": str, "user_prompt": str}` 문자열 파일 경로.

        Returns:
            Any: OpenAI Batch 생성 메타 정보.
        """
        id_list = []
        metadata_list = []

        with open(prompt_path, "r", encoding="utf-8") as file:
            PROMPT = file.read()

        PROMPT = ast.literal_eval(PROMPT)
        url = "/v1/chat/completions"
        print("## Generate model loaded")

        ## string validation test ##

        with open(batch_jsonl_path, "w", encoding="utf-8") as f:
            for idx, row in df.iterrows():

                user_input = f"""[input]\n<상담내용>{row['title']}</상담내용>\n[output]\n<내담자정보>"""

                json_line = json.dumps(
                    {
                        "custom_id": f"request-{idx}",
                        "method": "POST",
                        "url": url,
                        "body": {
                            "model": self.config["data_config"]["generate_model"],
                            "messages": [
                                {"role": "system", "content": PROMPT["system_prompt"]},
                                {
                                    "role": "user",
                                    "content": PROMPT["user_prompt"]
                                    + "\n"
                                    + user_input,
                                },
                            ],
                            "max_tokens": self.config["data_config"]["max_tokens"],
                        },
                    },
                    ensure_ascii=False,
                )
                f.write(json_line + "\n")

                id_list.append(f"request-{idx}")

                if "date" in df.columns:
                    metadata_list.append(
                        {
                            "상담제목": {"raw_text": row["title"], "Embedding": []},
                            "내담자_추가정보": {"raw_text": [], "Embedding": []},
                            "해결방법": {"raw_text": row["contents"], "Embedding": []},
                            "metadata": {
                                "url": row["url"],
                                "date": row["date"].strftime("%Y-%m-%d"),
                            },
                        }
                    )
                else:
                    metadata_list.append(
                        {
                            "상담제목": {"raw_text": row["title"], "Embedding": []},
                            "내담자_추가정보": {"raw_text": [], "Embedding": []},
                            "해결방법": {"raw_text": row["contents"], "Embedding": []},
                            "metadata": {"url": row["url"]},
                        }
                    )

        print(f"## jsonl file saved in {batch_jsonl_path}")

        # metafile 없을 때
        if os.path.exists(meta_path):
            print("## meta file already exists!")

        # metafile 있을 때
        else:
            metadata_csv = pd.DataFrame(columns=["request_id", "metadata"])
            metadata_csv["request_id"] = id_list
            metadata_csv["metadata"] = metadata_list

            metadata_csv.to_csv(meta_path, index=False)
            print(f"## meta file saved in {meta_path}")

        batch_input_file = self.client.files.create(
            file=open(batch_jsonl_path, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        api_meta = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=url,
            completion_window="24h",
            metadata={"description": "make generation"},
        )

        return api_meta

    # triplet extractor 배치 취소
    def cancel_batch(self, id_list, custom_id=None):
        """진행 중인 배치를 취소합니다(외부 데이터).

        Args:
            id_list (list[str]): 요청 ID 목록.
            custom_id (str | None): 특정 배치 ID.
        """
        # custom id 입력 있을 경우 -> 특정 request id만 cancel
        if custom_id != None:
            self.client.batches.cancel(custom_id)

        else:
            for request_id in id_list:
                self.client.batches.cancel(request_id)
        return

    # batch id 받아서 status 확인
    def get_batch_status(self, batch_id):
        """배치 상태를 조회합니다(외부 데이터)."""
        return self.client.batches.retrieve(batch_id)

    def get_batch_answer(self, output_file_id, result_path):
        """배치 결과 파일을 다운로드합니다(외부 데이터)."""
        answer = self.client.files.content(output_file_id).content
        with open(result_path, "wb") as file:
            file.write(answer)

        print(f"## file saved in {result_path}")
        return
