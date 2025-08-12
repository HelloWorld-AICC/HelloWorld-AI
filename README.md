# Hello World
안녕하세요! 저희는 외국인 노동자를 위한 지원센터 AICC 어플을 개발하는 "끝까지 간다" 팀입니다😄

[AI 개발 레포](https://github.com/yeowonh/HelloWorld-AI)

[서버 레포](https://github.com/HelloWorld-AICC/HelloWorld-Server) | [프론트 레포](https://github.com/HelloWorld-AICC/HelloWorld-Front)

## 환경 세팅
- .venv (uv)로 환경 관리
    - CUDA 자동 세팅 (없을 경우 skip)
    -
```python
bash init.sh
```

## 커밋 방식
    - git add {스테이징할 폴더/파일}
    - git commit
- commit 시 pre-hook 동작 후 자동으로 fix됨 (다시 git add 필요)
- commit 이후에는 새 코드창이 열림 -> commit template에 맞춰서 커밋 진행


## 디렉토리 구조
```bash
.
├── Azure // Azure 서버 실행 관련
│   ├── execute.sh
│   ├── function_app.py
│   ├── host.json
│   ├── local.settings.json
│   ├── logs
│   │   └── function_app.log
│   └── model.py // ChatOpenAI 이용한 기본적인 모델 형태
├── configs
│   └── config.json // 설정 파일
├── data
│   ├── BatchClient.py // (BatchClient) MongoDB 적재용 상담 내역 요약 생성 및 jsonl 결과물 산출
│   ├── crawler.ipynb // 크롤링 파일
│   ├── make_mongodb.py // 후처리된 json, jsonl -> MongoDB atlas 적재
│   ├── notebooks
│   │   ├── 외부데이터_데이터셋구축.ipynb // (BatchClient) 지식인, 법률 공단 외 기타 웹사이트 데이터 정제
│   │   ├── 상담사례_데이터셋구축.ipynb // (BatchClient) 경기도이민사회통합지원센터 데이터 정제
│   │   ├── make_testset.ipynb // 테스트셋 구축을 위한 Mongodb Atlas 탐색
│   │   └── rawdata_postprocess.ipynb // 후처리
│   ├── original_data
│   │   ├── batch_jsonl // raw data -> 후처리 결과 파일
│   │   │   ├── input
│   │   │   │   ├── consultation_documents_2022_상담내용요약_batch.jsonl
│   │   │   │   ├── consultation_documents_2022_내담자정보_batch.jsonl
│   │   │   │   ├── consultation_documents_2022_상담제목_batch.jsonl
│   │   │   │   ├── consultation_documents_2022_해결방법_batch.jsonl
│   │   │   │   ├── consultation_documents_2023_상담내용요약_batch.jsonl
│   │   │   │   ├── consultation_documents_2023_내담자정보_batch.jsonl
│   │   │   │   ├── consultation_documents_2023_상담제목_batch.jsonl
│   │   │   │   ├── consultation_documents_2023_해결방법_batch.jsonl
│   │   │   │   ├── jisikin_consultation_2020_2024_내담자정보_batch.jsonl
│   │   │   │   └── klac_내담자정보_batch.jsonl
│   │   │   └── output
│   │   │       ├── consultation_documents_2022_batch.jsonl
│   │   │       ├── consultation_documents_2022_상담내용요약_batch.jsonl
│   │   │       ├── consultation_documents_2022_내담자정보_batch.jsonl
│   │   │       ├── consultation_documents_2022_상담제목_batch.jsonl
│   │   │       ├── consultation_documents_2022_해결방법_batch.jsonl
│   │   │       ├── consultation_documents_2023_batch.jsonl
│   │   │       ├── consultation_documents_2023_상담내용요약_batch.jsonl
│   │   │       ├── consultation_documents_2023_내담자정보_batch.jsonl
│   │   │       ├── consultation_documents_2023_상담제목_batch.jsonl
│   │   │       ├── consultation_documents_2023_해결방법_batch.jsonl
│   │   │       ├── jisikin_consultation_2020_2024_batch.jsonl
│   │   │       ├── jisikin_consultation_2020_2024_내담자정보_batch.jsonl
│   │   │       ├── klac_내담자정보_batch.jsonl
│   │   │       └── klac_batch.jsonl
│   │   ├── contents // 원본 데이터
│   │   │   ├── 경기도외국인지원센터_상담사례_contents.json
│   │   │   ├── 외국인체류관리_contents.json
│   │   │   ├── 키워드처리완료.json
│   │   │   └── need_preprocess
│   │   │       ├── 대한법률구조공단_상담사례_contents.json
│   │   │       ├── 서울노동포털_contents.json
│   │   │       ├── easylaw_kr_contents.json
│   │   │       ├── 전처리필요.json
│   │   │       └── legalqa_contents.json
│   │   ├── 법률_filtered_data.json
│   │   ├── 법률_filtered_data-LAPTOP-CAHVEOTT.json
│   │   ├── 상담사례.json
│   │   ├── 비자.json
│   │   ├── Legal
│   │   │   ├── 대한법률구조공단_상담사례_contents.json
│   │   │   ├── 법률규정텍스트_근로자_contents.json
│   │   │   ├── 서울노동포털_contents.json
│   │   │   ├── 국내법률_민형사_근로기준법_판결문_contents.json
│   │   │   ├── 생활법령_contents.json
│   │   │   ├── 법률.json
│   │   │   ├── 법률-LAPTOP-CAHVEOTT.json
│   │   │   └── legalqa_contents.json
│   │   ├── merged.json
│   │   ├── merged_output.json
│   │   ├── merged_output-LAPTOP-CAHVEOTT.json
│   │   ├── meta_csv // (BatchClient) OpenAI batch api 활용하기 위한 concat용 메타 정보
│   │   │   ├── consultation_documents_2022_meta.csv
│   │   │   ├── consultation_documents_2023_meta.csv
│   │   │   ├── jisikin_consultation_2020_2024_meta.csv
│   │   │   └── klac_meta.csv
│   │   ├── preprocessed.csv // 전처리 데이터
│   │   ├── preprocessed.json // 전처리 데이터
│   │   ├── raw_data
│   │   │   ├── backup
│   │   │   │   ├── labor_counseling_details_page_1000.json
│   │   │   │   ├── labor_counseling_details_page_100.json
│   │   │   │   ├── labor_counseling_details_page_200.json
│   │   │   │   ├── labor_counseling_details_page_300.json
│   │   │   │   ├── labor_counseling_details_page_400.json
│   │   │   │   ├── labor_counseling_details_page_500.json
│   │   │   │   ├── labor_counseling_details_page_600.json
│   │   │   │   ├── labor_counseling_details_page_700.json
│   │   │   │   ├── labor_counseling_details_page_800.json
│   │   │   │   └── labor_counseling_details_page_900.json
│   │   │   ├── consultation_documents_filtered2022.jsonl
│   │   │   ├── consultation_documents_filtered2023.jsonl
│   │   │   ├── huggingface
│   │   │   │   └── 생활법령.json
│   │   │   └── Web
│   │   │       ├── 경기도외국인지원센터_상담사례_contents.json
│   │   │       ├── 외국인체류관리_contents.json
│   │   │       ├── 경기도외국인지원센터_상담사례.json
│   │   │       ├── 대한법률구조공단_상담사례.json
│   │   │       ├── 외국인체류관리.json
│   │   │       ├── 키워드처리완료.json
│   │   │       └── 서울노동포털.json
│   │   └── Visa
│   │       └── 외국인체류관리_contents.json
│   ├── postprocessed_data // 후처리 된 데이터
│   │   ├── comwel.json
│   │   ├── easylaw.json
│   │   ├── gov_foreigner.json
│   │   ├── hrdk.json
│   │   ├── immigration.json
│   │   ├── industrial_accident.json
│   │   ├── jisikin_2020_2024.json
│   │   ├── jisikin_consultation_2020_2024.csv
│   │   ├── jisikin_filtered.jsonl
│   │   ├── jisikin.json
│   │   ├── klac.csv
│   │   ├── klac.json
│   │   ├── klac.jsonl
│   │   ├── merged_add_summary.json
│   │   ├── merged.json
│   │   ├── moel_filtered.json
│   │   ├── moj.json
│   │   ├── nps.json
│   │   ├── preprocessed.csv
│   │   ├── preprocessed.json
│   │   ├── seoullabor.json
│   │   ├── visa.json
│   │   └── workinkorea.json
│   └── utils.py
├── gradio // gradio 예시용
│   ├── chat_gradio.py
│   ├── README.md
│   └── summary_gradio.py
├── init.sh // 초기화용 init shell (최초 실행 필요)
├── main.py // 추후 실행 파일 작성
├── model // 구동 모델 및 템플릿 작성
│   ├── app.py
│   ├── Dockerfile
│   ├── function_app.py
│   ├── inference_test
│   │   ├── 내담자정보_ANN_similarity_test.ipynb
│   │   ├── blossom_inference_test.ipynb
│   │   ├── create_client_embeddings.py
│   │   ├── create_client_title_embeddings.py
│   │   ├── create_legalQAv2_contents_embeddings.py
│   │   ├── create_legalQAv2_title_contents_embeddings.py
│   │   ├── create_legalQAv2_title_embeddings.py
│   │   ├── 내담자정보_embedding.ipynb
│   │   ├── 내담자정보_ENN_similarity_test.ipynb
│   │   ├── gemma_inference_test.ipynb
│   │   ├── model_quantization_bllossom.ipynb
│   │   ├── model_quantization_bllossom_vision.ipynb
│   │   └── vllm_test.ipynb
│   ├── model.py
│   ├── model_quantized.py
│   ├── templates
│   │   ├── chat_template.txt
│   │   └── resume_template.txt
│   └── utils.py
├── prompts // 프롬프트 저장
│   ├── cv_prompt.json
│   ├── generate_data_prompt_info_only.json
│   └── generate_data_prompt.json
├── pyproject.toml // uv (.venv) 용 버전관리
├── README.md
├── test // 간단한 모듈 테스트용 파일
│   ├── chat_test.py
│   ├── cv_test.py
│   ├── function_app.py
│   ├── logging_test.py
│   ├── model.py
│   ├── mongodb_test.py
│   ├── 이력서_test.ipynb
│   └── test.ipynb
├── utils // 기타 유틸
│   ├── huggingface_download.py
│   └── huggingface_upload.py
└── uv.lock

```




## 💡 개발 동기

2024년 올해, 외국인 노동자 지원센터에 대한 예산 삭감으로 인해 근무 인원이 대폭 줄게 되었고 이로 인해 도움이 필요한 외국인들이 지원을 받지 못하는 상황입니다. 이를 AICC(AI 기반 고객센터)로 해결하고자 본 프로젝트를 구상하게 되었습니다.

## ❓ AICC란?

> AICC는 AI Contact Center의 약자로, 음성 인식(STT), 자연어 처리(NLP), 음성 합성(TTS) 등의 인공지능(AI) 기술을 활용하여 컨택센터의 업무를 효율적이고 창의적으로 수행할 수 있도록 도와주는 서비스입니다. AICC는 AI 어시스턴트, AI 콜봇, 챗봇, 스마트콜백, 알림톡, 상담톡 등 다양한 기능을 제공하여 고객 상담 업무를 자동화하거나 최적화합니다.
LG U+나 쿠쿠와 같은 대기업의 사례를 보면 AICC의 도입으로 재상담률이 무려 75%나 감소하는 효과가 있었습니다.



## 🔧 주요 기능

- 외국인 근로자는 상담 받고자 하는 내용 (법률, 비자 문의 등) 을 챗봇으로 상담합니다.
- 챗봇은 데이터를 통해 법률 정보 및 예시, 구체적 해결 방안을 제시해줍니다.
- 추가적으로 상담이 필요한 경우, 어플 내에서 직접 상담 예약이 가능합니다.
- 오프라인 상담을 진행할 때의 수고를 덜고자, 채팅 요약문을 상담사에게 제공해 더욱 원활한 상담이 가능하도록 돕습니다.


## 😊 예상 기대 효과

- 상담을 위한 인력 비용 감축
- 채팅 요약문 제공을 통한 상담 효율성 증대
- 추후 누적된 데이터 활용한 상담 품질 개선 기대
- CTI(컴퓨터 전화 통합), IVR(음성 안내) 추가로 컨텍센터 콜 인프라 확장 가능

## UI
![350707716-5aa038d0-3c66-4557-ab09-1de46d74d288](https://github.com/user-attachments/assets/a8c7af83-ff8e-4472-9eeb-50360e112d8a)


## ⚙️ 시스템 아키텍처
![350707464-fb9b22ba-7dbc-4762-aa8d-0a374f5b7ac2](https://github.com/user-attachments/assets/ca5d7bd9-022f-4a02-a985-265cda524ea3)


- 다수의 리소스를 관리하기 위해 docker-compose를 사용합니다.
- 모델 호출을 위해 FastAPI 프레임워크를 사용하고, 이를 SpringbBoot에 연동합니다.
- LLM의 요청과 응답, 채팅 로그의 저장 등의 이벤트 기반 데이터 처리를 위해 Apache Kafka를 사용하는데, AWS에 맞춰서 배포부터 운영을 담당하는 Amazon MSK를 사용합니다.
- 로그데이터를 저장할 mongoDB와 대용량의 텍스트 데이터를 빠르게 색인하고 검색할 수 있게 elastic search로 mongoDB의 로그를 조회, 저장합니다. 이 로그 데이터들을 Input으로 Flan-t5모델에 넣습니다. Output인 요약문은 MYSQL에 고객정보와함께 저장됩니다.

## ⚙️ AI
![Group 28](https://github.com/user-attachments/assets/b63defe3-217a-464b-87fc-18001aaa5692)
