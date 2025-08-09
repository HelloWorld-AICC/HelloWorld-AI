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

- 커밋 방식
    - git add {스테이징할 폴더/파일}
    - git commit
- commit 시 pre-hook 동작 후 자동으로 fix됨 (다시 git add 필요)
- commit 이후에는 새 코드창이 열림 -> commit template에 맞춰서 커밋 진행


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
