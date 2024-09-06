# 기본 이미지로 Python 3.9 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

#CA 인증서 
RUN apt-get update && apt-get install -y ca-certificates

# 필요한 파일 복사
COPY requirements.txt .
COPY app.py .
COPY configs/ ./configs/
COPY .env .

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 포트 설정
EXPOSE 5000

# 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 애플리케이션 실행
CMD ["flask", "run"]