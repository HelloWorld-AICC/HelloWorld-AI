#!/bin/bash

# .env 파일에서 환경 변수 읽기
export $(grep -v '^#' .env | xargs)

# Flask 앱 실행
flask run