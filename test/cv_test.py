import requests
import json

def test_cv_generation():
    url = "http://localhost:7071/api/cv_generation"
    headers = {
        "Content-Type": "application/json"
    }
    
    # 테스트용 이력서 데이터
    test_data = {
        "personal": {
            "name": "황예원",
            "nationality": "베트남",
            "visa": "E-7"
        },
        "experience": [
            {"work": "소프트웨어 개발자로서 웹 애플리케이션 개발 및 유지보수 담당"},
            {"work": "고객 관리 시스템 구축 프로젝트 리드 개발자 역할 수행"},
            {"work": "데이터 분석 도구 개발 및 시각화 대시보드 구현"}
        ],
        "language": {
            "korean": "상",
            "others": [
                {"name": "영어", "level": "중"},
                {"name": "베트남어", "level": "상"}
            ]
        },
        "skills": ["Python", "React", "데이터분석", "Git"],
        "strengths": ["문제해결능력", "팀워크", "의사소통능력"],
        "desired_position": "데이터 엔지니어"
    }
    
    try:
        # API 요청 보내기
        response = requests.post(url, headers=headers, json=test_data)
        
        print("\n=== 이력서 생성 테스트 ===")
        
        if response.status_code == 200:
            result = response.json()
            print("\n[한 줄 자기소개]")
            print(result['resume']['introduction'])
            print("\n[상세 자기소개]")
            print(result['resume']['details'])
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_cv_generation() 