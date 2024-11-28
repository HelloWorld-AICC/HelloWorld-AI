import requests
import json

def test_chat():
    url = "http://localhost:7071/api/question"
    headers = {
        "Content-Type": "application/json"
    }
    
    conversation = []
    
    while True:
        question = input("\n질문을 입력하세요 (종료하려면 'q' 입력): ")
        
        if question.lower() == 'q':
            break
            
        conversation.append({
            "speaker": "human",
            "utterance": question
        })
        
        payload = {
            "Conversation": conversation
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            print(f"\n질문: {question}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"답변: {result['answer']}")
                
                conversation.append({
                    "speaker": "assistant",
                    "utterance": result['answer']
                })
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
        
        print("-" * 80)

if __name__ == "__main__":
    print("대화를 시작합니다. 종료하려면 'q'를 입력하세요.")
    test_chat() 