import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json

def load_model(checkpoint_path):
    # GPU가 있으면 GPU 사용, 없으면 CPU 사용
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text):
    # 프롬프트 구성
    prompt = f"### 질문: {instruction}\n\n{input_text}\n\n### 답변:"
    
    # 입력 텍스트 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 출력 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 결과 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### 답변:")[-1].strip()

def main():
    print("모델 로딩 중...")
    # 체크포인트 경로 설정
    checkpoint_path = "your_checkpoint_path"  # 체크포인트 경로를 여기에 입력하세요
    
    # 모델과 토크나이저 로드
    model, tokenizer = load_model(checkpoint_path)
    print("모델 로딩 완료!")
    
    while True:
        print("\n1: Football Query")
        print("2: Non-Football Query")
        print("q: 종료")
        choice = input("선택해주세요: ")
        
        if choice.lower() == 'q':
            break
            
        task_type = "football_query" if choice == "1" else "non_football"
        instruction = input("\n질문을 입력하세요: ")
        input_text = input("추가 입력(선택사항): ")
        
        print("\n응답 생성 중...")
        response = generate_response(model, tokenizer, instruction, input_text)
        print("\n생성된 응답:")
        print(response)

if __name__ == "__main__":
    main()