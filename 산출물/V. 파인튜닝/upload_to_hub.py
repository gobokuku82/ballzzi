from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os
from dotenv import load_dotenv

def upload_model_to_hub(
    checkpoint_path,
    repo_name,
    token=None,
    private=True,
    model_card_content=None
):
    """
    허깅페이스 허브에 모델을 업로드합니다.
    
    Args:
        checkpoint_path (str): 로컬 체크포인트 경로
        repo_name (str): 생성할 허깅페이스 레포지토리 이름 (예: "LaargePaw/KoAlpaca-Soccer-Query-Generator")
        token (str, optional): 허깅페이스 토큰
        private (bool): 비공개 레포지토리 여부
        model_card_content (str, optional): 모델 카드 내용
    """
    # 환경 변수에서 토큰 로드
    load_dotenv()
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("HF_TOKEN이 필요합니다. 환경변수나 파라미터로 제공해주세요.")

    # API 초기화
    api = HfApi()

    # 레포지토리 생성
    print(f"레포지토리 생성 중: {repo_name}")
    create_repo(repo_name, private=private, token=token, exist_ok=True)

    # 기본 모델 카드 내용
    if model_card_content is None:
        model_card_content = f"""
# {repo_name.split('/')[-1]}

이 모델은 beomi/KoAlpaca-Polyglot-5.8B를 기반으로 파인튜닝된 모델입니다.

## 모델 설명
- 베이스 모델: beomi/KoAlpaca-Polyglot-5.8B
- 학습 데이터: 축구 관련 SQL 쿼리 생성 데이터셋
- 특징: SQL 쿼리 생성에 특화된 한국어 모델

## 사용 예시
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 모델과 토크나이저 로드
base_model_name = "beomi/KoAlpaca-Polyglot-5.8B"
adapter_name = "{repo_name}"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(model, adapter_name)

# 예시 사용
instruction = "축구선수 데이터베이스에서 다음 질문에 대한 SQL 쿼리를 작성해주세요."
input_text = "스페인에서 패스가 뛰어난 선수를 뽑아줘"
prompt = f"### 질문: {{instruction}}\\n\\n{{input_text}}\\n\\n### 답변:"

# 생성
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
"""

    # 모델 카드 업로드
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(model_card_content)

    print("LoRA 어댑터를 허브에 업로드하는 중...")
    
    # Windows 환경에서 임시 디렉토리 생성 및 파일 복사
    os.makedirs("temp_adapter", exist_ok=True)
    os.system(f'xcopy /E /I /Y "{checkpoint_path}\\*" "temp_adapter"')
    
    # 파일 업로드
    api.upload_folder(
        folder_path="temp_adapter",
        repo_id=repo_name,
        token=token
    )
    
    # 임시 디렉토리 삭제
    os.system('rmdir /S /Q temp_adapter')

    print(f"업로드 완료! 모델을 확인하세요: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    # 환경 변수 설정 방법:
    # 1. .env 파일에 HF_TOKEN=your_token 추가
    # 또는
    # 2. 아래 token 파라미터에 직접 입력

    checkpoint_path = "C:/Users/User/Desktop/SKN/Personal/results/checkpoint-4731"  # 체크포인트 경로
    repo_name = "LaargePaw/KoAlpaca-Soccer-Query-Generator"    # 여기에 본인의 허깅페이스 계정명을 입력하세요

    upload_model_to_hub(
        checkpoint_path=checkpoint_path,
        repo_name=repo_name,
        private=True  # True: 비공개, False: 공개
    ) 