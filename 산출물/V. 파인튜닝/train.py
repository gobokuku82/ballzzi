import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 모델과 토크나이저 설정
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 8bit 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 모델 준비
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# JSONL 데이터셋 로드
dataset = load_dataset(
    'json',
    data_files='training_dataset.jsonl',
    split='train',
    cache_dir='./cache',  # 캐시 디렉토리 지정
    streaming=False  # 대용량 데이터의 경우 True로 설정 가능
)

print(f"데이터셋 크기: {len(dataset)} 개")
print(f"데이터셋 필드: {dataset.column_names}")
print("\n첫 번째 데이터 샘플:")
print(dataset[0])

def tokenize_function(examples):
    # 입력과 출력을 구분하여 토큰화
    task_type = examples['task_type']
    prompt = examples['instruction'] + "\n" + examples['input'] if examples['input'] else examples['instruction']
    
    # task_type에 따라 다른 프롬프트 형식 사용
    if task_type == "football_query":
        # SQL 쿼리 생성을 위한 프롬프트
        input_text = f"### SQL 쿼리 생성 요청: {prompt}\n\n### SQL 쿼리: "
        output_text = f"{examples['output']['sql_query']}</s>"
    else:  # non_football
        # 일반적인 응답을 위한 프롬프트
        input_text = f"### 질문: {prompt}\n\n### 답변: "
        # non_football의 경우 sql_query가 null
        output_text = f"{examples['output']['sql_query'] if examples['output']['sql_query'] else 'NULL'}</s>"
    
    # 입력 토큰화
    model_inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    
    # 레이블용 토큰화
    labels = tokenizer(
        output_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    
    # 레이블 설정
    model_inputs["labels"] = labels["input_ids"]
    
    # 타입 정보 추가 (나중에 평가나 분석에 활용 가능)
    model_inputs["task_type"] = task_type
    
    return model_inputs

# 데이터셋 전처리
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=[col for col in dataset.column_names if col != 'task_type']
)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,
)

# 트레이너 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 학습 시작
trainer.train() 