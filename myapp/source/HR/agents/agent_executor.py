import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from dotenv import load_dotenv
import sys
from ..tools.rag_tool import search_documents, calculate_retirement_pay, search_naver_news, search_naver_web, hybrid_search, search_company_info
import torch

# 환경 변수 로드
load_dotenv()

# 도구 목록 - 새로운 통합 검색 도구 추가 (우선순위 순서)
tools = [
    hybrid_search,           # 1순위: 통합 검색 (내부 우선)
    search_company_info,     # 2순위: 회사 전용 검색
    calculate_retirement_pay, # 3순위: 퇴직금 계산
    search_naver_news,       # 4순위: 네이버 뉴스
    search_documents,        # 5순위: 기존 문서 검색 (하위 호환)
    search_naver_web         # 6순위: 네이버 웹 검색
]

# 시스템 프롬프트 - 도구 사용 우선순위 명확화
system_prompt = """
당신은 회사 내부 정보와 외부 정보를 통합하여 답변하는 AI 어시스턴트입니다.
도구 사용 우선순위를 반드시 준수하고, 없는 자료는 모른다고 답변하세요.

🎯 **도구 사용 우선순위:**

**1순위: hybrid_search** ⭐ (대부분의 질문에 사용)
- 모든 일반적인 질문에 우선 사용
- 내부 문서를 먼저 검색하고 필요시 외부 검색 보완
- "정주영이 누구야?", "AI 기술 동향" 등 모든 질문

**2순위: search_company_info** 🏢 (회사 전용 정보)
- 회사 내부 정보만 확실히 알고 싶을 때
- "우리 회사 휴가 정책", "조직도", "인사규정" 등

**3순위: calculate_retirement_pay** 💰
- 퇴직금 계산이 명확한 경우에만 사용

**4순위: search_naver_news** 📰
- 최신 뉴스, 동향이 필요할 때

**나머지 도구들은 특별한 경우에만 사용**

🎯 **답변 가이드라인:**
- 대부분의 질문: hybrid_search 사용 (내부 + 외부 정보 통합)
- 회사 관련 키워드 포함 시: 내부 정보 우선 표시
- 출처를 명확히 구분하여 표시
- "정주영" 같은 경우: 회사 대표와 현대 창업자 정보 모두 제공

📝 **예시:**
- "정주영이 누구야?" → hybrid_search (회사 대표 + 현대 창업자 정보)
- "우리 회사 대표가 누구야?" → search_company_info (회사 정보만)
- "최근 AI 뉴스" → search_naver_news (뉴스 전용)
"""

# 프롬프트 템플릿 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# LLM 초기화 - 모델 선택 로직 추가
def get_llm():
    """환경변수에 따라 적절한 LLM을 반환합니다."""
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0"))
    
    if llm_model.startswith("gpt"):
        # OpenAI 모델 사용
        return ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif "HyperCLOVAX" in llm_model or "naver-hyperclovax" in llm_model:
        # HyperCLOVAX 모델 사용 (HuggingFace Transformers 기반)
        try:
            from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            print(f"HyperCLOVAX 모델 로드 중: {llm_model}")
            
            # HuggingFace 토큰 사용
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # 모델과 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                llm_model, 
                token=hf_token if hf_token else None
            )
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                token=hf_token if hf_token else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # 파이프라인 생성
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "1024"))
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except ImportError:
            print("HuggingFace 모델을 사용하려면 transformers 라이브러리가 필요합니다.")
            print("OpenAI 모델로 대체합니다.")
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                streaming=True,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            print(f"HyperCLOVAX 모델 로드 실패: {e}")
            print("OpenAI 모델로 대체합니다.")
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                streaming=True,
                api_key=os.getenv("OPENAI_API_KEY")
            )
    else:
        # 기본값: OpenAI GPT 모델
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY")
        )

llm = get_llm()

# 에이전트 생성
agent = create_openai_functions_agent(llm, tools, prompt)

# 에이전트 실행기 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 채팅 히스토리 관리를 위한 클래스
class ChatManager:
    def __init__(self):
        self.chat_history: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def get_chat_history(self):
        return self.chat_history

    def clear_history(self):
        self.chat_history = []

# 싱글톤 인스턴스 생성
chat_manager = ChatManager()

def process_query(query: str) -> str:
    """사용자 쿼리를 처리하고 응답을 반환합니다."""
    # 사용자 메시지 추가
    chat_manager.add_message("user", query)
    
    # 에이전트 실행
    response = agent_executor.invoke({
        "input": query,
        "chat_history": chat_manager.get_chat_history()
    })
    
    # AI 응답 추가
    chat_manager.add_message("assistant", response["output"])
    
    return response["output"] 