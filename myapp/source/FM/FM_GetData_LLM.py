import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_community.utilities import SQLDatabase
from .tools.create_prompt import create_sql_prompt,create_final_prompt
from .tools.SQL_create import SQLPostprocessChain
from .tools.SQL_execute import SQLExecuteChain
import json
import re
import streamlit as st

# 🔹 환경 변수 로드 (.env에 OPENAI_API_KEY가 들어 있어야 함)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
os.environ["OPENAI_API_KEY"] = api_key

# 🔹 DB 연결
base_dir = os.path.dirname(__file__)
db_path = os.path.join(base_dir, "data", "players_position.db")

if not os.path.exists(db_path):
    raise FileNotFoundError(f"❌ DB 파일이 존재하지 않습니다: {db_path}")

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# 🔹 자연어 → SQL 쿼리 프롬프트
sql_prompt = create_sql_prompt()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 🔹 SQL 생성 체인 생성
sql_chain = sql_prompt | llm
sql_postprocess_chain = SQLPostprocessChain(llm_chain=sql_chain)

# 🔹 SQL 쿼리 실행 체인 생성 
sql_execute_chain = SQLExecuteChain(db_engine=db)

# 🔹 최종 자연어 답변 생성용 체인인
final_prompt = create_final_prompt()
final_llm_chain = final_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# 🔹 외부 호출 함수
def get_answer_from_question(question: str) -> list[dict]:
    try:
        # Step 1: SQL 쿼리 생성
        sql_result = sql_postprocess_chain.invoke({"question": question})
        
        # Step 2: SQL 실행
        execute_result = sql_execute_chain.invoke({"sql_query": sql_result["sql_query"]})

        # Step 3: 최종 답변 생성
        final_result = final_llm_chain.invoke({
            "question": question,
            "query_result": execute_result["query_result"]
        })

        # Step 4: JSON 파싱
        if hasattr(final_result, 'content'):
            llm_result = final_result.content
        else:
            llm_result = str(final_result)
            
        parsed = extract_json_array(llm_result)
        return parsed
    except json.JSONDecodeError:
        return [{"error": "❌ JSON 파싱 실패. 포맷 오류 가능."}]
    except Exception as e:
        return [{"error": f"❌ 예외 발생: {str(e)}"}]

def extract_json_array(text: str) -> list[dict]:
    try:
        # 첫 번째 [ 와 마지막 ] 기준으로 JSON 배열 추출
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return [{"error": "❌ JSON 배열 형식을 찾을 수 없습니다."}]
        
        json_str = match.group(0)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return [{"error": "❌ JSON 파싱 실패. 포맷 오류 가능."}]
    except Exception as e:
        return [{"error": f"❌ 예외 발생: {str(e)}"}]

# 🔹 테스트 실행
# if __name__ == "__main__":
#     sample_question = "10000000 예산 안에서 영입가능한 수비수"
#     print("💬 질문:", sample_question)
#     answer = get_answer_from_question(sample_question)
#     print("\n📢 답변:\n", answer)
