from langchain.prompts import PromptTemplate
def create_sql_prompt():
    # Step 1. 자연어 → SQL 변환 프롬프트
    sql_prompt = PromptTemplate(
    input_variables=["question", "schema"],
    template="""
    당신은 축구와 관련된 데이터베이스에서 입력된 자연어를 SQL구문으로 변경해주는 모델입니다.
    사용자가 다음과 같이 질문했습니다:
    "{question}"

    아래는 축구 선수의 포지션별 데이터 테이블의 스키마입니다:
    {schema}
    현재_능력과 잠재력은 1~200 사이이고
    다른 모든 스탯은 1~20사이입니다.

    이 질문에 맞게 문법에 맞는 SQL 쿼리를 작성해 주세요. 
    where절에 포지션은 포함하지 말아주고
    select절에는 *로 통일해줘
    바로 사용할 수 있도록 꼭 문법에 맞는 SQL 구문만 출력해 주세요.
    sql구문을 생성할 때는 주 스탯의 크기로 정렬해주세요.
    출력할 선수는 최대 3명으로 제한해주세요.
    하나의 테이블 안에서만 검색을 해주세요
    한글로 선수의 이름이 입력된다면 가장 유사한 영어 이름으로 매핑해주세요.
    **질문이 축구와 관련이 없으면 정중히 사유를 말하며 거절해주세요.**
    """
    )

    return sql_prompt

def create_final_prompt():
    # Step 1. 자연어 → SQL 변환 프롬프트
    final_prompt = PromptTemplate(
    input_variables=["question", "query_result"],
    template='''
    당신은 축구 선수 데이터 분석 전문가입니다.

    사용자가 다음과 같은 질문을 했습니다:
    "{question}"

    쿼리 결과는 다음과 같습니다:
    {query_result}

    ---

    이 정보를 바탕으로 아래 형식에 맞춰 응답해 주세요:
    형식 : 
    [
        {{"Name": "이름", "설명": "마크다운 형식으로 자연스럽고 자세한 설명"}}
    ]

    각 선수의 설명에는 다음 요소를 포함해 주세요:
    - 현재 능력과 잠재력
    - 주요 장점 스탯 (예: 패스, 태클, 시야 등)
    - 어떤 역할을 수행할 수 있는지, 어떤 스타일인지
    - 설명은 챕터를 나눠서, 정보를 분류해서 볼 수 있게해주세요.
    - 결과는 오직 JSON 데이터로 바로 변환 할 수 있게 주세요.

    오직 형식대로만 답변하세요.
    결과는 반드시 **JSON 형식**으로 출력하고,
    **설명은 마크다운 형식**으로 작성해 주세요.
    '''
    )   

    return final_prompt