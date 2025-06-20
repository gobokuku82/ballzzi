import random
import json

def generate_non_football_dataset():
    # 축구와 무관한 질문 템플릿 확장
    non_football_templates = [
        {
            "question": "오늘 날씨 어때?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 포함하고 있어 날씨 정보는 제공할 수 없습니다."
        },
        {
            "question": "주식 시장 전망이 어떻게 되나요?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 주식 시장 관련 정보는 포함하지 않습니다."
        },
        {
            "question": "가장 가까운 식당 추천해줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 포함하고 있어 식당 추천은 제공할 수 없습니다."
        },
        {
            "question": "내일 기차 시간표 알려줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 데이터만을 포함하고 있어 기차 시간표 정보는 제공할 수 없습니다."
        },
        {
            "question": "비트코인 시세는 어떻게 되나요?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 암호화폐 시세 정보는 포함하지 않습니다."
        },
        {
            "question": "근처 영화관에서 상영하는 영화 알려줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 포함하고 있어 영화 상영 정보는 제공할 수 없습니다."
        },
        {
            "question": "오늘 저녁 뭐 먹을까?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 음식 추천은 제공할 수 없습니다."
        },
        {
            "question": "서울에서 부산까지 가는 KTX 예약하고 싶어",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 포함하고 있어 기차 예약 서비스는 제공할 수 없습니다."
        },
        {
            "question": "요즘 인기있는 드라마 추천해줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 드라마 추천은 제공할 수 없습니다."
        },
        {
            "question": "내일 미세먼지 농도는 어떻게 될까?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 데이터만을 포함하고 있어 대기질 정보는 제공할 수 없습니다."
        },
        {
            "question": "주변 맛집 추천해줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 맛집 추천은 제공할 수 없습니다."
        },
        {
            "question": "다음 주 로또 번호 예측해줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 포함하고 있어 로또 번호 예측은 제공할 수 없습니다."
        },
        {
            "question": "요가 초보자를 위한 동작 알려줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 요가 동작 안내는 제공할 수 없습니다."
        },
        {
            "question": "스페인어로 '안녕하세요'는 뭐야?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 포함하고 있어 언어 번역 서비스는 제공할 수 없습니다."
        },
        {
            "question": "내일 면접 볼 때 뭐 입으면 좋을까?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 패션 조언은 제공할 수 없습니다."
        },
        {
            "question": "주식 투자 조언 부탁해",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 투자 조언은 제공할 수 없습니다."
        },
        {
            "question": "이번 주말 여행지 추천해줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 여행지 추천은 제공할 수 없습니다."
        },
        {
            "question": "다이어트 식단 좀 알려줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 식단 추천은 제공할 수 없습니다."
        },
        {
            "question": "영어 회화 학습 방법 알려줘",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 언어 학습 방법은 제공할 수 없습니다."
        },
        {
            "question": "오늘 주식 시장 어떻게 됐어?",
            "response": "죄송합니다. 이 데이터베이스는 축구 선수 정보만을 다루고 있어 주식 시장 정보는 제공할 수 없습니다."
        }
    ]
    
    non_football_dataset = []
    for template in non_football_templates:
        non_football_dataset.append({

            "input": template["question"],
            "output": template["response"]
        })
    
    return non_football_dataset

def generate_training_dataset():
    # 국가명 매핑 (한글-영문)
    country_mapping = {
        "대한민국": "Korea Republic",
        "한국": "Korea Republic",
        "일본": "Japan",
        "중국": "China PR",
        "독일": "Germany",
        "프랑스": "France",
        "스페인": "Spain",
        "잉글랜드": "England",
        "영국": "England",
        "이탈리아": "Italy",
        "브라질": "Brazil",
        "아르헨티나": "Argentina",
        "포르투갈": "Portugal",
        "네덜란드": "Netherlands",
        "벨기에": "Belgium"
    }

    # 능력치 표현 매핑 확장
    ability_expressions = {
        "최고의": 18,
        "매우 뛰어난": 17,
        "뛰어난": 16,
        "우수한": 15,
        "좋은": 14,
        "평균 이상의": 13,
        "괜찮은": 12,
        "보통의": 10
    }

       # 능력치 자연어 변형
    ability_variations = {
        "반사신경": ["반사신경이 {level}", "순발력이 {level}", "반응속도가 {level}"],
        "페널티_방어": ["페널티 방어를 잘하는", "페널티킥 상황에서 강한", "승부차기에 강한"],
        "태클": ["태클이 {level}", "수비 태클이 {level}", "태클 능력이 {level}"],
        "대인방어": ["대인방어가 {level}", "1대1 수비가 {level}", "맨투맨 수비가 {level}"],
        "헤딩": ["헤딩이 {level}", "공중볼 경합이 {level}", "헤딩 능력이 {level}"],
        "패스": ["패스가 {level}", "패스 능력이 {level}", "볼 배급이 {level}"],
        "시야": ["시야가 {level}", "경기 읽는 능력이 {level}", "경기 분석력이 {level}"],
        "드리블": ["드리블이 {level}", "볼 컨트롤이 {level}", "개인기가 {level}"],
        "슛": ["슛이 {level}", "슈팅이 {level}", "득점력이 {level}"],
        "침착성": ["침착성이 {level}", "결정력이 {level}", "마무리가 {level}"],
        # 누락된 능력치들 추가
        "체력": ["체력이 {level}", "스태미나가 {level}", "지구력이 {level}"],
        "수비력": ["수비력이 {level}", "수비 능력이 {level}", "수비 실력이 {level}"],
        "속도": ["속도가 {level}", "스피드가 {level}", "발이 빠른"],
        "잠재력": ["잠재력이 {level}", "성장 가능성이 {level}", "발전 가능성이 {level}"],
        "현재_능력": ["실력이 {level}", "현재 기량이 {level}", "능력이 {level}"]
    }

    # 질문 템플릿 변형
    question_templates = [
        "{nationality} 선수 중에서 {ability} 선수 추천해줘",
        "{nationality} 출신으로 {ability} 선수 있을까?",
        "{ability} {nationality} 선수를 찾고 있어",
        "{nationality} 선수들 중 {ability} 선수 알려줄래?",
        "{nationality}에서 {ability} 선수를 뽑아줘"
    ]

    # 포지션별 특화 템플릿
    position_templates = {
        "GK": [
            {
                "abilities": ["반사신경", "페널티_방어"],
                "table": "GK"
            }
        ],
        "D": [
            {
                "abilities": ["태클", "대인방어", "헤딩"],
                "table": "D"
            }
        ],
        "M": [
            {
                "abilities": ["패스", "시야", "체력"],
                "table": "M"
            }
        ],
        "AM": [
            {
                "abilities": ["드리블", "패스", "시야"],
                "table": "AM"
            }
        ],
        "ST": [
            {
                "abilities": ["슛", "침착성", "드리블"],
                "table": "ST"
            }
        ]
    }

    # 복합 능력치 템플릿
    complex_templates = [
        {
            "questions": [
                "전방 압박을 잘하는 공격수를 찾고 있어",
                "압박 축구에 적합한 공격수 추천해줘",
                "높은 위치에서 수비가 가능한 공격수 알려줘"
            ],
            "query": """
                SELECT 이름, 소속팀, 체력, 수비력, 현재_능력
                FROM ST
                WHERE 체력 >= {ability_value} AND 수비력 >= {ability_value}
                ORDER BY (체력 + 수비력) DESC
                LIMIT 5;
            """
        },
        {
            "questions": [
                "빌드업에 능한 수비수 찾아줘",
                "공격 전개가 가능한 수비수 추천해줘",
                "공격 가담이 좋은 수비수를 알려줘"
            ],
            "query": """
                SELECT 이름, 소속팀, 패스, 시야, 현재_능력
                FROM D
                WHERE 패스 >= {ability_value} AND 시야 >= {ability_value}
                ORDER BY (패스 + 시야) DESC
                LIMIT 5;
            """
        }
    ]
        # 시장 가치 비교 템플릿
    value_comparison_templates = [
        {
            "questions": [
                "{position}에서 {ability} 능력이 비슷한 선수들 중에 시장 가치가 낮은 선수 찾아줘",
                "{ability} 수준이 비슷한 {position} 중에서 시장 가치가 저렴한 선수 추천해줘",
                "다른 {position}들과 {ability} 능력은 비슷하지만 시장 가치가 낮은 선수 알려줘"
            ],
            "query": """
                WITH similar_players AS (
                    SELECT 이름, 소속팀, {ability_column}, 시장_가치, 현재_능력,
                           ABS({ability_column} - (
                               SELECT AVG({ability_column})
                               FROM {table}
                               WHERE {ability_column} >= {ability_value}
                           )) as ability_diff
                    FROM {table}
                    WHERE {ability_column} >= {ability_value}
                )
                SELECT 이름, 소속팀, {ability_column}, 시장_가치, 현재_능력
                FROM similar_players
                WHERE ability_diff <= 2
                ORDER BY 시장_가치 ASC
                LIMIT 5;
            """
        },
        {
            "questions": [
                "{nationality} {position} 중에서 {ability} 능력은 좋은데 시장 가치가 낮은 선수 찾아줘",
                "{nationality}의 {position}들 중에서 {ability} 수준은 비슷하고 시장 가치가 낮은 선수 추천해줘",
                "{nationality} 출신 {position} 중에서 {ability} 능력치는 비슷한데 시장 가치가 저렴한 선수 있어?"
            ],
            "query": """
                WITH similar_players AS (
                    SELECT 이름, 소속팀, {ability_column}, 시장_가치, 현재_능력,
                           ABS({ability_column} - (
                               SELECT AVG({ability_column})
                               FROM {table}
                               WHERE {ability_column} >= {ability_value}
                               AND 국적 LIKE '%{nationality}%'
                           )) as ability_diff
                    FROM {table}
                    WHERE {ability_column} >= {ability_value}
                    AND 국적 LIKE '%{nationality}%'
                )
                SELECT 이름, 소속팀, {ability_column}, 시장_가치, 현재_능력
                FROM similar_players
                WHERE ability_diff <= 2
                ORDER BY 시장_가치 ASC
                LIMIT 5;
            """
        },
        {
            "questions": [
                "{age}세 이하 {position} 중에서 {ability} 능력은 비슷한데 시장 가치가 낮은 선수 찾아줘",
                "젊은 {position} 중에서 {ability} 수준은 비슷하고 시장 가치가 저렴한 선수 추천해줘",
                "{age}세 미만의 {position}들 중에서 {ability} 능력치는 비슷한데 시장 가치가 낮은 선수 있어?"
            ],
            "query": """
                WITH similar_players AS (
                    SELECT 이름, 소속팀, 나이, {ability_column}, 시장_가치, 현재_능력,
                           ABS({ability_column} - (
                               SELECT AVG({ability_column})
                               FROM {table}
                               WHERE {ability_column} >= {ability_value}
                               AND 나이 <= {age}
                           )) as ability_diff
                    FROM {table}
                    WHERE {ability_column} >= {ability_value}
                    AND 나이 <= {age}
                )
                SELECT 이름, 소속팀, 나이, {ability_column}, 시장_가치, 현재_능력
                FROM similar_players
                WHERE ability_diff <= 2
                ORDER BY 시장_가치 ASC
                LIMIT 5;
            """
        }
    ]

    dataset = []

    # 시장 가치 비교 템플릿 데이터 생성
    for position, templates in position_templates.items():
        position_kr = {
            "GK": "골키퍼",
            "D": "수비수",
            "M": "미드필더",
            "AM": "공격형 미드필더",
            "ST": "공격수"
        }[position]
        
        for template in templates:
            for ability in template["abilities"]:
                # 기본 시장 가치 비교
                for question in value_comparison_templates[0]["questions"]:
                    for level, value in ability_expressions.items():
                        ability_expr = ability_variations[ability][0].format(level=level)
                        question_filled = question.format(
                            position=position_kr,
                            ability=ability_expr
                        )
                        query = value_comparison_templates[0]["query"].format(
                            ability_column=ability,
                            table=template["table"],
                            ability_value=value
                        )
                        dataset.append({
                            "input": question_filled,
                            "output": query.strip()
                        })

                # 국적 기반 시장 가치 비교
                for question in value_comparison_templates[1]["questions"]:
                    for nationality_kor, nationality_eng in country_mapping.items():
                        for level, value in ability_expressions.items():
                            ability_expr = ability_variations[ability][0].format(level=level)
                            question_filled = question.format(
                                nationality=nationality_kor,
                                position=position_kr,
                                ability=ability_expr
                            )
                            query = value_comparison_templates[1]["query"].format(
                                ability_column=ability,
                                table=template["table"],
                                ability_value=value,
                                nationality=nationality_eng
                            )
                            dataset.append({
                                "input": question_filled,
                                "output": query.strip()
                            })

                # 나이 기반 시장 가치 비교
                ages = [21, 23, 25, 27]  # 대표적인 나이 기준점
                for question in value_comparison_templates[2]["questions"]:
                    for age in ages:
                        for level, value in ability_expressions.items():
                            ability_expr = ability_variations[ability][0].format(level=level)
                            question_filled = question.format(
                                age=age,
                                position=position_kr,
                                ability=ability_expr
                            )
                            query = value_comparison_templates[2]["query"].format(
                                ability_column=ability,
                                table=template["table"],
                                ability_value=value,
                                age=age
                            )
                            dataset.append({
                                "input": question_filled,
                                "output": query.strip()
                            })
    # 포지션별 기본 템플릿 데이터 생성
    for position, templates in position_templates.items():
        for template in templates:
            for ability in template["abilities"]:
                for nationality_kor, nationality_eng in country_mapping.items():
                    for level, value in ability_expressions.items():
                        for ability_variation in ability_variations[ability]:
                            for question_template in question_templates:
                                ability_expr = ability_variation.format(level=level)
                                question = question_template.format(
                                    nationality=nationality_kor,
                                    ability=ability_expr
                                )
                                
                                query = f"""
                                    SELECT 이름, 소속팀, {ability}, 현재_능력
                                    FROM {template['table']}
                                    WHERE 국적 LIKE '%{nationality_eng}%'
                                    AND {ability} >= {value}
                                    ORDER BY {ability} DESC
                                    LIMIT 5;
                                """
                                
                                dataset.append({
                                    "input": question,
                                    "output": query.strip()
                                })

    # 복합 능력치 템플릿 데이터 생성
    for template in complex_templates:
        for level, value in ability_expressions.items():
            for question in template["questions"]:
                dataset.append({
                    "input": question,
                    "output": template["query"].format(ability_value=value).strip()
                })

    # non-football 데이터 추가
    dataset.extend(generate_non_football_dataset())

     # JSON 파일로 저장 후 예시 출력하는 부분 수정
    with open('training_data.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"총 {len(dataset)}개의 학습 데이터가 생성되었습니다.")
    
    # 축구 관련 예시 출력
    football_examples = [d for d in dataset if "SELECT" in d['output']]
    print("\n축구 관련 예시 데이터:")
    for i in range(min(3, len(football_examples))):
        print(f"\n예시 {i+1}:")
        print(f"질문: {football_examples[i]['input']}")
        print(f"쿼리: {football_examples[i]['output']}")
    
    # 축구 무관 예시 출력
    non_football_examples = [d for d in dataset if "SELECT" not in d['output']]
    print("\n축구 무관 예시 데이터:")
    for i in range(min(3, len(non_football_examples))):
        print(f"\n예시 {i+1}:")
        print(f"질문: {non_football_examples[i]['input']}")
        print(f"응답: {non_football_examples[i]['output']}")

if __name__ == "__main__":
    generate_training_dataset()