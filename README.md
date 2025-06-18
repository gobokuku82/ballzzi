# ⚽ Ballzzi -RAG 기반 데이터 통합 관리 챗봇
  - 축구선수 분석 및 & 회사 내부정보 기반 LLM시스템
   ```
    8400명의 축구선수 데이터 - 선수이미지를 포함한 세부데이터 검색 및 분석기능
    8개챕터의 모든 사내규정 - 출장비 및 휴가규정 조회 및 설명기능
    변동된 모든 인사정보 - 실시간 인사정보 및 조직도 업데이트 조회기능
    메세지 입력칸 변경없이 모든 정보를 하나의 메세지창에서 조회가능 
  ```
## 👥 팀 소개

<table>
  <tr>
    <td align="center">
      <img src="./img/스크린샷 2025-06-05 095502.png" width="120px"><br/>
      <b>김도윤</b>
    </td>
    <td align="center">
      <img src="./img/스크린샷 2025-06-05 095123.png" width="120px"><br/>
      <b>최요섭</b>
    </td>
    <td align="center">
      <img src="./img/스크린샷 2025-06-05 095207.png" width="120px"><br/>
      <b>김재현</b>
    </td>
    <td align="center">
      <img src="./img/스크린샷 2025-06-05 095306.png" width="120px"><br/>
      <b>이석원</b>
    </td>
    <td align="center">
      <img src="./img/스크린샷 2025-06-05 095415.png" width="120px"><br/>
      <b>윤권</b>
    </td>
  </tr>
</table>

---

<div align="center"><h1>📚 STACKS</h1></div>

<div align="center"> 
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
  <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"> 
  <img src="https://img.shields.io/badge/sqlite-003B57?style=for-the-badge&logo=sqlite&logoColor=white"> 
  <img src="https://img.shields.io/badge/faiss-0099CC?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjMDA5OUM4IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4=&logoColor=white" alt="faiss"> 
  <img src="https://img.shields.io/badge/openai-412991?style=for-the-badge&logo=openai&logoColor=white"> 
  <img src="https://img.shields.io/badge/langchain-4BAF50?style=for-the-badge"> 
  <img src="https://img.shields.io/badge/huggingface-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black">
  <br>
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"> 
  <img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
</div>


## I. 프로젝트 개요
### 1. 목적
 - 다양한 형태의 데이터와 LLM 연동하여 웹UI시각화
 - 테이블데이터,혼합형데이터(글,숫자,표)를 각각 정형DB와 벡터DB로 구축하여 LLM와 연동
 - 정형DB : 테이블데이터
 - 벡터DB : 혼합형데이터(글,숫자,표) / 테이블데이터 / 특문구조형문서
 ```
이중 도메인 시스템: FM(축구) + HR(회사 규정/인사)
자동 질문 분류 및 모듈 라우팅
Streamlit 기반 UI / LangChain 기반 Agent
LLM: GPT-4o-mini, 임베딩: KURE-v1, Reranker: bge-reranker-v2-m3-ko
데이터: FAISS Vector DB + SQLite
```
## II. 시스템 아키텍처 
### 작동 흐름 요약
```
FM/HR 분류기 : 사용자 입력 → classify() → 도메인 판단
├── FM : "soccer" → get_answer_from_question() → SQL + LLM + 이미지
└── HR : "company" → process_query() → LangChain Agent → hybrid search → GPT
```
### 1. 전체 프로세스 흐름도 (Main Flowchart)
![image](https://github.com/user-attachments/assets/56d13dfc-d8a1-4b34-83a7-3c72e48df6ba)

### 2. 세부 프로세스 흐름도 - 질문 자동 분류 (Sub Flowchart - Automated Question Classification)
<img src="https://github.com/user-attachments/assets/940c4d10-adc9-4726-92d0-d81fe09f1106" width="100%"></td>


## III. 주요 기능

### 1. 질문 자동 분류 (Question Routing)
- **모델**: `sentence-transformers/all-MiniLM-L6-v2`
- **기술**: FAISS 기반 벡터 유사도 검색
- **분류**: 'soccer' (축구 관련) / 'company' (회사 관련)

### 2. FM 모듈 (축구 선수 정보)
- **자연어 → SQL 변환**: GPT-4o-mini 모델 사용
- **데이터베이스**: SQLite (players_position.db)
- **이미지 검색**: Bing 이미지 크롤링 (Selenium)
- **응답 형식**: JSON 배열로 구조화된 선수 정보

### 3. HR 모듈 (회사 정보)
- **RAG 시스템**: FAISS 벡터 데이터베이스 활용
- **다중 검색 도구**: 
  - 통합 검색 (내부+외부)
  - 회사 전용 검색
  - 네이버 뉴스/웹 검색
  - 퇴직금 계산기
- **LLM 지원**: OpenAI GPT 및 HyperCLOVAX

## IV. 디렉토리 구조

```
AJR/
├── app.py                          # 메인 Streamlit 애플리케이션
├── question_Routing.py             # 질문 분류 시스템
├── requirements.txt                # Python 패키지 의존성
│  
├── FM/                           # Football Manager 모듈
│   ├── FM_GetData_LLM.py         # 메인 축구 정보 처리
│   ├── tools/
│   │   ├── create_prompt.py      # SQL/자연어 프롬프트 생성
│   │   ├── SQL_create.py         # SQL 프롬프트 템플릿 반환
│   │   ├── SQL_execute.py        # SQL 프롬프트 템플릿 반환
│   │   └── image_craper.py       # Bing "BING"에서 이미지 크롤링
│   └── data/
│       └── players_position.db   # 축구 선수 데이터베이스 ( 정형DB )
│
└── HR/                           # Human Resources 모듈
    ├── agents/
    │   └── agent_executor.py     # Langchain Agent 실행기
    ├── tools/
    │   └── rag_tool.py          # RAG 도구 모음  
    └── data/                    
        ├── faiss_win/           # 사내 규정 FAISS 인덱스 ( 벡터DB )
        └── faiss_org_hr/        # 인사 구조 FAISS 인덱스 ( 벡터DB )
```

```
- **LangChain**: 4개 패키지 (core, community, openai, experimental)
- **OpenAI GPT-4o-mini**: 메인 언어 모델
- **HyperCLOVAX**: 대체 언어 모델 지원
- **Sentence Transformers**: 질문 분류용 임베딩
- **FAISS**: 벡터 유사도 검색
- **Transformers**: 허깅페이스 모델 지원

### 웹 & 데이터베이스
- **Streamlit**: 웹 인터페이스
- **SQLite**: 축구 선수 데이터베이스
- **SQLAlchemy**: 데이터베이스 ORM

### 웹 크롤링 & 이미지
- **Selenium**: 웹 브라우저 자동화
- **BeautifulSoup4**: HTML 파싱
- **Pillow**: 이미지 처리
- **Requests**: HTTP 요청

### 데이터 분석
- **Pandas**: 데이터 프레임 조작
- **NumPy**: 수치 연산
- **Scikit-learn**: 머신러닝 도구
```

## V. 설치 및 실행

### 1. 환경 설정
```
```bash
# 저장소 클론
git clone [repository-url](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN12-3RD-4TEAM.git

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정
```
`.env` 파일 생성:
```env
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here
# HuggingFace 토큰
HUGGINGFACE_TOKEN=your_huggingface_token_here
# 네이버 검색 API
NAVER_CLIENT_ID=your_naver_id_here
NAVER_CLIENT_SECRET=your_naver_token_here
# 모델 설정
LLM_MODEL=gpt-4o-mini
# RERANKER_MODEL=BAAI/bge-reranker-v2-m3 # 사용하지 않는 모델은 주석처리
LLM_MODEL=gpt-4o-mini # 사용하는 모델설정
# 벡터 데이터베이스 설정
VECTOR_DB_PATH=HR/data/faiss_win
VECTOR_DB_PATH_HR=HR/data/faiss_org_hr
```
### 3. 애플리케이션 실행

```bash
streamlit run app.py
```

## :bulb: 사용 방법

### 축구 관련 질문 예시
- "리오넬 메시에 대해 알려줘"
- "호날두의 커리어는 어때?"
- "손흥민은 어떤 팀에서 뛰고 있어?"
- "10000000 예산 안에서 영입가능한 수비수"

### 회사 관련 질문 예시
- "연차는 어떻게 써요?"
- "복지 포인트는 어디서 확인해?"
- "퇴사하려면 뭐부터 해야 해?"
- "재택근무 신청은 어디서 합니까?"

## :mag: 핵심 구성 요소

### 1. 질문 라우팅 시스템 (`question_Routing.py`)
- 60개의 예시 문장으로 학습된 분류기
- FAISS 인덱스를 통한 빠른 유사도 검색
- 축구/회사 도메인 자동 분류

### 2. FM 모듈 처리 흐름
```
사용자 질문 → SQL 생성 → 데이터베이스 쿼리 → 자연어 응답 생성 → 이미지 검색
```

### 3. HR 모듈 도구 우선순위
1. `hybrid_search`: 통합 검색 (내부+외부)
2. `search_company_info`: 회사 전용 검색
3. `calculate_retirement_pay`: 퇴직금 계산
4. `search_naver_news`: 네이버 뉴스
5. `search_documents`: 기존 문서 검색
6. `search_naver_web`: 네이버 웹 검색

## :dart: 모델 및 API 정보

### 주요 모델
- **GPT-4o-mini**: 메인 언어 모델 (OpenAI)
- **all-MiniLM-L6-v2**: 문장 임베딩 (Sentence Transformers)
- **HyperCLOVAX**: 대체 언어 모델 (네이버)

### 데이터베이스
- **축구 선수 DB**: 788KB SQLite 파일
- **HR 벡터 DB**: FAISS 인덱스 (사내 규정/인사 구조)

## :wrench: 개발 도구

### 디버깅 & 테스트
- Verbose 모드로 LangChain 에이전트 실행 로그 확인

### 확장성
- 새로운 LLM 모델 쉽게 추가 가능
- 도구 우선순위 시스템으로 기능 확장 용이
- 모듈식 구조로 독립적 개발 가능

## :chart_with_upwards_trend: 성능 최적화

- **캐싱**: Streamlit 세션 상태 활용
- **병렬 처리**: FAISS 벡터 검색 최적화
- **메모리 관리**: PyTorch 클래스 감시 제외
- **이미지 최적화**: PIL 이미지 처리 및 압축

## :rotating_light: 주의사항

1. **API 키 필수**: OpenAI API 키 없이는 LLM 기능 사용 불가
2. **크롬 드라이버**: 이미지 크롤링을 위한 자동 설치
3. **메모리 사용량**: 대용량 모델 로드 시 충분한 RAM 필요
4. **네트워크 의존성**: 실시간 이미지/뉴스 검색 기능

---

**개발자**: Ballzzi Team  
**최종 업데이트**: 2025년 06월 04일

```
