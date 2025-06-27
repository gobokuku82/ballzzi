

# **⚽ GOLDENKICK 프로젝트-Ballzzi**<br/>&nbsp;&nbsp;-&nbsp;&nbsp;RAG 기반 LLM 챗봇&nbsp;&nbsp;-

&nbsp;&nbsp;***Our Customer's Victory is Our Victory***

---

</div>


## 👥 팀 소개

<table>
  <tr>
    <td align="center">
      <img src="./static/icons/ballzzi.png" width="120px"><br/>
      <b>김도윤</b><br/><span style="font-size:14px;">프론트 팀장</sub>
    </td>
    <td align="center">
      <img src="./static/icons/shame_ballzzi.png" width="120px"><br/>
      <b>최요섭</b><br/><span style="font-size:14px;">데이터 팀장</sub>
    </td>
    <td align="center">
      <img src="./static/icons/happy_ballzzi.png" width="120px"><br/>
      <b>김재현</b><br/><span style="font-size:14px;">백엔드 팀장</sub>
    </td>
    <td align="center">
      <img src="./static/icons/sad_ballzzi.png" width="120px"><br/>
      <b>이석원</b><br/><span style="font-size:14px;"> P M </sub>
    </td>
    <td align="center">
      <img src="./static/icons/evil_ballzzi.png" width="120px"><br/>
      <b>윤권</b><br/><span style="font-size:14px;">인프라 팀장</sub>
    </td>
  </tr>
</table>


<div style="border: 1px solid #ccc; border-radius: 10px; padding: 20px; background-color: #f9f9f9">
  <h3>📌 팀 역할 및 기여 내용</h3>
  <p>
    본 프로젝트는 전 팀원이 함께 기획, 개발, 디자인 전반에 걸쳐 협력하여 진행하였습니다.<br/>
    모두가 기획 아이디어를 제안하고, 코드를 작성하며, 디자인 요소에도 의견을 나누는 등<br/>
    역할의 경계를 두지 않고 적극적으로 참여하였습니다.<br/><br/>
    특히 프론트엔드와 백엔드, 챗봇 구성 등 여러 파트를 유연하게 넘나들며<br/>
    서로의 업무를 지원하고 피드백을 나눈 점이 본 팀의 강점이었습니다.
  </p>
</div>
</div><h1>📚 STACKS</h1></div>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-4285F4?style=for-the-badge&logo=meta&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge&logoColor=black)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)
![KURE](https://img.shields.io/badge/KURE--v1-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white)
![BGE Reranker](https://img.shields.io/badge/BGE_Reranker--v2--m3-4ECDC4?style=for-the-badge&logo=huggingface&logoColor=white)
![Google OAuth](https://img.shields.io/badge/Google_OAuth-4285F4?style=for-the-badge&logo=google&logoColor=white)

![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud_Deploy-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
</div>

## I. 프로젝트 개요

**GOLDENKICK**프로젝트는 Ballzzi라는 케릭터를 활용한 'Ballzzi봇'-LLM챗봇-으로, RAG 기반의 langchain agent로 구성했습니다. 8,000명 이상의 축구 선수 데이터베이스와 인사정보 및 내부규정을 검색가능한 RAG 기반 LLM 챗봇 시스템으로, 정형DB인 축구선수 데이터와 인사정보 및 내부규정는 벡터DB로 구성하였습니다. 이러한 기능은 하나의 채팅창에서 이루어집니다.

- **프로젝트명**: GOLDENKICK (Ballzzi)
- **기업형태**: 스포츠 에이전시 전문 기업
- **핵심기술**: Django + LangChain + FAISS + Sentence Transformers + SQLite
- **서비스분야**: 축구 선수 스카우팅, 매칭, 인사정보,내부규정 검색

## II. 시스템 아키텍처
<img src="./산출물/4. 시스템 구성도.png" style="width:100%; max-width:1000px;">

## III. 주요 기능
### 1. AI 챗봇 시스템
- **이중 도메인 분류**: 축구 선수 정보 vs 인사업무 질문 자동 구분
- **RAG 기반 검색**: FAISS 벡터 데이터베이스 활용한 정확한 답변
- **실시간 이미지 크롤링**: Bing 검색 API를 통한 선수 이미지 자동 수집
- **멀티모달 응답**: 텍스트 정보 + 이미지 + 구조화된 데이터 제공

### 2. 축구 선수 정보 시스템 (FM 모듈)
- **대규모 선수 DB**: 8,000명+ 축구 선수 데이터베이스
- **SQL 쿼리 자동 생성**: 자연어 질문을 SQL로 변환
- **실시간 데이터 분석**: 선수 능력치, 포지션, 시장가치 등 종합 분석
- **이미지 매칭**: 선수명 기반 자동 이미지 검색 및 매칭

### 3. HR 업무 자동화 시스템 (HR 모듈)
- **통합 검색 시스템**: 내부 문서 + 외부 정보 하이브리드 검색
- **퇴직금 자동 계산**: 근무 기간별 퇴직금 자동 산출
- **네이버 API 연동**: 실시간 뉴스 및 웹 검색 결과 제공
- **다중 RAG 도구**: 상황별 최적화된 검색 도구 자동 선택

### 4. 사용자 인증 시스템
- **Django Allauth**: 회원가입, 로그인, 로그아웃 시스템
- **Google OAuth**: 구글 소셜 로그인 지원
- **세션 관리**: 로그인 상태 유지 및 보안 강화

## IV. 기술 스택

### Backend Framework
```
Django==5.2                 # 웹 프레임워크
django-allauth==65.9.0      # 사용자 인증
python-dotenv==1.1.1        # 환경변수 관리
```

### AI/ML Stack
```
langchain==0.3.26           # LLM 체인 관리
langchain-community==0.3.26 # 커뮤니티 도구
langchain-openai==0.3.25    # OpenAI 연동
sentence-transformers==4.1.0 # 문장 임베딩
torch==2.7.1                # 딥러닝 프레임워크
faiss-cpu==1.11.0           # 벡터 검색 엔진
transformers==4.52.4        # 트랜스포머 모델
```

### Web Crawling & Data Processing
```
selenium==4.33.0            # 웹 크롤링
chromedriver-autoinstaller==0.6.4 # 크롬 드라이버 자동 설치
requests==2.32.4            # HTTP 요청
Pillow==11.2.1              # 이미지 처리
```

### Database & Scientific Computing
```
SQLAlchemy==2.0.41          # SQL 도구
numpy==2.2.6                # 수치 계산
scikit-learn==1.7.0         # 머신러닝
scipy==1.15.3               # 과학 계산
```

## V. 프로젝트 구조

```
ballzzi_work/
├── 📄 manage.py                    # Django 관리 스크립트
├── 📄 requirements.txt             # Python 의존성 패키지
├── 📄 styles.css                  # 전역 스타일시트
│
├── 📁 proj4/                      # Django 프로젝트 설정
│   ├── __init__.py
│   ├── settings.py                # Django 설정 파일
│   ├── urls.py                    # 메인 URL 라우팅
│   ├── wsgi.py                    # WSGI 설정
│   └── asgi.py                    # ASGI 설정
│
├── 📁 myapp/                      # 메인 Django 앱
│   ├── __init__.py
│   ├── admin.py                   # Django 관리자 설정
│   ├── apps.py                    # 앱 설정
│   ├── models.py                  # 데이터 모델 (현재 미사용)
│   ├── views.py                   # 뷰 로직 (메인 비즈니스 로직)
│   ├── urls.py                    # 앱 URL 라우팅
│   ├── tests.py                   # 테스트 파일
│   ├── 📁 migrations/             # 데이터베이스 마이그레이션
│   │   └── __init__.py
│   └── 📁 source/                 # 핵심 AI 모듈 (가동파일)
│       ├── __init__.py
│       ├── question_Routing.py    # 질문 분류 시스템
│       │
│       ├── 📁 FM/                 # 축구 선수 정보 모듈
│       │   ├── __init__.py
│       │   ├── FM_GetData_LLM.py  # 메인 FM 처리 로직
│       │   ├── 📁 data/           # 선수 데이터베이스
│       │   │   └── players_position.db # SQLite 데이터베이스
│       │   └── 📁 tools/          # FM 도구 모음
│       │       ├── __init__.py
│       │       ├── create_prompt.py      # 프롬프트 생성
│       │       ├── image_craper.py       # 이미지 크롤링
│       │       ├── SQL_create.py         # SQL 쿼리 생성
│       │       └── SQL_execute.py        # SQL 실행
│       │
│       └── 📁 HR/                 # 인사업무 자동화 모듈
│           ├── __init__.py
│           ├── 📁 agents/         # AI 에이전트 시스템
│           │   ├── __init__.py
│           │   └── agent_executor.py     # langchaing agent 실행기
│           ├── 📁 data/           # RAG 벡터 데이터베이스
│           │   ├── 📁 faiss_org_hr/     # 인사정보 벡터DB
│           │   │   ├── index.faiss
│           │   │   └── index.pkl
│           │   └── 📁 faiss_win/        # 내부규정 벡터DB
│           │       ├── index.faiss
│           │       └── index.pkl
│           └── 📁 tools/          # HR 도구 모음
│               ├── __init__.py
│               └── rag_tool.py           # RAG agent tool
│
├── 📁 templates/                  # Django 템플릿
│   ├── base.html                  # 기본 템플릿
│   ├── 📁 myapp/                  # myapp 전용 템플릿
│   │   ├── index.html             # 메인 홈페이지
│   │   └── chatbot.html           # 챗봇 페이지
│   ├── 📁 account/                # 계정 관리 템플릿
│   │   ├── login.html
│   │   ├── logout.html
│   │   └── signup.html
│   └── 📁 socialaccount/          # 소셜 로그인 템플릿
│       └── login.html
│
└── 📁 static/                     # 정적 파일
    ├── styles.css                 # 전역 스타일
    ├── 📁 css/                    # 스타일시트
    │   ├── styles.css
    │   └── login_style.css
    ├── 📁 js/                     # JavaScript
    │   └── script.js              # 메인 스크립트
    ├── 📁 icons/                  # 아이콘 및 로고
    │   ├── ballzzi.png           # 메인 로고
    │   ├── google-logo.png       # 구글 로고
    │   ├── 1.png ~ 5.png         # index페이지 링크 아이콘
    │   └── *_ballzzi.png         # 감정별 마스코트 (6개)
    ├── 📁 profiles/               # index페이지 소스 이미지    
    └── 📁 video/                  # 동영상 파일
        └── Company_introduction.mp4 # 회사 소개 영상
```

## VI. 핵심 모듈 상세 분석

### 1. 질문 분류 시스템 (`question_Routing.py`)
```python
# 종합적인 분류 시스템
- Sentence Transformers 모델: "jhgan/ko-sroberta-multitask"
- FAISS 인덱스를 통한 유사도 기반 분류
- 축구 관련 질문 vs HR 질문 자동 구분
- 60개의 축구 예시 + 33개의 HR 예시 학습 데이터
```

### 2. FM 모듈 (축구 선수 정보)
#### 1) 메인 처리기 (`FM_GetData_LLM.py`)
```python
# LLM 기반 데이터 처리 시스템
- OpenAI GPT-4o-mini 모델 활용
- SQLite DB 연동 (players_position.db)
- 자연어 → SQL 쿼리 자동 변환
- JSON 형태의 구조화된 응답 생성
```

#### 2) 핵심 도구들
- **`create_prompt.py`** (64줄): SQL 및 최종 응답 프롬프트 생성
- **`image_craper.py`** (306줄): Selenium 기반 Bing 이미지 크롤링
- **`SQL_create.py`** (47줄): LLM 체인을 통한 SQL 쿼리 후처리
- **`SQL_execute.py`** (50줄): 안전한 SQL 실행 및 결과 처리

### 3. HR 모듈 (인사업무 자동화)
#### 1) 에이전트 실행기 (`agent_executor.py`)
```python
# 종합적인 AI 에이전트 시스템
- OpenAI Functions Agent 구조
- 6개의 전문 도구 우선순위별 활용:
  1. hybrid_search (통합 검색)
  2. search_company_info (회사 전용)
  3. calculate_retirement_pay (퇴직금 계산)
  4. search_naver_news (뉴스 검색)
  5. search_documents (문서 검색)
  6. search_naver_web (웹 검색)
```

#### 2) RAG 도구 (`rag_tool.py`)
```python
# 다기능 RAG 검색 시스템
- FAISS 벡터 데이터베이스 검색
- 네이버 뉴스/웹 API 연동
- 하이브리드 검색 (내부 + 외부)
- 퇴직금 자동 계산 기능
```

### 4. 웹 인터페이스
#### 1) Django Views (`views.py`)
```python
# 메인 비즈니스 로직
- 메인 페이지 렌더링 (타임스탬프 캐시 무효화)
- 챗봇 API 엔드포인트 (POST/GET 처리)
- 로그인 필수 인증 (@login_required)
- 질문 분류 → 모듈별 처리 → 응답 생성
```

#### 2) 프론트엔드 (`templates/myapp/index.html`)
```html
<!-- 현대적인 반응형 웹 페이지 -->
- 회사 소개 슬라이더 시스템
- 동영상 배경 및 애니메이션
- Google OAuth 소셜 로그인
- 모바일 반응형 네비게이션
```

#### 3) JavaScript (`static/js/script.js`)
```javascript
// 인터랙티브 기능
- 슬라이더 자동 재생 및 제어
- 부드러운 스크롤 애니메이션
- 햄버거 메뉴 토글
- 동적 UI 상태 관리
```

## VII. 설치 및 실행

### 1. 환경 설정
```bash
# 1. 프로젝트 클론
git clone [repository-url]
cd ballzzi_work

# 2. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정
`.env` 파일을 생성하고 다음 내용을 추가:
```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Naver API
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret

# LLM 설정
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0
MAX_NEW_TOKENS=1024
```

### 3. 데이터베이스 초기화
```bash
python manage.py migrate
python manage.py collectstatic
```

### 4. 서버 실행
```bash
python manage.py runserver
```

### 5. 접속 URL
- **메인 페이지**: http://ballzzi.duckdns.org:8000/
- **챗봇 서비스**: http://ballzzi.duckdns.org:8000/chatbot/

## VIII. 주요 기능 사용법

### 1. 축구 선수 정보 질문
```
질문 예시:
- "손흥민은 어떤 선수야?"
- "10000000 예산 안에서 영입가능한 수비수"
- "패스 정확도 높은 선수 누구야?"
- "EPL에서 떠오르는 유망주 있어?"

응답 형태:
- 선수명, 포지션, 능력치 등 구조화된 정보
- 실시간 크롤링을 통한 선수 이미지
- 관련 선수들의 비교 분석 데이터
```

### 2. HR 업무 질문
```
질문 예시:
- "연차는 어떻게 써요?"
- "퇴사하려면 뭐부터 해야 해?"
- "복지 포인트는 어디서 확인해?"
- "재택근무 신청은 어디서 합니까?"

응답 형태:
- 회사 내부 규정 기반 정확한 답변
- 절차 및 방법론 단계별 안내
- 관련 외부 정보 보완 제공
```

## IX. 프로젝트 요약

## 1. 성능 및 특징

### 1) 시스템 성능
- **데이터베이스**: 8,000+ 축구 선수 정보
- **벡터 검색**: FAISS 기반 고속 유사도 검색
- **응답 시간**: 평균 2-5초 (크롤링 포함)
- **동시 사용자**: Django 기본 설정 (확장 가능)

### 2) 핵심 특징
- **지능형 분류**: 질문 의도 자동 파악
- **멀티모달**: 텍스트 + 이미지 통합 응답
- **실시간 크롤링**: 최신 이미지 자동 수집
- **하이브리드 RAG**: 내부 + 외부 정보 통합
- **확장 가능**: 모듈식 구조로 기능 추가 용이

## 2. 보안 및 제한사항

### 1) 보안 기능
- Django CSRF 보호
- 사용자 인증 필수 (로그인 기반)
- 환경변수 기반 API 키 관리
- SQL Injection 방지

### 2) 사용 제한사항
- OpenAI API 사용량 제한
- 크롤링 속도 제한 (Bot 탐지 방지)
- 동시 접속자 수 제한 (서버 사양 의존)
- 데이터베이스 읽기 전용 (현재 설정)

---
## 한줄 회고록

<table>
  <tr>
    <td><b>김&nbsp;도&nbsp;윤</b></td>
    <td>3차에서는 RAG 시스템 구축을, 4차에서는 프론트 구조 및 특수기능 구현을 맡아 담당했습니다. 능력 있는 팀원들과의 협력 덕분에 각 단계에서 안정적으로 시스템을 완성할 수 있었습니다.</td>
  </tr>
  <tr>
    <td><b>최&nbsp;요&nbsp;섭</b></td>
    <td>지난 프로젝트에서는 역량을 많이 발휘하지 못했지만, 이번 팀 프로젝트를 통해 팀원들의 도움으로 부족한 부분을 많이 보완할 수 있었습니다. 특히 실제로 눈에 보이는 웹 개발을 직접 해보며 역량을 많이 발휘할 수 있었습니다.</td>
  </tr>
  <tr>
    <td><b>김&nbsp;재&nbsp;현</b></td>
    <td>3차와 같은 팀 구성으로 4차까지 함께하며, 점점 팀워크가 잘 맞아지는 걸 느꼈습니다. 진행하면서 스스로 문제를 해결하며 한 단계 성장하는 계기가 되었습니다.</td>
  </tr>
  <tr>
    <td><b>이&nbsp;석&nbsp;원</b></td>
    <td>3차와 4차 프로젝트를 같은 팀원들과 함께 진행하며, 협업 과정에서 다양한 의견을 접하며 시야를 넓힐 수 있었고, 팀원들과 함께 배우며 한층 더 성장할 수 있었습니다.</td>
  </tr>
  <tr>
    <td><b>윤&nbsp;&nbsp;&nbsp;&nbsp;권</b></td>
    <td>성능 개선을 목표로 파인튜닝을 장시간 시도했으나 기대치에는 미치지 못해 아쉬웠습니다. 하지만 클라우드 배포 경험은 큰 자산이 되었습니다.</td>
  </tr>
</table>
--

*최종 업데이트: 2025년 6월 27일*
