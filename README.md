# 🏆 GOLDENKICK (Ballzzi) - AI 기반 축구 선수 정보 & HR 통합 시스템

## 📋 프로젝트 개요

**GOLDENKICK**은 AI 기술을 활용한 스포츠 에이전시 전문 플랫폼으로, 8,000명 이상의 축구 선수 데이터베이스와 인사업무 자동화를 통합한 RAG 기반 LLM 챗봇 시스템입니다.

- **프로젝트명**: GOLDENKICK (Ballzzi)
- **기업형태**: 스포츠 에이전시 전문 기업
- **핵심기술**: Django + LangChain + FAISS + Sentence Transformers
- **서비스분야**: 축구 선수 스카우팅, 매칭, 인사업무 자동화

## 🚀 주요 기능

### 🤖 AI 챗봇 시스템
- **이중 도메인 분류**: 축구 선수 정보 vs 인사업무 질문 자동 구분
- **RAG 기반 검색**: FAISS 벡터 데이터베이스 활용한 정확한 답변
- **실시간 이미지 크롤링**: Bing 검색 API를 통한 선수 이미지 자동 수집
- **멀티모달 응답**: 텍스트 정보 + 이미지 + 구조화된 데이터 제공

### ⚽ 축구 선수 정보 시스템 (FM 모듈)
- **대규모 선수 DB**: 8,000명+ 축구 선수 데이터베이스
- **SQL 쿼리 자동 생성**: 자연어 질문을 SQL로 변환
- **실시간 데이터 분석**: 선수 능력치, 포지션, 시장가치 등 종합 분석
- **이미지 매칭**: 선수명 기반 자동 이미지 검색 및 매칭

### 🏢 HR 업무 자동화 시스템 (HR 모듈)
- **통합 검색 시스템**: 내부 문서 + 외부 정보 하이브리드 검색
- **퇴직금 자동 계산**: 근무 기간별 퇴직금 자동 산출
- **네이버 API 연동**: 실시간 뉴스 및 웹 검색 결과 제공
- **다중 RAG 도구**: 상황별 최적화된 검색 도구 자동 선택

### 🔐 사용자 인증 시스템
- **Django Allauth**: 회원가입, 로그인, 로그아웃 시스템
- **Google OAuth**: 구글 소셜 로그인 지원
- **세션 관리**: 로그인 상태 유지 및 보안 강화

## 🛠 기술 스택

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

## 📁 프로젝트 구조

```
ballzzi_work/
├── 📄 manage.py                    # Django 관리 스크립트
├── 📄 requirements.txt             # Python 의존성 패키지
├── 📄 SETUP_REPORT.md             # 설치 및 설정 보고서
├── 📄 CRAWLING_PATCH_NOTES.md     # 크롤링 개선 내역
├── 📄 HOMEPAGE_PATCH_NOTES.md     # 홈페이지 개선 내역
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
│   └── 📁 source/                 # 🔥 핵심 AI 모듈 (가동파일)
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
│           │   └── agent_executor.py     # 메인 에이전트 실행기
│           ├── 📁 data/           # RAG 벡터 데이터베이스
│           │   ├── 📁 faiss_org_hr/     # 조직 HR 벡터DB
│           │   │   ├── index.faiss
│           │   │   ├── index.pkl
│           │   │   └── 📁 index/
│           │   │       ├── index.faiss
│           │   │       └── index.pkl
│           │   └── 📁 faiss_win/        # Windows 환경 벡터DB
│           │       ├── index.faiss
│           │       ├── index.pkl
│           │       └── 📁 index/
│           │           ├── index.faiss
│           │           └── index.pkl
│           └── 📁 tools/          # HR 도구 모음
│               ├── __init__.py
│               └── rag_tool.py           # RAG 검색 도구 (17KB)
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
    │   └── script.js              # 메인 스크립트 (728줄)
    ├── 📁 icons/                  # 아이콘 및 로고
    │   ├── ballzzi.png           # 메인 로고
    │   ├── google-logo.png       # 구글 로고
    │   ├── 1.png ~ 5.png         # 서비스 아이콘
    │   └── *_ballzzi.png         # 감정별 마스코트 (6개)
    ├── 📁 profiles/               # 선수 프로필 이미지
    │   ├── 메시_프로필.jpg
    │   ├── 손흥민_프로필.jpg
    │   ├── 호날두_프로필.jpg
    │   ├── 클롭_프로필.jpg
    │   ├── 아스날_프로필.jpg
    │   └── *_짤.jpg              # 선수별 이미지
    └── 📁 video/                  # 동영상 파일
        └── Company_introduction.mp4 # 회사 소개 영상
```

## 🔧 핵심 모듈 상세 분석

### 1. 질문 분류 시스템 (`question_Routing.py`)
```python
# 148줄의 종합적인 분류 시스템
- Sentence Transformers 모델: "jhgan/ko-sroberta-multitask"
- FAISS 인덱스를 통한 유사도 기반 분류
- 축구 관련 질문 vs HR 질문 자동 구분
- 60개의 축구 예시 + 33개의 HR 예시 학습 데이터
```

### 2. FM 모듈 (축구 선수 정보)
#### 메인 처리기 (`FM_GetData_LLM.py`)
```python
# 92줄의 LLM 기반 데이터 처리 시스템
- OpenAI GPT-4o-mini 모델 활용
- SQLite DB 연동 (players_position.db)
- 자연어 → SQL 쿼리 자동 변환
- JSON 형태의 구조화된 응답 생성
```

#### 핵심 도구들
- **`create_prompt.py`** (64줄): SQL 및 최종 응답 프롬프트 생성
- **`image_craper.py`** (306줄): Selenium 기반 Bing 이미지 크롤링
- **`SQL_create.py`** (47줄): LLM 체인을 통한 SQL 쿼리 후처리
- **`SQL_execute.py`** (50줄): 안전한 SQL 실행 및 결과 처리

### 3. HR 모듈 (인사업무 자동화)
#### 에이전트 실행기 (`agent_executor.py`)
```python
# 213줄의 종합적인 AI 에이전트 시스템
- OpenAI Functions Agent 구조
- 6개의 전문 도구 우선순위별 활용:
  1. hybrid_search (통합 검색)
  2. search_company_info (회사 전용)
  3. calculate_retirement_pay (퇴직금 계산)
  4. search_naver_news (뉴스 검색)
  5. search_documents (문서 검색)
  6. search_naver_web (웹 검색)
```

#### RAG 도구 (`rag_tool.py`)
```python
# 383줄의 다기능 RAG 검색 시스템
- FAISS 벡터 데이터베이스 검색
- 네이버 뉴스/웹 API 연동
- 하이브리드 검색 (내부 + 외부)
- 퇴직금 자동 계산 기능
```

### 4. 웹 인터페이스
#### Django Views (`views.py`)
```python
# 90줄의 메인 비즈니스 로직
- 메인 페이지 렌더링 (타임스탬프 캐시 무효화)
- 챗봇 API 엔드포인트 (POST/GET 처리)
- 로그인 필수 인증 (@login_required)
- 질문 분류 → 모듈별 처리 → 응답 생성
```

#### 프론트엔드 (`templates/myapp/index.html`)
```html
<!-- 747줄의 현대적인 반응형 웹 페이지 -->
- 회사 소개 슬라이더 시스템
- 동영상 배경 및 애니메이션
- Google OAuth 소셜 로그인
- 모바일 반응형 네비게이션
```

#### JavaScript (`static/js/script.js`)
```javascript
// 728줄의 인터랙티브 기능
- 슬라이더 자동 재생 및 제어
- 부드러운 스크롤 애니메이션
- 햄버거 메뉴 토글
- 동적 UI 상태 관리
```

## ⚙️ 설치 및 실행

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
- **메인 페이지**: http://127.0.0.1:8000/
- **챗봇 서비스**: http://127.0.0.1:8000/chatbot/
- **관리자 페이지**: http://127.0.0.1:8000/admin/

## 🎯 주요 기능 사용법

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

## 🔍 API 엔드포인트

### 챗봇 API
```
POST /chatbot/
Content-Type: application/json

Request Body:
{
    "question": "사용자 질문"
}

Response (축구 관련):
{
    "type": "FM",
    "replies": [
        {
            "name": "선수명",
            "description": "선수 설명",
            "image_url": "이미지 URL"
        }
    ]
}

Response (HR 관련):
{
    "type": "HR",
    "answer": "답변 내용"
}
```

## 📊 시스템 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   프론트엔드     │    │   Django 백엔드   │    │   AI 모듈 계층   │
│                │    │                 │    │                │
│ • HTML/CSS/JS  │◄──►│ • Views/URLs    │◄──►│ • Question      │
│ • 반응형 UI     │    │ • Authentication │    │   Routing       │
│ • 슬라이더      │    │ • Session Mgmt  │    │ • FM Module     │
│ • 애니메이션     │    │ • Static Files  │    │ • HR Module     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▲                        ▲
                                │                        │
                    ┌──────────────────┐    ┌─────────────────┐
                    │   데이터 계층     │    │   외부 API 계층  │
                    │                 │    │                │
                    │ • SQLite DB     │    │ • OpenAI API    │
                    │ • FAISS Vector  │    │ • Naver API     │
                    │ • Session Store │    │ • Bing Search   │
                    │ • Static Files  │    │ • Web Crawling  │
                    └──────────────────┘    └─────────────────┘
```

## 🚀 성능 및 특징

### 시스템 성능
- **데이터베이스**: 8,000+ 축구 선수 정보
- **벡터 검색**: FAISS 기반 고속 유사도 검색
- **응답 시간**: 평균 2-5초 (크롤링 포함)
- **동시 사용자**: Django 기본 설정 (확장 가능)

### 핵심 특징
- **지능형 분류**: 질문 의도 자동 파악
- **멀티모달**: 텍스트 + 이미지 통합 응답
- **실시간 크롤링**: 최신 이미지 자동 수집
- **하이브리드 RAG**: 내부 + 외부 정보 통합
- **확장 가능**: 모듈식 구조로 기능 추가 용이

## 🛡️ 보안 및 제한사항

### 보안 기능
- Django CSRF 보호
- 사용자 인증 필수 (로그인 기반)
- 환경변수 기반 API 키 관리
- SQL Injection 방지

### 사용 제한사항
- OpenAI API 사용량 제한
- 크롤링 속도 제한 (Bot 탐지 방지)
- 동시 접속자 수 제한 (서버 사양 의존)
- 데이터베이스 읽기 전용 (현재 설정)

## 📈 향후 개발 계획

### 단기 목표
- [ ] 선수 데이터 실시간 업데이트 시스템
- [ ] 챗봇 대화 히스토리 저장
- [ ] 관리자 대시보드 구축
- [ ] 성능 모니터링 시스템

### 중기 목표
- [ ] 모바일 앱 개발
- [ ] 다국어 지원 (영어, 일본어)
- [ ] 실시간 시장 가격 연동
- [ ] 선수 비교 분석 도구

### 장기 목표
- [ ] 예측 모델 개발 (선수 성장 예측)
- [ ] 블록체인 기반 계약 시스템
- [ ] AI 기반 스카우팅 추천
- [ ] 글로벌 에이전시 네트워크 구축

## 🤝 기여 방법

### 개발 환경 설정
1. 이슈 등록 및 논의
2. 브랜치 생성 (`feature/기능명`)
3. 개발 및 테스트
4. Pull Request 생성
5. 코드 리뷰 및 머지

### 코드 스타일
- Python: PEP 8 준수
- JavaScript: ESLint 권장 설정
- HTML/CSS: 들여쓰기 2칸
- 주석: 한국어 권장

## 📞 지원 및 문의

### 개발팀 연락처
- **이메일**: [개발팀 이메일]
- **GitHub**: [프로젝트 레포지토리]
- **문서**: [위키 또는 가이드 링크]

### 버그 신고
- GitHub Issues를 통한 버그 신고
- 상세한 재현 단계 및 환경 정보 포함
- 스크린샷 또는 로그 첨부 권장

---

## 📄 라이선스

이 프로젝트는 [라이선스 타입]에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

*최종 업데이트: 2025년 1월 31일*
*버전: 1.0.0* 