# 🤖 Ballzzi Seokwon - AI 챗봇 프로젝트

## 📋 프로젝트 개요
Django 기반의 지능형 챗봇 시스템으로, HR(인사) 관련 질문과 Football Manager(FM) 축구 선수 정보에 대한 질문을 자동으로 분류하여 적절한 AI 에이전트로 라우팅하는 웹 애플리케이션입니다.

## 🏗️ 시스템 아키텍처

### 주요 기능
- **질문 자동 분류**: 사용자 질문을 HR 또는 축구 관련으로 자동 분류
- **HR 챗봇**: RAG(Retrieval-Augmented Generation) 기반 인사 관련 질의응답
- **FM 챗봇**: 축구 선수 정보 검색 및 이미지 크롤링
- **사용자 인증**: Django Allauth를 통한 Google 소셜 로그인
- **반응형 웹 UI**: 현대적이고 사용자 친화적인 채팅 인터페이스

### 기술 스택
- **Backend**: Django 5.2, Python
- **AI/ML**: 
  - SentenceTransformers (ko-sroberta-multitask)
  - LangChain
  - HuggingFace Embeddings
  - FAISS Vector Store
  - BGE Reranker
- **Authentication**: Django Allauth (Google OAuth)
- **Database**: SQLite (개발용)
- **Frontend**: HTML5, CSS3, JavaScript

## 📁 프로젝트 구조

```
ballzzi_seokwon/
├── manage.py                    # Django 관리 명령어
├── proj4/                       # 메인 프로젝트 설정
│   ├── settings.py             # Django 설정
│   ├── urls.py                 # 메인 URL 라우팅
│   └── wsgi.py                 # WSGI 배포 설정
├── myapp/                       # 메인 애플리케이션
│   ├── views.py                # 뷰 로직
│   ├── urls.py                 # 앱 URL 설정
│   └── source/                 # AI 모듈
│       ├── question_Routing.py # 질문 분류 모듈
│       ├── FM/                 # Football Manager 모듈
│       │   ├── FM_GetData_LLM.py
│       │   ├── data/           # 선수 데이터베이스
│       │   └── tools/          # 이미지 크롤링 등
│       └── HR/                 # 인사 관련 모듈
│           ├── agents/         # AI 에이전트
│           ├── data/           # FAISS 벡터 스토어
│           └── tools/          # RAG 도구
├── templates/                   # HTML 템플릿
│   └── myapp/
│       ├── index.html          # 메인 페이지
│       └── chatbot.html        # 챗봇 페이지
├── static/                      # 정적 파일
│   ├── styles.css              # 스타일시트
│   └── script.js               # JavaScript
├── requirements.txt             # Python 의존성
└── README.md                   # 이 파일
```

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
프로젝트 루트에 `.env` 파일 생성:
```env
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

### 3. 데이터베이스 마이그레이션
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. 서버 실행
```bash
python manage.py runserver
```

### 5. 접속
- 메인 페이지: http://127.0.0.1:8000/
- 챗봇 페이지: http://127.0.0.1:8000/chatbot/
- 관리자 페이지: http://127.0.0.1:8000/admin/

## 🌐 URL 구조

| URL | 설명 | 기능 |
|-----|------|------|
| `/` | 메인 페이지 | 홈페이지 (index.html) |
| `/chatbot/` | 챗봇 페이지 | AI 챗봇 인터페이스 |
| `/accounts/login/` | 로그인 | Google 소셜 로그인 |
| `/accounts/signup/` | 회원가입 | 새 계정 생성 |
| `/admin/` | 관리자 | Django 관리자 페이지 |

## 🤖 AI 시스템 상세

### 질문 분류 시스템
- **모델**: SentenceTransformers (ko-sroberta-multitask)
- **기능**: 한국어 질문을 HR/축구 카테고리로 자동 분류
- **알고리즘**: FAISS 기반 유사도 검색

### HR 챗봇
- **RAG 시스템**: LangChain + FAISS Vector Store
- **임베딩 모델**: nlpai-lab/KURE-v1
- **Reranker**: BAAI/bge-reranker-v2-m3
- **데이터**: 조직 HR 정책 및 절차 문서

### FM 챗봇
- **데이터베이스**: SQLite 선수 정보 DB
- **이미지 검색**: Bing 이미지 크롤링
- **기능**: 선수 정보 검색 및 시각화

## 🔧 환경 설정

### Django 설정 주요 사항
- **DEBUG**: True (개발 환경)
- **ALLOWED_HOSTS**: 로컬호스트 + 배포 IP
- **인증**: Django Allauth + Google OAuth
- **정적 파일**: `/static/` 경로
- **템플릿**: Django 템플릿 엔진

### AI 모델 설정
- **임베딩 모델**: 자동 다운로드 및 캐싱
- **FAISS 인덱스**: 사전 구축된 벡터 스토어 사용
- **모델 로드**: 애플리케이션 시작 시 자동 초기화

## 🐛 문제 해결

### 자주 발생하는 문제
1. **FAISS 로드 오류**: 벡터 스토어 경로 확인
2. **Google OAuth 오류**: .env 파일의 클라이언트 ID/Secret 확인
3. **임베딩 모델 다운로드**: 충분한 디스크 공간 확보

### 성능 최적화
- 모델 로딩은 처음 한 번만 수행
- FAISS 인덱스 캐싱 활용
- 정적 파일 압축 및 캐싱

## 📈 향후 개발 계획

### 우선순위 높음
- [ ] 로그인 기반 접근 제어 (@login_required)
- [ ] 채팅 세션 관리 시스템
- [ ] 응답 포맷 개선 및 UI/UX 향상

### 우선순위 중간
- [ ] 로그인/회원가입 페이지 CSS 스타일링
- [ ] 이미지 크롤링 안정성 개선
- [ ] 다국어 지원 (영어)

### 우선순위 낮음
- [ ] PostgreSQL 데이터베이스 마이그레이션
- [ ] Docker 컨테이너화
- [ ] 클라우드 배포 (AWS/GCP)

## 🤝 기여 방법
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 👥 개발팀
- **Developer**: Seokwon
- **Project**: Ballzzi AI Chatbot System

---
*마지막 업데이트: 2024년*
