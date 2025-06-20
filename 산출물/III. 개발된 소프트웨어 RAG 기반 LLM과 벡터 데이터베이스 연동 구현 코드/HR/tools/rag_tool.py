import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
import torch
import numpy as np
import requests

# 환경 변수 로드
load_dotenv()

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Reranker 초기화
        기본값: BAAI/bge-reranker-v2-m3 (multilingual, 한국어 지원)
        """
        # 허깅페이스 토큰 가져오기
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        print(f"Reranker 모델 로드 중: {model_name}")
        
        if hf_token:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Reranker 모델을 CUDA로 이동했습니다.")

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents
            
        pairs = [(query, doc.page_content) for doc in documents]
        features = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        if torch.cuda.is_available():
            features = {k: v.cuda() for k, v in features.items()}
        
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze()
        
        # 단일 문서인 경우 처리
        if len(documents) == 1:
            return documents
            
        # 점수에 따라 문서 재정렬
        sorted_indices = torch.argsort(scores, descending=True)
        return [documents[i] for i in sorted_indices]

class RAGTool:
    def __init__(self):
        # 허깅페이스 토큰 가져오기
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # 환경 변수에서 모델명 가져오기 (기본값 설정)
        embedding_model = os.getenv("EMBEDDING_MODEL", "nlpai-lab/KURE-v1")
        reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        
        print(f"임베딩 모델 로드 중: {embedding_model}")
        
        # 임베딩 모델 초기화 - 허깅페이스에서 직접 다운로드
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        
        # 토큰이 있으면 model_kwargs에 포함
        if hf_token:
            model_kwargs["token"] = hf_token
            # 환경변수로도 설정 (일부 transformers 라이브러리에서 사용)
            os.environ["HF_TOKEN"] = hf_token
        
        # HuggingFaceEmbeddings 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        print("임베딩 모델 로드 완료")
        
        # 여러 FAISS 인덱스 로드 - 환경변수에서 경로 가져오기
        self.vectorstores = {}
        
        # 환경변수에서 DB 경로들 가져오기
        main_db_path = os.getenv("VECTOR_DB_PATH", "HR/data/faiss_win")
        hr_db_path = os.getenv("VECTOR_DB_PATH_HR", "HR/data/faiss_org_hr")
        
        db_configs = [
            ("faiss_win", main_db_path),
            ("faiss_org_hr", hr_db_path)
        ]
        
        # 현재 작업 디렉토리 확인
        current_dir = os.getcwd()
        print(f"현재 작업 디렉토리: {current_dir}")
        
        for db_name, db_path in db_configs:
            try:
                # 현재 작업 디렉토리 출력
                print(f"[DEBUG] 현재 작업 디렉토리: {os.getcwd()}")

                # 경로 조정 (app 폴더 실행 시)
                if os.getcwd().endswith("app") or "app" in os.path.basename(os.getcwd()):
                    db_path = f"../{db_path}"
                    print(f"[DEBUG] app 폴더에서 실행 감지, 경로 조정됨: {db_path}")

                absolute_path = os.path.abspath(db_path)
                print(f"[DEBUG] {db_name} 절대 경로: {absolute_path}")
                print(f"[DEBUG] index.faiss 존재? {os.path.exists(os.path.join(absolute_path, 'index.faiss'))}")
                print(f"[DEBUG] index.pkl 존재? {os.path.exists(os.path.join(absolute_path, 'index.pkl'))}")

                vectorstore = FAISS.load_local(
                    absolute_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.vectorstores[db_name] = vectorstore
                print(f"[DEBUG] {db_name} 벡터스토어 로드 성공 ✅")
            
            except Exception as e:
                print(f"[❌ ERROR] {db_name} FAISS 로드 실패: {e}")
                continue

        if not self.vectorstores:
            raise Exception("모든 벡터 데이터베이스 로드에 실패했습니다.")
        
        print(f"로드된 벡터 DB: {list(self.vectorstores.keys())}")

        # Reranker 초기화
        self.reranker = Reranker(reranker_model)
        print("RAG 시스템 초기화 완료")

    def query_rag(self, query: str) -> str:
        try:
            # 모든 벡터스토어에서 검색
            all_docs = []
            top_k = int(os.getenv("TOP_K", "3"))
            
            for db_name, vectorstore in self.vectorstores.items():
                docs = vectorstore.similarity_search(query, k=top_k)
                # DB 이름을 메타데이터에 추가
                for doc in docs:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['source_db'] = db_name
                all_docs.extend(docs)
            
            if not all_docs:
                return f"질문: {query}\n\n관련 문서를 찾을 수 없습니다."
            
            # Reranker로 재정렬 (모든 DB 결과 통합)
            reranked_docs = self.reranker.rerank(query, all_docs)
            
            # Top-1 문서 추출
            top_doc = reranked_docs[0]
            
            # 메타데이터가 있으면 포함
            metadata_info = ""
            if hasattr(top_doc, 'metadata') and top_doc.metadata:
                metadata_info = f"\n출처: {top_doc.metadata}\n"
            
            # 컨텍스트 구성
            context = f"질문: {query}\n{metadata_info}\n관련 문서:\n{top_doc.page_content}"
            
            return context
            
        except Exception as e:
            return f"문서 검색 중 오류 발생: {str(e)}"

# 싱글톤 인스턴스 생성
print("RAG 도구 초기화 중...")
rag_tool_instance = RAGTool()

# LangChain 도구로 래핑
@tool
def search_documents(query: str) -> str:
    """문서 검색 및 관련 정보를 찾습니다. 벡터 검색과 reranking을 통해 가장 관련성 높은 문서를 반환합니다."""
    return rag_tool_instance.query_rag(query) 

@tool
def calculate_retirement_pay(avg_salary: float, years: int) -> str:
    """평균임금과 근속연수로 퇴직금을 계산합니다."""
    try:
        pay = avg_salary * years
        return f"퇴직금은 약 {pay:,.0f}원입니다."
    except Exception as e:
        return f"계산 중 오류 발생: {str(e)}"

# 네이버 검색 API 도구
@tool
def search_naver_news(query: str, max_results: int = 3) -> str:
    """네이버 뉴스 검색을 통해 최신 정보를 가져옵니다."""
    try:
        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            return "네이버 API 키가 설정되지 않았습니다. NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 확인해주세요."
        
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        params = {
            "query": query,
            "display": max_results,
            "start": 1,
            "sort": "date"  # 최신순 정렬
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                return f"'{query}'에 대한 뉴스 검색 결과가 없습니다."
            
            results = []
            for i, item in enumerate(items, 1):
                title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                description = item.get("description", "").replace("<b>", "").replace("</b>", "")
                link = item.get("link", "")
                pub_date = item.get("pubDate", "")
                
                result = f"{i}. {title}\n{description}\n발행일: {pub_date}\n링크: {link}\n"
                results.append(result)
            
            return f"네이버 뉴스 검색 결과 ('{query}'):\n\n" + "\n".join(results)
        else:
            return f"네이버 뉴스 검색 실패: HTTP {response.status_code}"
            
    except Exception as e:
        return f"네이버 뉴스 검색 중 오류 발생: {str(e)}"

@tool
def search_naver_web(query: str, max_results: int = 3) -> str:
    """네이버 웹 검색을 통해 일반 정보를 가져옵니다."""
    try:
        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            return "네이버 API 키가 설정되지 않았습니다. NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 확인해주세요."
        
        url = "https://openapi.naver.com/v1/search/webkr.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        params = {
            "query": query,
            "display": max_results,
            "start": 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                return f"'{query}'에 대한 웹 검색 결과가 없습니다."
            
            results = []
            for i, item in enumerate(items, 1):
                title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                description = item.get("description", "").replace("<b>", "").replace("</b>", "")
                link = item.get("link", "")
                
                result = f"{i}. {title}\n{description}\n링크: {link}\n"
                results.append(result)
            
            return f"네이버 웹 검색 결과 ('{query}'):\n\n" + "\n".join(results)
        else:
            return f"네이버 웹 검색 실패: HTTP {response.status_code}"
            
    except Exception as e:
        return f"네이버 웹 검색 중 오류 발생: {str(e)}"

# 내부 문서 우선 키워드 목록
INTERNAL_PRIORITY_KEYWORDS = [
    # 회사 관련
    "회사", "대표", "사장", "임원", "직원", "조직", "부서", "팀",
    # 인사 관련  
    "휴가", "연차", "병가", "출근", "퇴근", "근무", "급여", "임금", "퇴직금", "승진",
    # 규정 관련
    "규정", "정책", "규칙", "절차", "프로세스", "업무", "보고",
    # 복리후생
    "복리후생", "보험", "지원금", "교육", "연수"
]

def has_internal_keywords(query: str) -> bool:
    """질문에 내부 문서 관련 키워드가 있는지 확인"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in INTERNAL_PRIORITY_KEYWORDS)

@tool
def hybrid_search(query: str) -> str:
    """내부 문서를 우선 검색하고, 필요시 네이버 검색으로 보완하는 통합 검색 도구입니다."""
    try:
        # 1단계: 내부 문서 검색 (항상 실행)
        internal_result = rag_tool_instance.query_rag(query)
        
        # 2단계: 결과 품질 + 키워드 분석
        has_internal_content = "관련 문서를 찾을 수 없습니다" not in internal_result
        has_internal_keywords_flag = has_internal_keywords(query)
        
        # 3단계: 검색 전략 결정
        if has_internal_content and has_internal_keywords_flag:
            # 내부 문서에 결과가 있고 내부 관련 키워드면 → 내부 결과만 반환
            return f"🏢 **회사 내부 문서 검색 결과:**\n\n{internal_result}"
            
        elif has_internal_content and not has_internal_keywords_flag:
            # 내부 문서에 결과가 있지만 일반적 질문이면 → 내부 + 외부 검색
            try:
                external_result = search_naver_web(query, max_results=2)
                return f"🏢 **회사 내부 정보:**\n{internal_result}\n\n" + \
                       f"🌐 **일반 정보 (네이버 검색):**\n{external_result}"
            except:
                return f"🏢 **회사 내부 문서 검색 결과:**\n\n{internal_result}"
                
        else:
            # 내부 문서에 결과가 없으면 → 외부 검색으로 보완
            try:
                external_result = search_naver_web(query, max_results=3)
                return f"🏢 **회사 내부 검색:** 관련 문서를 찾을 수 없습니다.\n\n" + \
                       f"🌐 **네이버 웹 검색 결과:**\n{external_result}"
            except:
                return f"검색 결과를 찾을 수 없습니다: {query}"
                
    except Exception as e:
        return f"통합 검색 중 오류 발생: {str(e)}"

@tool  
def search_company_info(query: str) -> str:
    """회사 내부 정보만을 전용으로 검색합니다. 인사규정, 조직정보, 회사 정책 등을 찾을 때 사용하세요."""
    try:
        result = rag_tool_instance.query_rag(query)
        if "관련 문서를 찾을 수 없습니다" in result:
            return f"🏢 **회사 내부 문서 검색 결과:**\n\n'{query}'에 대한 회사 내부 정보를 찾을 수 없습니다.\n\n" + \
                   "다음과 같은 정보를 검색할 수 있습니다:\n" + \
                   " - 인사 규정 및 정책\n - 조직 구조 및 업무 분장\n - 복리후생 제도\n - 근무 규칙"
        else:
            return f"🏢 **회사 내부 문서 검색 결과:**\n\n{result}"
    except Exception as e:
        return f"회사 내부 검색 중 오류 발생: {str(e)}" 