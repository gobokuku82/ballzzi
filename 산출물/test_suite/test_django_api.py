"""
Django API 엔드포인트 테스트
- 챗봇 API 통합 테스트
- CSRF 보호 테스트
- JSON 응답 형식 테스트
- 에러 처리 테스트
"""

import sys
import os
import time
import json
import requests
from typing import Dict, List
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

# Django 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj4.settings')

import django
django.setup()

class DjangoAPITester:
    def __init__(self):
        self.client = Client()
        self.base_url = "http://localhost:8000"
        self.test_user = None
        self.setup_test_user()
        
        self.test_cases = [
            # FM 모듈 테스트 케이스
            {"question": "손흥민에 대해 알려줘", "expected_type": "FM", "category": "football"},
            {"question": "메시 정보 알려줘", "expected_type": "FM", "category": "football"},
            {"question": "최고의 수비수는?", "expected_type": "FM", "category": "football"},
            
            # HR 모듈 테스트 케이스
            {"question": "연차는 어떻게 써요?", "expected_type": "HR", "category": "hr_policy"},
            {"question": "출근 시간은?", "expected_type": "HR", "category": "hr_policy"},
            {"question": "월급날 언제야?", "expected_type": "HR", "category": "hr_policy"},
            
            # 에러 케이스
            {"question": "", "expected_type": "ERROR", "category": "error"},
            {"question": None, "expected_type": "ERROR", "category": "error"},
        ]
        
        self.results = []
    
    def setup_test_user(self):
        """테스트용 사용자 생성"""
        try:
            self.test_user = User.objects.get(username='testuser')
        except User.DoesNotExist:
            self.test_user = User.objects.create_user(
                username='testuser',
                email='test@example.com',
                password='testpass123'
            )
        
        # 로그인
        self.client.login(username='testuser', password='testpass123')
        print("✅ 테스트 사용자 로그인 완료")
    
    def test_chatbot_get_request(self) -> Dict:
        """챗봇 페이지 GET 요청 테스트"""
        start_time = time.time()
        
        try:
            response = self.client.get('/chatbot/')
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            result = {
                "test_type": "GET /chatbot/",
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "has_content": len(response.content) > 0,
                "content_type": response.get('Content-Type', ''),
                "status": "PASS" if response.status_code == 200 else "FAIL"
            }
            
            print(f"GET /chatbot/ - {response.status_code} ({response_time}ms)")
            return result
            
        except Exception as e:
            return {
                "test_type": "GET /chatbot/",
                "status_code": 500,
                "response_time_ms": 0,
                "status": "ERROR",
                "error": str(e)
            }
    
    def test_chatbot_post_request(self, test_case: Dict) -> Dict:
        """챗봇 POST 요청 테스트"""
        start_time = time.time()
        
        try:
            # CSRF 토큰 획득
            csrf_response = self.client.get('/chatbot/')
            csrf_token = csrf_response.cookies.get('csrftoken')
            
            # POST 요청 데이터 준비
            if test_case["question"] is None:
                post_data = {}
            else:
                post_data = {"question": test_case["question"]}
            
            # POST 요청 실행
            response = self.client.post(
                '/chatbot/',
                data=json.dumps(post_data),
                content_type='application/json',
                HTTP_X_CSRFTOKEN=csrf_token.value if csrf_token else ''
            )
            
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            # 응답 분석
            try:
                response_data = json.loads(response.content)
                is_valid_json = True
            except json.JSONDecodeError:
                response_data = {}
                is_valid_json = False
            
            result = {
                "test_type": "POST /chatbot/",
                "question": test_case["question"],
                "expected_type": test_case["expected_type"],
                "category": test_case["category"],
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "is_valid_json": is_valid_json,
                "response_data": response_data,
                "has_type_field": "type" in response_data,
                "actual_type": response_data.get("type", "UNKNOWN"),
                "has_answer": "answer" in response_data or "replies" in response_data,
                "status": "PASS"
            }
            
            # 성공/실패 판정
            if test_case["expected_type"] == "ERROR":
                if response.status_code >= 400:
                    result["status"] = "PASS"
                else:
                    result["status"] = "FAIL"
            else:
                if (response.status_code == 200 and 
                    is_valid_json and 
                    result["actual_type"] == test_case["expected_type"]):
                    result["status"] = "PASS"
                else:
                    result["status"] = "FAIL"
            
            # 결과 출력
            if result["status"] == "PASS":
                print(f"✅ 성공 - {result['actual_type']} 응답 ({response_time}ms)")
            else:
                print(f"❌ 실패 - 예상: {test_case['expected_type']}, 실제: {result['actual_type']} ({response_time}ms)")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            result = {
                "test_type": "POST /chatbot/",
                "question": test_case["question"],
                "expected_type": test_case["expected_type"],
                "category": test_case["category"],
                "status_code": 500,
                "response_time_ms": response_time,
                "status": "ERROR",
                "error": str(e)
            }
            
            print(f"🚨 예외 발생 - {str(e)} ({response_time}ms)")
            return result
    
    def test_csrf_protection(self) -> Dict:
        """CSRF 보호 테스트"""
        start_time = time.time()
        
        try:
            # CSRF 토큰 없이 POST 요청
            response = self.client.post(
                '/chatbot/',
                data=json.dumps({"question": "테스트"}),
                content_type='application/json'
            )
            
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            result = {
                "test_type": "CSRF Protection Test",
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "csrf_protected": response.status_code == 403,
                "status": "PASS" if response.status_code == 403 else "FAIL"
            }
            
            if result["csrf_protected"]:
                print(f"✅ CSRF 보호 작동 - 403 Forbidden ({response_time}ms)")
            else:
                print(f"❌ CSRF 보호 미작동 - {response.status_code} ({response_time}ms)")
            
            return result
            
        except Exception as e:
            return {
                "test_type": "CSRF Protection Test",
                "status_code": 500,
                "response_time_ms": 0,
                "status": "ERROR",
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 케이스 실행"""
        print("🚀 Django API 테스트 시작...")
        print("=" * 80)
        
        all_results = []
        
        # 1. GET 요청 테스트
        print("\n📄 GET 요청 테스트:")
        get_result = self.test_chatbot_get_request()
        all_results.append(get_result)
        
        # 2. CSRF 보호 테스트
        print("\n🔒 CSRF 보호 테스트:")
        csrf_result = self.test_csrf_protection()
        all_results.append(csrf_result)
        
        # 3. POST 요청 테스트
        print(f"\n💬 POST 요청 테스트 ({len(self.test_cases)}개 케이스):")
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i:2d}/{len(self.test_cases)}] 테스트 실행 중...")
            print(f"질문: {test_case['question']}")
            
            result = self.test_chatbot_post_request(test_case)
            all_results.append(result)
        
        # 통계 계산
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r["status"] == "PASS")
        failed_tests = sum(1 for r in all_results if r["status"] == "FAIL")
        error_tests = sum(1 for r in all_results if r["status"] == "ERROR")
        
        total_time = sum(r["response_time_ms"] for r in all_results)
        avg_response_time = total_time / total_tests if total_tests > 0 else 0
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": round(success_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "all_results": all_results
        }
        
        self.print_summary(summary)
        return summary
    
    def print_summary(self, summary: Dict):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 Django API 테스트 결과 요약")
        print("=" * 80)
        
        print(f"전체 테스트: {summary['total_tests']}개")
        print(f"성공: {summary['passed']}개")
        print(f"실패: {summary['failed']}개")
        print(f"오류: {summary['errors']}개")
        print(f"성공률: {summary['success_rate']}%")
        print(f"평균 응답시간: {summary['avg_response_time_ms']}ms")
        
        # 테스트 유형별 결과
        print("\n📈 테스트 유형별 결과:")
        test_types = {}
        for result in summary["all_results"]:
            test_type = result["test_type"]
            if test_type not in test_types:
                test_types[test_type] = {"total": 0, "passed": 0}
            test_types[test_type]["total"] += 1
            if result["status"] == "PASS":
                test_types[test_type]["passed"] += 1
        
        for test_type, stats in test_types.items():
            rate = (stats["passed"] / stats["total"]) * 100
            print(f"  {test_type}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        # 전체 평가
        if summary["success_rate"] >= 95:
            print(f"\n🎉 우수함: Django API가 안정적으로 작동")
        elif summary["success_rate"] >= 85:
            print(f"\n👍 양호함: 대부분 정상 작동, 일부 개선 필요")
        elif summary["success_rate"] >= 70:
            print(f"\n⚠️ 주의: API 엔드포인트 점검 필요")
        else:
            print(f"\n🚨 심각: Django 백엔드 전반적인 점검 필요")
    
    def save_results(self, filename: str = "django_api_test_results.json"):
        """테스트 결과를 JSON 파일로 저장"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "Django API Test",
            "results": self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 테스트 결과가 '{filename}'에 저장되었습니다.")

def main():
    """메인 테스트 실행 함수"""
    tester = DjangoAPITester()
    
    try:
        summary = tester.run_all_tests()
        tester.save_results()
        
        # 성공률에 따른 종료 코드 설정
        if summary["success_rate"] >= 85:
            exit_code = 0
        else:
            exit_code = 1
            
        print(f"\n테스트 완료. 종료 코드: {exit_code}")
        return exit_code
        
    except Exception as e:
        print(f"\n🚨 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 