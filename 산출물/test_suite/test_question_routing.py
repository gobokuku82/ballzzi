"""
질문 라우팅 시스템 테스트
- Sentence-Transformer 기반 분류 정확도 테스트
- FAISS 인덱스 검색 성능 테스트
- FM/HR 모듈 분기 로직 테스트
"""

import sys
import os
import time
import json
from typing import List, Dict

# Django 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj4.settings')

import django
django.setup()

from myapp.source.question_Routing import classify

class QuestionRoutingTester:
    def __init__(self):
        self.test_cases = [
            # FM 모듈 테스트 케이스 (축구 관련)
            {"question": "손흥민에 대해 알려줘", "expected": False, "category": "FM"},
            {"question": "리오넬 메시의 특징은?", "expected": False, "category": "FM"},
            {"question": "호날두와 메시 중 누가 더 좋아?", "expected": False, "category": "FM"},
            {"question": "최고의 수비수는 누구야?", "expected": False, "category": "FM"},
            {"question": "EPL에서 떠오르는 유망주는?", "expected": False, "category": "FM"},
            {"question": "드리블 잘하는 선수 추천해줘", "expected": False, "category": "FM"},
            {"question": "김민재 수비 실력 어때?", "expected": False, "category": "FM"},
            {"question": "이강인 패스 능력은?", "expected": False, "category": "FM"},
            {"question": "황희찬 골 결정력은?", "expected": False, "category": "FM"},
            {"question": "스피드 빠른 공격수 알려줘", "expected": False, "category": "FM"},
            
            # HR 모듈 테스트 케이스 (회사 정책 관련)
            {"question": "연차는 어떻게 써요?", "expected": True, "category": "HR"},
            {"question": "복지 포인트는 어디서 확인해?", "expected": True, "category": "HR"},
            {"question": "퇴사하려면 뭐부터 해야 해?", "expected": True, "category": "HR"},
            {"question": "출근 시간은 몇 시인가요?", "expected": True, "category": "HR"},
            {"question": "재택근무 신청은 어디서?", "expected": True, "category": "HR"},
            {"question": "월급날 언제야?", "expected": True, "category": "HR"},
            {"question": "야근수당 나와요?", "expected": True, "category": "HR"},
            {"question": "점심시간 몇 시부터지?", "expected": True, "category": "HR"},
            {"question": "회사 노트북 언제 반납해요?", "expected": True, "category": "HR"},
            {"question": "상여금 지급일 알려줘", "expected": True, "category": "HR"},
            
            # 경계 케이스 (애매한 질문들)
            {"question": "회사 축구팀 있어요?", "expected": True, "category": "BOUNDARY"},
            {"question": "직원 체육대회 언제 해요?", "expected": True, "category": "BOUNDARY"},
            {"question": "우리 회사 대표가 누구야?", "expected": True, "category": "BOUNDARY"},
            {"question": "정주영에 대해 알려줘", "expected": True, "category": "BOUNDARY"},
            {"question": "축구 관련 업무 있나요?", "expected": True, "category": "BOUNDARY"},
        ]
        
        self.results = []
        
    def run_single_test(self, test_case: Dict) -> Dict:
        """단일 테스트 케이스 실행"""
        start_time = time.time()
        
        try:
            # 라우팅 함수 호출 (True = HR, False = FM)
            result = classify(test_case["question"])
            end_time = time.time()
            
            # 결과 검증
            is_correct = result == test_case["expected"]
            response_time = round((end_time - start_time) * 1000, 2)  # ms
            
            return {
                "question": test_case["question"],
                "expected_category": test_case["category"],
                "expected_result": "HR" if test_case["expected"] else "FM",
                "actual_result": "HR" if result else "FM",
                "is_correct": is_correct,
                "response_time_ms": response_time,
                "status": "PASS" if is_correct else "FAIL"
            }
            
        except Exception as e:
            return {
                "question": test_case["question"],
                "expected_category": test_case["category"],
                "expected_result": "HR" if test_case["expected"] else "FM",
                "actual_result": "ERROR",
                "is_correct": False,
                "response_time_ms": 0,
                "status": "ERROR",
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 케이스 실행"""
        print("🚀 질문 라우팅 시스템 테스트 시작...")
        print("=" * 80)
        
        total_tests = len(self.test_cases)
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        total_time = 0
        
        # 카테고리별 통계
        category_stats = {"FM": {"total": 0, "passed": 0}, 
                         "HR": {"total": 0, "passed": 0}, 
                         "BOUNDARY": {"total": 0, "passed": 0}}
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i:2d}/{total_tests}] 테스트 실행 중...")
            print(f"질문: {test_case['question']}")
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            # 통계 업데이트
            category = test_case["category"]
            category_stats[category]["total"] += 1
            
            if result["status"] == "PASS":
                passed_tests += 1
                category_stats[category]["passed"] += 1
                print(f"✅ {result['status']} - {result['actual_result']} ({result['response_time_ms']}ms)")
            elif result["status"] == "FAIL":
                failed_tests += 1
                print(f"❌ {result['status']} - 예상: {result['expected_result']}, 실제: {result['actual_result']} ({result['response_time_ms']}ms)")
            else:
                error_tests += 1
                print(f"🚨 {result['status']} - {result.get('error', 'Unknown error')}")
            
            total_time += result["response_time_ms"]
        
        # 결과 요약
        success_rate = (passed_tests / total_tests) * 100
        avg_response_time = total_time / total_tests
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": round(success_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "category_stats": category_stats
        }
        
        self.print_summary(summary)
        return summary
    
    def print_summary(self, summary: Dict):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 질문 라우팅 테스트 결과 요약")
        print("=" * 80)
        
        print(f"전체 테스트: {summary['total_tests']}개")
        print(f"성공: {summary['passed']}개")
        print(f"실패: {summary['failed']}개") 
        print(f"오류: {summary['errors']}개")
        print(f"성공률: {summary['success_rate']}%")
        print(f"평균 응답시간: {summary['avg_response_time_ms']}ms")
        
        print("\n📈 카테고리별 성능:")
        for category, stats in summary["category_stats"].items():
            if stats["total"] > 0:
                rate = (stats["passed"] / stats["total"]) * 100
                print(f"  {category}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        # 전체 평가
        if summary["success_rate"] >= 95:
            print(f"\n🎉 우수함: 성공률 {summary['success_rate']}%로 배포 준비 완료!")
        elif summary["success_rate"] >= 90:
            print(f"\n👍 양호함: 성공률 {summary['success_rate']}%로 일부 개선 후 배포 가능")
        elif summary["success_rate"] >= 80:
            print(f"\n⚠️  주의: 성공률 {summary['success_rate']}%로 추가 튜닝 필요")
        else:
            print(f"\n🚨 심각: 성공률 {summary['success_rate']}%로 시스템 점검 필요")
    
    def save_results(self, filename: str = "question_routing_test_results.json"):
        """테스트 결과를 JSON 파일로 저장"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "Question Routing System Test",
            "results": self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 테스트 결과가 '{filename}'에 저장되었습니다.")

def main():
    """메인 테스트 실행 함수"""
    tester = QuestionRoutingTester()
    
    try:
        summary = tester.run_all_tests()
        tester.save_results()
        
        # 성공률에 따른 종료 코드 설정
        if summary["success_rate"] >= 90:
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