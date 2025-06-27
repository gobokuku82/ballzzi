"""
HR 모듈 RAG 시스템 테스트
- 벡터 검색 성능 테스트
- 리랭커 정확도 테스트
- 하이브리드 검색 테스트
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

from myapp.source.HR.agents.agent_executor import process_query

class HRRAGTester:
    def __init__(self):
        self.test_cases = [
            # HR 정책 관련 질문들
            {"question": "연차는 어떻게 써요?", "category": "vacation_policy"},
            {"question": "복지 포인트는 어디서 확인해?", "category": "welfare"}, 
            {"question": "출근 시간은 몇 시인가요?", "category": "work_hours"},
            {"question": "재택근무 신청은 어떻게?", "category": "remote_work"},
            {"question": "월급날 언제야?", "category": "salary"},
            {"question": "퇴사 절차 알려줘", "category": "resignation"},
            {"question": "야근수당 나와요?", "category": "overtime"},
            {"question": "회사 노트북 반납 절차", "category": "equipment"},
            {"question": "신입사원 교육 일정", "category": "training"},
            {"question": "부서 이동 신청 방법", "category": "transfer"},
            
            # 회사 정보 관련
            {"question": "우리 회사 대표가 누구야?", "category": "company_info"},
            {"question": "회사 주소 알려줘", "category": "company_info"},
            {"question": "회사 연혁이 궁금해", "category": "company_info"},
            {"question": "정주영에 대해 알려줘", "category": "historical_figure"},
            
            # 복합 질문
            {"question": "연차 쓰고 싶은데 승인 절차와 주의사항 알려줘", "category": "complex"},
            {"question": "퇴사 시 인수인계와 장비 반납은 어떻게?", "category": "complex"},
            
            # 일반 지식 질문 (외부 검색 필요)
            {"question": "최근 AI 기술 동향은?", "category": "external_knowledge"},
            {"question": "2024년 경제 전망", "category": "external_knowledge"},
        ]
        
        self.results = []
    
    def analyze_response_quality(self, question: str, response: str) -> Dict:
        """응답 품질 분석"""
        quality_metrics = {
            "has_content": len(response.strip()) > 0,
            "is_structured": any(marker in response for marker in ["**", "##", "1.", "-", "•"]),
            "has_korean": any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in response),
            "word_count": len(response.split()),
            "contains_error": any(error_word in response.lower() for error_word in ["error", "오류", "실패", "exception"]),
            "relevance_score": 0  # 간단한 관련성 점수
        }
        
        # 간단한 관련성 점수 계산
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        if question_words:
            quality_metrics["relevance_score"] = len(question_words & response_words) / len(question_words)
        
        # 전체 품질 점수 (0-100)
        quality_score = 0
        if quality_metrics["has_content"]: quality_score += 20
        if quality_metrics["is_structured"]: quality_score += 20
        if quality_metrics["has_korean"]: quality_score += 20
        if quality_metrics["word_count"] >= 10: quality_score += 20
        if not quality_metrics["contains_error"]: quality_score += 10
        if quality_metrics["relevance_score"] > 0.1: quality_score += 10
        
        quality_metrics["overall_score"] = quality_score
        
        return quality_metrics
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """단일 테스트 케이스 실행"""
        start_time = time.time()
        
        print(f"질문: {test_case['question']}")
        
        try:
            # HR Agent 호출
            response = process_query(test_case["question"])
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            # 응답 품질 분석
            quality = self.analyze_response_quality(test_case["question"], response)
            
            result = {
                "question": test_case["question"],
                "category": test_case["category"],
                "response": response,
                "response_time_ms": response_time,
                "quality_metrics": quality,
                "status": "PASS" if quality["overall_score"] >= 60 else "FAIL"
            }
            
            # 결과 출력
            if quality["overall_score"] >= 80:
                print(f"✅ 우수 - 품질점수: {quality['overall_score']}/100 ({response_time}ms)")
            elif quality["overall_score"] >= 60:
                print(f"👍 양호 - 품질점수: {quality['overall_score']}/100 ({response_time}ms)")
            else:
                print(f"❌ 부족 - 품질점수: {quality['overall_score']}/100 ({response_time}ms)")
            
            print(f"   응답 길이: {quality['word_count']}단어")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            result = {
                "question": test_case["question"],
                "category": test_case["category"],
                "response": None,
                "response_time_ms": response_time,
                "quality_metrics": {"overall_score": 0, "contains_error": True},
                "status": "ERROR",
                "error": str(e)
            }
            
            print(f"🚨 예외 발생 - {str(e)} ({response_time}ms)")
            return result
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 케이스 실행"""
        print("🚀 HR RAG 시스템 테스트 시작...")
        print("=" * 80)
        
        total_tests = len(self.test_cases)
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        total_time = 0
        total_quality_score = 0
        
        # 카테고리별 통계
        category_stats = {}
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i:2d}/{total_tests}] 테스트 실행 중...")
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            # 통계 업데이트
            category = test_case["category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0, "total_score": 0}
            
            category_stats[category]["total"] += 1
            category_stats[category]["total_score"] += result["quality_metrics"]["overall_score"]
            
            if result["status"] == "PASS":
                passed_tests += 1
                category_stats[category]["passed"] += 1
            elif result["status"] == "FAIL":
                failed_tests += 1
            else:
                error_tests += 1
            
            total_time += result["response_time_ms"]
            total_quality_score += result["quality_metrics"]["overall_score"]
        
        # 결과 요약
        success_rate = (passed_tests / total_tests) * 100
        avg_response_time = total_time / total_tests
        avg_quality_score = total_quality_score / total_tests
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": round(success_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "avg_quality_score": round(avg_quality_score, 2),
            "category_stats": category_stats
        }
        
        self.print_summary(summary)
        return summary
    
    def print_summary(self, summary: Dict):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 HR RAG 시스템 테스트 결과 요약")
        print("=" * 80)
        
        print(f"전체 테스트: {summary['total_tests']}개")
        print(f"성공: {summary['passed']}개")
        print(f"실패: {summary['failed']}개")
        print(f"오류: {summary['errors']}개")
        print(f"성공률: {summary['success_rate']}%")
        print(f"평균 응답시간: {summary['avg_response_time_ms']}ms")
        print(f"평균 품질점수: {summary['avg_quality_score']}/100")
        
        # 성능 평가
        if summary["avg_response_time_ms"] < 3000:
            print("⚡ 응답속도: 우수 (3초 미만)")
        elif summary["avg_response_time_ms"] < 5000:
            print("👍 응답속도: 양호 (5초 미만)")
        else:
            print("⚠️ 응답속도: 개선 필요 (5초 초과)")
        
        if summary["avg_quality_score"] >= 80:
            print("🎯 응답품질: 우수")
        elif summary["avg_quality_score"] >= 60:
            print("👍 응답품질: 양호")
        else:
            print("⚠️ 응답품질: 개선 필요")
        
        print("\n📈 카테고리별 성능:")
        for category, stats in summary["category_stats"].items():
            if stats["total"] > 0:
                rate = (stats["passed"] / stats["total"]) * 100
                avg_score = stats["total_score"] / stats["total"]
                print(f"  {category}: {stats['passed']}/{stats['total']} ({rate:.1f}%, 품질: {avg_score:.1f})")
        
        # 전체 평가
        if summary["success_rate"] >= 90 and summary["avg_quality_score"] >= 75:
            print(f"\n🎉 우수함: RAG 시스템이 고품질 응답 생성")
        elif summary["success_rate"] >= 80 and summary["avg_quality_score"] >= 60:
            print(f"\n👍 양호함: 대부분 적절한 응답, 일부 개선 필요")
        elif summary["success_rate"] >= 70:
            print(f"\n⚠️ 주의: 벡터 검색 또는 리랭커 튜닝 필요")
        else:
            print(f"\n🚨 심각: RAG 시스템 전반적인 점검 필요")
    
    def save_results(self, filename: str = "hr_rag_test_results.json"):
        """테스트 결과를 JSON 파일로 저장"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "HR RAG System Test",
            "results": self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 테스트 결과가 '{filename}'에 저장되었습니다.")

def main():
    """메인 테스트 실행 함수"""
    tester = HRRAGTester()
    
    try:
        summary = tester.run_all_tests()
        tester.save_results()
        
        # 성공률에 따른 종료 코드 설정
        if summary["success_rate"] >= 80:
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