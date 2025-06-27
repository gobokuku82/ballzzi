"""
FM 모듈 테스트 (축구 선수 정보)
- SQL 쿼리 생성 정확도 테스트
- SQLite DB 연동 안정성 테스트
- JSON 응답 파싱 성공률 테스트
"""

import sys
import os
import time
import json
import sqlite3
from typing import List, Dict, Any

# Django 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj4.settings')

import django
django.setup()

from myapp.source.FM.FM_GetData_LLM import get_answer_from_question

class FMModuleTester:
    def __init__(self):
        # 테스트 케이스들
        self.test_cases = [
            # 기본 선수 정보 질의
            {"question": "손흥민에 대해 알려줘", "expected_type": "player_info"},
            {"question": "리오넬 메시 정보", "expected_type": "player_info"},
            {"question": "호날두는 어떤 선수야?", "expected_type": "player_info"},
            
            # 예산 기반 질의
            {"question": "10000000 예산 안에서 영입가능한 수비수", "expected_type": "budget_search"},
            {"question": "5000만원 이하 공격수 추천", "expected_type": "budget_search"},
            {"question": "2억 예산으로 미드필더 찾아줘", "expected_type": "budget_search"},
            
            # 포지션별 질의  
            {"question": "최고의 골키퍼는 누구야?", "expected_type": "position_search"},
            {"question": "수비수 중에서 키 큰 선수", "expected_type": "position_search"},
            {"question": "스피드 빠른 윙어 알려줘", "expected_type": "position_search"},
            
            # 복합 조건 질의
            {"question": "젊고 실력 좋은 공격수 추천", "expected_type": "complex_search"},
            {"question": "드리블 잘하고 패스도 정확한 선수", "expected_type": "complex_search"},
            {"question": "EPL에서 뛰는 한국 선수들", "expected_type": "complex_search"},
            
            # 비교 질의
            {"question": "손흥민과 황희찬 비교해줘", "expected_type": "comparison"},
            {"question": "메시와 호날두 중 누가 더 좋아?", "expected_type": "comparison"},
            
            # 에러 케이스
            {"question": "존재하지않는선수12345", "expected_type": "error_case"},
            {"question": "abc123!@# 선수", "expected_type": "error_case"},
        ]
        
        self.results = []
        self.db_connection_test()
    
    def db_connection_test(self):
        """데이터베이스 연결 테스트"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "myapp", "source", "FM", "data", "players_position.db")
            
            if not os.path.exists(db_path):
                print(f"❌ DB 파일이 존재하지 않습니다: {db_path}")
                return False
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 테이블 구조 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✅ DB 연결 성공. 테이블: {[table[0] for table in tables]}")
            
            # 데이터 개수 확인 (첫 번째 테이블)
            if tables:
                table_name = tables[0][0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"✅ {table_name} 테이블에 {count}개 레코드 존재")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            return False
    
    def validate_json_response(self, response: List[Dict]) -> Dict:
        """JSON 응답 유효성 검사"""
        validation_result = {
            "is_valid": True,
            "has_error": False,
            "player_count": 0,
            "has_images": False,
            "fields_present": [],
            "issues": []
        }
        
        try:
            if not isinstance(response, list):
                validation_result["is_valid"] = False
                validation_result["issues"].append("응답이 리스트 형태가 아님")
                return validation_result
            
            if len(response) == 0:
                validation_result["issues"].append("빈 응답")
                return validation_result
            
            # 에러 응답 체크
            if any("error" in item for item in response):
                validation_result["has_error"] = True
                validation_result["issues"].append("에러 응답 포함")
            
            # 선수 정보 분석
            player_count = 0
            common_fields = set()
            
            for item in response:
                if isinstance(item, dict) and "Name" in item:
                    player_count += 1
                    common_fields.update(item.keys())
            
            validation_result["player_count"] = player_count
            validation_result["fields_present"] = list(common_fields)
            
            # 일반적인 필드들이 있는지 확인
            expected_fields = ["Name", "설명"]
            present_expected = [field for field in expected_fields if field in common_fields]
            
            if len(present_expected) < len(expected_fields):
                validation_result["issues"].append(f"필수 필드 누락: {set(expected_fields) - set(present_expected)}")
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"JSON 검증 오류: {str(e)}")
        
        return validation_result
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """단일 테스트 케이스 실행"""
        start_time = time.time()
        
        print(f"질문: {test_case['question']}")
        
        try:
            # FM 모듈 호출
            response = get_answer_from_question(test_case["question"])
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            # JSON 응답 검증
            validation = self.validate_json_response(response)
            
            result = {
                "question": test_case["question"],
                "expected_type": test_case["expected_type"],
                "response_time_ms": response_time,
                "response_data": response,
                "validation": validation,
                "status": "PASS" if validation["is_valid"] else "FAIL"
            }
            
            # 결과 출력
            if validation["is_valid"]:
                print(f"✅ 성공 - {validation['player_count']}명 선수 정보 ({response_time}ms)")
                if validation["fields_present"]:
                    print(f"   필드: {', '.join(validation['fields_present'])}")
            else:
                print(f"❌ 실패 - {', '.join(validation['issues'])} ({response_time}ms)")
            
            if validation["has_error"]:
                print(f"⚠️  에러 포함됨")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            result = {
                "question": test_case["question"],
                "expected_type": test_case["expected_type"], 
                "response_time_ms": response_time,
                "response_data": None,
                "validation": {"is_valid": False, "issues": [str(e)]},
                "status": "ERROR",
                "error": str(e)
            }
            
            print(f"🚨 예외 발생 - {str(e)} ({response_time}ms)")
            return result
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 케이스 실행"""
        print("🚀 FM 모듈 테스트 시작...")
        print("=" * 80)
        
        total_tests = len(self.test_cases)
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        total_time = 0
        total_players_found = 0
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i:2d}/{total_tests}] 테스트 실행 중...")
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            # 통계 업데이트
            if result["status"] == "PASS":
                passed_tests += 1
                if result["validation"]["player_count"]:
                    total_players_found += result["validation"]["player_count"]
            elif result["status"] == "FAIL":
                failed_tests += 1
            else:
                error_tests += 1
            
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
            "total_players_found": total_players_found,
            "avg_players_per_query": round(total_players_found / total_tests, 2)
        }
        
        self.print_summary(summary)
        return summary
    
    def print_summary(self, summary: Dict):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 FM 모듈 테스트 결과 요약")
        print("=" * 80)
        
        print(f"전체 테스트: {summary['total_tests']}개")
        print(f"성공: {summary['passed']}개")
        print(f"실패: {summary['failed']}개")
        print(f"오류: {summary['errors']}개")
        print(f"성공률: {summary['success_rate']}%")
        print(f"평균 응답시간: {summary['avg_response_time_ms']}ms")
        print(f"총 검색된 선수: {summary['total_players_found']}명")
        print(f"쿼리당 평균 선수 수: {summary['avg_players_per_query']}명")
        
        # 성능 평가
        if summary["avg_response_time_ms"] < 2000:
            print("⚡ 응답속도: 우수 (2초 미만)")
        elif summary["avg_response_time_ms"] < 5000:
            print("👍 응답속도: 양호 (5초 미만)")
        else:
            print("⚠️ 응답속도: 개선 필요 (5초 초과)")
        
        # 전체 평가
        if summary["success_rate"] >= 95:
            print(f"\n🎉 우수함: SQL 생성 및 DB 연동이 안정적으로 작동")
        elif summary["success_rate"] >= 85:
            print(f"\n👍 양호함: 대부분 정상 작동, 일부 개선 필요")
        elif summary["success_rate"] >= 70:
            print(f"\n⚠️ 주의: SQL 쿼리 생성 로직 점검 필요")
        else:
            print(f"\n🚨 심각: FM 모듈 전반적인 점검 필요")
    
    def save_results(self, filename: str = "fm_module_test_results.json"):
        """테스트 결과를 JSON 파일로 저장"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "FM Module Test (Football Manager)",
            "results": self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 테스트 결과가 '{filename}'에 저장되었습니다.")

def main():
    """메인 테스트 실행 함수"""
    tester = FMModuleTester()
    
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