#!/usr/bin/env python3
"""
통합 테스트 러너
모든 테스트 모듈을 순차적으로 실행하고 종합 결과를 생성합니다.
"""

import sys
import os
import time
import json
import importlib
from typing import Dict, List

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class IntegratedTestRunner:
    def __init__(self):
        self.test_modules = [
            {
                "name": "질문 라우팅 테스트",
                "module": "test_question_routing",
                "critical": True
            },
            {
                "name": "FM 모듈 테스트",
                "module": "test_fm_module", 
                "critical": True
            },
            {
                "name": "HR RAG 테스트",
                "module": "test_hr_rag",
                "critical": True
            },
            {
                "name": "Django API 테스트",
                "module": "test_django_api",
                "critical": True
            }
        ]
        
        self.results = {}
        self.start_time = time.time()
    
    def run_single_module(self, module_config: Dict) -> Dict:
        """단일 테스트 모듈 실행"""
        module_name = module_config["name"]
        module_file = module_config["module"]
        is_critical = module_config["critical"]
        
        print(f"\n{'='*20} {module_name} {'='*20}")
        
        start_time = time.time()
        
        try:
            # 모듈 동적 import 및 실행
            module = importlib.import_module(module_file)
            exit_code = module.main()
            
            end_time = time.time()
            execution_time = round(end_time - start_time, 2)
            
            result = {
                "module_name": module_name,
                "module_file": module_file,
                "is_critical": is_critical,
                "exit_code": exit_code,
                "execution_time_sec": execution_time,
                "status": "PASS" if exit_code == 0 else "FAIL",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if exit_code == 0:
                print(f"✅ {module_name} 완료 ({execution_time}초)")
            else:
                print(f"❌ {module_name} 실패 (코드: {exit_code}, {execution_time}초)")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = round(end_time - start_time, 2)
            
            result = {
                "module_name": module_name,
                "module_file": module_file,
                "is_critical": is_critical,
                "exit_code": 1,
                "execution_time_sec": execution_time,
                "status": "ERROR",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"🚨 {module_name} 예외 발생: {str(e)} ({execution_time}초)")
            return result
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 모듈 실행"""
        print("🚀 통합 테스트 시작...")
        print(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        total_modules = len(self.test_modules)
        passed_modules = 0
        failed_modules = 0
        critical_failures = 0
        
        for i, module_config in enumerate(self.test_modules, 1):
            print(f"\n[{i}/{total_modules}] {module_config['name']} 실행 중...")
            
            result = self.run_single_module(module_config)
            self.results[module_config["module"]] = result
            
            if result["status"] == "PASS":
                passed_modules += 1
            else:
                failed_modules += 1
                if module_config["critical"]:
                    critical_failures += 1
        
        # 전체 실행 시간 계산
        total_time = round(time.time() - self.start_time, 2)
        
        # 종합 결과 생성
        summary = {
            "test_run_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_execution_time_sec": total_time,
                "total_modules": total_modules
            },
            "results_summary": {
                "passed_modules": passed_modules,
                "failed_modules": failed_modules,
                "critical_failures": critical_failures,
                "success_rate": round((passed_modules / total_modules) * 100, 2)
            },
            "module_results": self.results
        }
        
        self.print_final_summary(summary)
        self.save_integrated_results(summary)
        
        return summary
    
    def print_final_summary(self, summary: Dict):
        """최종 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("🎯 통합 테스트 최종 결과")
        print("=" * 80)
        
        run_info = summary["test_run_info"]
        results = summary["results_summary"]
        
        print(f"실행 시간: {run_info['timestamp']}")
        print(f"총 실행 시간: {run_info['total_execution_time_sec']}초")
        print(f"테스트 모듈 수: {run_info['total_modules']}개")
        print()
        print(f"성공: {results['passed_modules']}개")
        print(f"실패: {results['failed_modules']}개")
        print(f"중요 실패: {results['critical_failures']}개")
        print(f"전체 성공률: {results['success_rate']}%")
        
        print("\n📊 모듈별 상세 결과:")
        for module_key, result in summary["module_results"].items():
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            critical_mark = " [중요]" if result["is_critical"] else ""
            print(f"  {status_emoji} {result['module_name']}{critical_mark} - {result['execution_time_sec']}초")
        
        # 최종 평가
        if results["critical_failures"] == 0 and results["success_rate"] >= 90:
            print(f"\n🎉 전체 평가: 우수 - 프로덕션 배포 준비 완료!")
            overall_status = "EXCELLENT"
        elif results["critical_failures"] == 0 and results["success_rate"] >= 75:
            print(f"\n👍 전체 평가: 양호 - 일부 개선 후 배포 가능")
            overall_status = "GOOD"
        elif results["critical_failures"] <= 1:
            print(f"\n⚠️ 전체 평가: 주의 - 중요 시스템 점검 필요")
            overall_status = "WARNING"
        else:
            print(f"\n🚨 전체 평가: 심각 - 전반적인 시스템 재검토 필요")
            overall_status = "CRITICAL"
        
        summary["overall_status"] = overall_status
        
        return summary
    
    def save_integrated_results(self, summary: Dict):
        """통합 결과를 JSON 파일로 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 통합 테스트 결과가 '{filename}'에 저장되었습니다.")
        
        # 간단한 요약 파일도 생성
        summary_filename = f"test_summary_{timestamp}.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("RAG 기반 챗봇 시스템 테스트 결과 요약\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"실행 시간: {summary['test_run_info']['timestamp']}\n")
            f.write(f"총 실행 시간: {summary['test_run_info']['total_execution_time_sec']}초\n")
            f.write(f"성공률: {summary['results_summary']['success_rate']}%\n")
            f.write(f"전체 평가: {summary['overall_status']}\n\n")
            
            f.write("모듈별 결과:\n")
            for result in summary["module_results"].values():
                status = "성공" if result["status"] == "PASS" else "실패"
                f.write(f"- {result['module_name']}: {status}\n")
        
        print(f"📄 요약 보고서가 '{summary_filename}'에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    runner = IntegratedTestRunner()
    
    try:
        summary = runner.run_all_tests()
        
        # 종료 코드 결정
        if summary["results_summary"]["critical_failures"] == 0:
            exit_code = 0
        else:
            exit_code = 1
        
        print(f"\n통합 테스트 완료. 종료 코드: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
        return 130
    except Exception as e:
        print(f"\n🚨 통합 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 