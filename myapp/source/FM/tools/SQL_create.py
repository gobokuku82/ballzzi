from langchain.chains.base import Chain
from typing import Dict, Any, List

class SQLPostprocessChain(Chain):
    def __init__(self, llm_chain):
        super().__init__()
        self._llm_chain = llm_chain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["sql_query"]

    @property
    def common_cols(self) -> List[str]:
        return ["이름", "나이", "포지션", "현재_능력", "잠재력", "국적", "소속팀", "키", "몸무게", "왼발_능력", "오른발_능력", "시장_가치"]

    @property
    def pos_columns(self) -> Dict[str, List[str]]:
        return {
            "ST": self.common_cols + ["슛", "드리블", "퍼스트_터치", "헤딩", "예측력", "오프_더_볼", "침착성", "가속력", "체력", "민첩성"],
            "AM": self.common_cols + ["드리블", "중거리슛", "패스", "퍼스트_터치", "예측력", "오프_더_볼", "창의성", "판단력"],
            "M": self.common_cols + ["태클", "패스", "퍼스트_터치", "팀워크", "판단력", "지구력", "시야", "침착성", "활동량"],
            "D": self.common_cols + ["마킹", "태클", "헤딩", "수비_위치_선정", "체력", "점프력", "스피드", "경합_능력"],
            "GK": self.common_cols + ["1대1_대처", "튀어나오기", "반사_신경", "지역_장악력", "수비_위치_선정", "예측력", "집중력", "침착성", "민첩성"],
        }

    def _call(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, str]:
        question = inputs["question"]
        schema_description = "\n\n".join([
            f"{pos} 테이블 컬럼: {', '.join(stats)}"
            for pos, stats in self.pos_columns.items()
        ])
        new_inputs = {"question": question, "schema": schema_description}
    
        result = self._llm_chain.invoke(new_inputs)
        
        result = result.content
         
        cleaned = result.replace("```sql", "").replace("```", "").strip()
        cleaned = cleaned.split(';')[0]
    
        return {"sql_query": str(cleaned)}
