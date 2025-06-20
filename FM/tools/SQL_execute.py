from langchain.chains.base import Chain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import text
from typing import ClassVar, List, Dict, Any
from pydantic import Field

class SQLExecuteChain(Chain):
    db_engine: SQLDatabase = Field(..., exclude=True)

    input_keys: ClassVar[List[str]] = ["sql_query"]
    output_keys: ClassVar[List[str]] = ["query_result"]

    def execute_sql_query(self, query):
        """
        생성된 SQL 구문 실행
        """
        with self.db_engine._engine.connect() as connection:
            try:
                result = connection.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()
                return columns, rows

            except Exception as e:
                return None
                
    def format_rows_to_sentences(self, columns, rows):
        """
        추출한 데이터를 LLM이 인식하기 쉽도록 자연어 문장으로 변환
        """
        columns = list(columns)
        sentences = []
        for row in rows:
            player_info = dict(zip(columns, row))
            parts = [f"{col}은 {player_info[col]}이고" for col in columns[:-1]]
            parts.append(f"{columns[-1]}은 {player_info[columns[-1]]}입니다.")
            sentence = ", ".join(parts)
            
            sentences.append(sentence)
        return "\n".join(sentences)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sql_query = inputs["sql_query"]
        result = self.execute_sql_query(sql_query)
        
        if result:
            result = self.format_rows_to_sentences(*result)
        
        return {"query_result": str(result)}
