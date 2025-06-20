import streamlit as st
import sys
import torch
sys.modules['torch.classes'].__path__ = []

from FM.FM_GetData_LLM import get_answer_from_question
from FM.tools.image_craper import get_player_image_from_bing
from question_Routing import classify
from HR.agents.agent_executor import process_query

# 페이지 설정
st.set_page_config(page_title="⚽ Soccer Energies Company 챗봇", layout="wide")
st.title("⚽ Soccer Energies Company 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# 🔹 왼쪽 사이드바 메뉴바
with st.sidebar:
    st.markdown("## 📋 예시 질문")
    example_questions = [
        "리오넬 메시에 대해 알려줘",
        "호날두의 커리어는 어때?",
        "손흥민은 어떤 팀에서 뛰고 있어?",
        "네이마르의 특징은?",
        "휴가를 사용 하는 방법이 있어?",
        "출장을 갈 때 어떤 이동 수단을 타야하지?",
        "연차에 대한 규정 알려줘"

    ]
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.pending_input = q

# 🔸 오른쪽: 채팅 인터페이스
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# 🔻 입력창
prompt = st.chat_input("축구선수/내부문서 이름 또는 질문을 입력하세요...")

# 예시 버튼 클릭 시 입력 적용
if st.session_state.pending_input:
    prompt = st.session_state.pending_input
    st.session_state.pending_input = None

# 🔁 입력 처리
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🤖 답변을 생성하는 중입니다..."):
        if classify(prompt):
            reply = process_query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            reply = get_answer_from_question(prompt)
            full_response = ""
            for chat in reply:
                full_response += f"### {chat['Name']}\n{chat['설명']}\n\n"
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            with st.chat_message("assistant"):
                for chat in reply:
                    image = get_player_image_from_bing(chat['Name'])
                    if image:
                        st.image(image, caption=f"{chat['Name']} 사진", use_container_width=True)
                    else:
                        st.markdown(f"{chat['Name']}의 이미지를 불러올 수 없습니다.")
                    st.markdown(chat['설명'], unsafe_allow_html=True)











