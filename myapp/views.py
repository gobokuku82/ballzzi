# myapp/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Create your views here.
def main_view(request):
    # 메인 페이지 로직
    return render(request, 'myapp/index.html')


from .source.question_Routing import classify
from .source.FM.FM_GetData_LLM import get_answer_from_question
from .source.FM.tools.image_craper import get_player_image_from_bing # FM 폴더 안의 tools 폴더에 있음
from .source.HR.agents.agent_executor import process_query # HR 폴더 안의 agents 폴더에 있음

# Streamlit 환경에서 필요했던 sys.modules['torch.classes'].__path__ = [] 같은 코드는
# Django 환경에서는 보통 필요 없습니다. 만약 관련 오류가 발생하면 그때 다시 고려하세요.
# import sys
# import torch
# try:
#     sys.modules['torch.classes'].__path__ = []
# except AttributeError:
#     pass


# ==========================================================
# 챗봇 페이지 뷰 (GET 요청 처리) 및 API 뷰 (POST 요청 처리)
# ==========================================================

@csrf_exempt
def chatbot_page(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_question = data.get('question', '')

            if not user_question:
                return JsonResponse({'error': 'No question provided'}, status=400)

            # --- 님의 챗봇 로직 (Streamlit app.py에서 가져온 부분) ---
            if classify(user_question):
                response_type = 'HR'
                reply = process_query(user_question)
                return JsonResponse({'type': response_type, 'answer': reply})
            
            else:
                response_type = 'FM'
                fm_replies = get_answer_from_question(user_question)
                parsed_replies = []
                for chat_item in fm_replies:
                    player_name = chat_item.get('Name', '')
                    description = chat_item.get('설명', '')
                    # 이미지 크롤러는 URL을 반환하도록 로직이 변경되어야 합니다.
                    image_url = get_player_image_from_bing(player_name)
                    parsed_replies.append({
                        'name': player_name,
                        'description': description,
                        'image_url': image_url
                    })
                return JsonResponse({'type': response_type, 'replies': parsed_replies})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            # 오류 발생 시 디버깅을 위해 에러 메시지를 포함하여 응답
            print(f"Error in chatbot_page (POST): {e}")
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

    # --- GET 요청 처리 (챗봇 HTML 페이지 렌더링) ---
    else: # request.method == 'GET'
        chat_messages = [] # 초기에는 빈 메시지 리스트 전달
        return render(request, 'myapp/chatbot.html', {'chat_messages': chat_messages})
