# myapp/urls.py
from django.urls import path
from . import views

app_name = 'myapp' # 앱 네임스페이스
urlpatterns = [
    path('', views.chatbot_page, name='chatbot_page'),
]