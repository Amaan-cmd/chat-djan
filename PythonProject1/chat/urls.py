# chat/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat_view'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('async/', views.async_chat_view, name='async_chat'),
    path('status/<str:task_id>/', views.chat_status_view, name='chat_status'),
    path('demo/', views.async_demo_view, name='async_demo'),
    path('test/', views.test_logging, name='test_logging'),
]