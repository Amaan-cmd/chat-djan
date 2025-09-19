# chat/urls.py

from django.urls import path
from . import views
from .live_gem_views_clean import live_gem_page, live_gem_extract, live_gem_chat

urlpatterns = [
    path('', views.chat_view, name='chat_view'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('async/', views.async_chat_view, name='async_chat'),
    path('status/<str:task_id>/', views.chat_status_view, name='chat_status'),
    path('demo/', views.async_demo_view, name='async_demo'),
    path('test/', views.test_logging, name='test_logging'),
    # Live GeM integration
    path('live-gem/', live_gem_page, name='live_gem_page'),
    path('live-gem-extract/', live_gem_extract, name='live_gem_extract'),
    path('live-gem-chat/', live_gem_chat, name='live_gem_chat'),
]