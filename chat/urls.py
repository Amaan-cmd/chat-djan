# chat/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # This pattern matches the root of the app (the empty string '').
    # So when a user goes to '/chat/', this rule applies.
    # It runs the 'chat_view' function from your views.py file.
    path('', views.chat_view, name='chat_view'),
]