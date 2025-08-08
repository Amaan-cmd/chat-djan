# chat/views.py

from django.shortcuts import render
from django.http import JsonResponse
from .models import ChatFeedback
# from langchain_core.messages import HumanMessage, AIMessage #<-- This import is not used
from .chatbot_graph import create_graph
from .chatbot_logic import memory_saver
import re

chatbot_app = create_graph(checkpointer=memory_saver)

def chat_view(request):
    chat_history = request.session.get('chat_history', [])

    if request.method == 'POST':
        question = request.POST.get('question', '')
        
        # Check if this is a disambiguation choice
        if question.lower() in ['calamity', 'general']:
            original_question = request.session.get('original_question', 'unknown')
            initial_state = {
                "question": original_question,
                "chat_history": [],
                "user_choice": question.lower()
            }
            request.session['original_question'] = ''
        else:
            initial_state = {
                "question": question,
                "chat_history": [],
                "user_choice": ""  # Clear any previous choice
            }

        config = {"configurable": {"thread_id": request.session.session_key}}
        
        try:
            # Handle workflow that might end with disambiguation
            final_state = None
            for state in chatbot_app.stream(initial_state, config=config):
                final_state = state
            
            if final_state and isinstance(final_state, dict) and len(final_state) == 1:
                final_state = list(final_state.values())[0]
            
            answer = final_state.get('answer', 'Sorry, I encountered an error.') if final_state else 'No response'
            
            # Store original question if disambiguation triggered
            if final_state and final_state.get('generation_source') == 'disambiguation':
                request.session['original_question'] = question
                
        except Exception as e:
            answer = f"Error: {str(e)}"

        chat_history = [
            {'role': 'user', 'content': question},
            {'role': 'ai', 'content': answer}
        ]

    request.session['chat_history'] = chat_history
    return render(request, 'chat/chat.html', {'chat_history': chat_history})

def feedback_view(request):
    if request.method == 'POST':
        feedback_type = request.POST.get('feedback_type')
        question = request.POST.get('question', '')
        answer = request.POST.get('answer', '')
        
        ChatFeedback.objects.create(
            session_id=request.session.session_key,
            question=question,
            answer=answer,
            feedback_type=feedback_type
        )
        
        return JsonResponse({'status': 'success', 'message': 'Thanks for your feedback!'})
    
    return JsonResponse({'status': 'error'})