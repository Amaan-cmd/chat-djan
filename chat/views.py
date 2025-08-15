# chat/views.py

import logging
from django.shortcuts import render

logger = logging.getLogger(__name__)
from django.http import JsonResponse
from .models import ChatFeedback
from langchain_core.messages import HumanMessage, AIMessage
from .chatbot_graph import create_graph
from .chatbot_logic import memory_saver
from .cache_service import ChatCacheService
from .async_chat import AsyncChatProcessor
import re
import json

chatbot_app = create_graph(checkpointer=memory_saver)

def chat_view(request):
    chat_history = request.session.get('chat_history', [])

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()
        
        # Input validation
        display_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        if not question:
            return render(request, 'chat/chat.html', {'chat_history': display_history, 'error': 'Please enter a question'})
        if len(question) > 1000:
            return render(request, 'chat/chat.html', {'chat_history': display_history, 'error': 'Question too long (max 1000 characters)'})
        if len(question) < 3:
            return render(request, 'chat/chat.html', {'chat_history': display_history, 'error': 'Question too short (min 3 characters)'})
        
        # Build chat history from session
        langchain_chat_history = []
        session_history = request.session.get('chat_history', [])
        
        # Convert session history to LangChain messages
        for chat in session_history:
            if chat['role'] == 'user':
                langchain_chat_history.append(HumanMessage(content=chat['content']))
            elif chat['role'] == 'ai':
                langchain_chat_history.append(AIMessage(content=chat['content']))
        
        # Check if this is a disambiguation choice
        if question.lower() in ['calamity', 'general']:
            original_question = request.session.get('original_question', 'unknown')
            initial_state = {
                "question": original_question,
                "chat_history": langchain_chat_history,
                "user_choice": question.lower()
            }
            request.session['original_question'] = ''
        else:
            initial_state = {
                "question": question,
                "chat_history": langchain_chat_history,
                "user_choice": ""  # Clear any previous choice
            }

        # Check cache for quick responses
        cached_response = ChatCacheService.get_cached_response(question, langchain_chat_history)
        if cached_response:
            print("---CACHE HIT: Using cached response---")
            answer = cached_response
        else:
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
                
                # Cache the response for future use
                ChatCacheService.cache_response(question, langchain_chat_history, answer)
                    
            except Exception as e:
                logger.error(f"Chat processing error for question '{question[:50]}...': {str(e)}")
                
                # Check for API key issues
                if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                    answer = "Configuration error: Please contact the administrator to update the API key."
                else:
                    answer = "I'm experiencing technical difficulties. Please try again in a moment."

        # Add current exchange to chat history
        existing_history = request.session.get('chat_history', [])
        chat_history = existing_history + [
            {'role': 'user', 'content': question},
            {'role': 'ai', 'content': answer}
        ]

    request.session['chat_history'] = chat_history
    # Show only last 3 exchanges (6 messages) to user
    display_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    return render(request, 'chat/chat.html', {'chat_history': display_history})

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

def async_chat_view(request):
    """Start async chat processing"""
    if request.method == 'POST':
        import sys
        import os
        from datetime import datetime
        
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        
        # Log to file since terminal doesn't show
        log_file = os.path.join(os.path.dirname(__file__), '..', 'debug.log')
        with open(log_file, 'a') as f:
            f.write(f"\n>>> ASYNC ENDPOINT HIT at {datetime.now()} <<<\n")
            f.write(f"Raw question: '{question}'\n")
        
        # Input validation
        if not question or len(question) < 3 or len(question) > 1000:
            return JsonResponse({'status': 'error', 'message': 'Invalid question'})
        
        # Check cache first
        chat_history = request.session.get('chat_history', [])
        cached_response = ChatCacheService.get_cached_response(question, chat_history)
        if cached_response:
            with open(log_file, 'a') as f:
                f.write(f"\n=== INSTANT CACHE HIT ===\n")
                f.write(f"Question: {question[:50]}...\n")
                f.write(f"Cached response length: {len(cached_response)} characters\n")
                f.write(f"Response time: <0.01s (INSTANT!)\n")
                f.write(f"=== CACHE HIT COMPLETE ===\n\n")
            return JsonResponse({'status': 'completed', 'answer': cached_response})
        
        # Start async processing
        with open(log_file, 'a') as f:
            f.write(f"\n=== STARTING ASYNC PROCESSING ===\n")
            f.write(f"Question: {question[:50]}...\n")
        
        task_id = AsyncChatProcessor.process_chat_async(
            question, chat_history, request.session.session_key
        )
        
        with open(log_file, 'a') as f:
            f.write(f"Task ID: {task_id[:8]}...\n")
            f.write(f"User will get INSTANT response with task_id\n")
            f.write(f"=== ASYNC TASK QUEUED ===\n\n")
        
        return JsonResponse({'status': 'processing', 'task_id': task_id})
    
    return JsonResponse({'status': 'error'})

def chat_status_view(request, task_id):
    """Get status of async chat processing"""
    status = AsyncChatProcessor.get_task_status(task_id)
    return JsonResponse(status)

def async_demo_view(request):
    """Demo page for async chat"""
    import sys
    import os
    from datetime import datetime
    
    # Log to both terminal and file
    log_msg = f"\n=== ASYNC DEMO PAGE LOADED at {datetime.now()} ===\n"
    print(log_msg)
    sys.stdout.flush()
    
    # Also write to log file
    log_file = os.path.join(os.path.dirname(__file__), '..', 'debug.log')
    with open(log_file, 'a') as f:
        f.write(log_msg)
        f.write("User accessed /chat/demo/\n")
        f.write("=== DEMO PAGE READY ===\n\n")
    
    return render(request, 'chat/async_chat.html')

def test_logging(request):
    """Simple test to verify logging works"""
    import sys
    import os
    from datetime import datetime
    
    # Write to both terminal AND file
    log_msg = f"\n*** LOGGING TEST at {datetime.now()} ***\n"
    print(log_msg)
    sys.stdout.flush()
    
    # Also write to a log file
    log_file = os.path.join(os.path.dirname(__file__), '..', 'debug.log')
    with open(log_file, 'a') as f:
        f.write(log_msg)
        f.write(f"Request method: {request.method}\n")
        f.write(f"Request path: {request.path}\n")
        f.write("*** TEST COMPLETE ***\n\n")
    
    return JsonResponse({'message': f'Check terminal AND {log_file} for logs!', 'status': 'success'})