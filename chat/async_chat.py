import asyncio
import threading
from django.core.cache import cache
import uuid

class AsyncChatProcessor:
    @staticmethod
    def process_chat_async(question, chat_history, session_key):
        """Process chat in background thread"""
        import os
        from datetime import datetime
        
        task_id = str(uuid.uuid4())
        log_file = os.path.join(os.path.dirname(__file__), '..', 'debug.log')
        
        with open(log_file, 'a') as f:
            f.write(f"\n>>> ASYNC PROCESSOR CALLED at {datetime.now()} <<<\n")
            f.write(f"Task ID generated: {task_id[:8]}...\n")
        
        # Set initial status
        cache.set(f"chat_status_{task_id}", {"status": "processing", "progress": "Analyzing question..."}, 60)
        
        def background_task():
            import time
            start_time = time.time()
            
            with open(log_file, 'a') as f:
                f.write(f"\n=== ASYNC TASK {task_id[:8]} STARTED ===\n")
                f.write(f"Question: {question[:50]}...\n")
            
            try:
                from .chatbot_graph import create_graph
                from .chatbot_logic import memory_saver
                from langchain_core.messages import HumanMessage, AIMessage
                
                # Update progress
                with open(log_file, 'a') as f:
                    f.write(f"[{time.time() - start_time:.2f}s] Setting up retrieval...\n")
                cache.set(f"chat_status_{task_id}", {"status": "processing", "progress": "Retrieving documents..."}, 60)
                
                chatbot_app = create_graph(checkpointer=memory_saver)
                
                # Convert chat history
                with open(log_file, 'a') as f:
                    f.write(f"[{time.time() - start_time:.2f}s] Converting chat history...\n")
                langchain_chat_history = []
                for chat in chat_history:
                    if chat['role'] == 'user':
                        langchain_chat_history.append(HumanMessage(content=chat['content']))
                    elif chat['role'] == 'ai':
                        langchain_chat_history.append(AIMessage(content=chat['content']))
                
                initial_state = {
                    "question": question,
                    "chat_history": langchain_chat_history,
                    "user_choice": ""
                }
                
                config = {"configurable": {"thread_id": session_key}}
                
                # Update progress
                with open(log_file, 'a') as f:
                    f.write(f"[{time.time() - start_time:.2f}s] Starting LLM processing...\n")
                cache.set(f"chat_status_{task_id}", {"status": "processing", "progress": "Generating response..."}, 60)
                
                # Process the chat
                with open(log_file, 'a') as f:
                    f.write(f"[{time.time() - start_time:.2f}s] Running chatbot workflow...\n")
                final_state = None
                for state in chatbot_app.stream(initial_state, config=config):
                    final_state = state
                
                if final_state and isinstance(final_state, dict) and len(final_state) == 1:
                    final_state = list(final_state.values())[0]
                
                answer = final_state.get('answer', 'Sorry, I encountered an error.') if final_state else 'No response'
                
                # Cache the response for future use
                from .cache_service import ChatCacheService
                ChatCacheService.cache_response(question, chat_history, answer)
                
                # Set final result
                total_time = time.time() - start_time
                with open(log_file, 'a') as f:
                    f.write(f"[{total_time:.2f}s] ASYNC TASK COMPLETED\n")
                    f.write(f"Answer length: {len(answer)} characters\n")
                    f.write(f"Response cached for future queries\n")
                    f.write(f"=== ASYNC TASK {task_id[:8]} FINISHED ===\n\n")
                cache.set(f"chat_status_{task_id}", {"status": "completed", "answer": answer}, 60)
                
            except Exception as e:
                error_time = time.time() - start_time
                with open(log_file, 'a') as f:
                    f.write(f"[{error_time:.2f}s] ASYNC TASK ERROR: {str(e)}\n")
                    f.write(f"=== ASYNC TASK {task_id[:8]} FAILED ===\n\n")
                cache.set(f"chat_status_{task_id}", {"status": "error", "error": str(e)}, 60)
        
        # Start background thread
        thread = threading.Thread(target=background_task)
        thread.daemon = True
        thread.start()
        
        return task_id
    
    @staticmethod
    def get_task_status(task_id):
        """Get current status of async task"""
        return cache.get(f"chat_status_{task_id}", {"status": "not_found"})