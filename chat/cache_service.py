from functools import lru_cache
from django.core.cache import cache
import hashlib
import json

class ChatCacheService:
    @staticmethod
    def get_query_hash(question, chat_history):
        """Generate hash for caching based on question and recent history"""
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
        cache_key = f"{question}_{str(recent_history)}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    @staticmethod
    def get_cached_response(question, chat_history):
        """Get cached response if available"""
        cache_key = f"chat_response_{ChatCacheService.get_query_hash(question, chat_history)}"
        return cache.get(cache_key)
    
    @staticmethod
    def cache_response(question, chat_history, response, timeout=300):
        """Cache response for 5 minutes"""
        cache_key = f"chat_response_{ChatCacheService.get_query_hash(question, chat_history)}"
        cache.set(cache_key, response, timeout)
    
    @staticmethod
    def get_cached_documents(question):
        """Get cached document retrieval"""
        cache_key = f"docs_{hashlib.md5(question.encode()).hexdigest()}"
        return cache.get(cache_key)
    
    @staticmethod
    def cache_documents(question, documents, timeout=600):
        """Cache documents for 10 minutes"""
        cache_key = f"docs_{hashlib.md5(question.encode()).hexdigest()}"
        # Store document content and metadata
        doc_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        cache.set(cache_key, doc_data, timeout)