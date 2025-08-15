from django.core.cache import cache
from django.http import JsonResponse
import time

class RateLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith('/chat/') and request.method == 'POST':
            session_key = request.session.session_key or 'anonymous'
            cache_key = f'rate_limit_{session_key}'
            
            # Get last request time
            last_request = cache.get(cache_key, 0)
            current_time = time.time()
            
            # Enforce 2-second cooldown
            if current_time - last_request < 2:
                return JsonResponse({'error': 'Please wait before sending another message'}, status=429)
            
            # Update last request time
            cache.set(cache_key, current_time, 60)  # Cache for 1 minute
        
        return self.get_response(request)