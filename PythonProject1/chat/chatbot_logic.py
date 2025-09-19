"""
Chatbot Logic - Refactored for better maintainability
"""
# Import from the new refactored modules
from .chatbot_service import chatbot_service, memory_saver, get_chatbot_service

# Maintain backward compatibility for existing imports
ChatbotService = chatbot_service.__class__