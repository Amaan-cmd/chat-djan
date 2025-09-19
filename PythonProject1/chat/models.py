from django.db import models

class ChatFeedback(models.Model):
    session_id = models.CharField(max_length=100)
    question = models.TextField()
    answer = models.TextField()
    feedback_type = models.CharField(max_length=20)  # 'thumbs_up', 'thumbs_down', 'correction'
    feedback_data = models.TextField(blank=True)  # For correction text
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.feedback_type} - {self.question[:50]}"
