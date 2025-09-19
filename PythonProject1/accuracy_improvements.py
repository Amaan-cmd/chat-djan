"""
Accuracy Improvements for Multi-Domain Chatbot
"""

def add_query_rewriting():
    """Rewrite queries for better retrieval"""
    
    query_rewrites = {
        # GeM specific rewrites
        "bid end date": "Bid End Date/Time closing date deadline",
        "opening date": "Bid Opening Date/Time opening time",
        "ministry": "Ministry/State Name ministry department organization",
        "payment terms": "payment terms conditions timeline schedule",
        "delivery": "delivery schedule period timeline consignee",
        
        # Add synonyms for better matching
        "organisation": "organisation organization office department",
        "quantity": "total quantity amount number units",
        "officer": "reporting officer consignee contact person"
    }
    
    def rewrite_query(question):
        question_lower = question.lower()
        for key, expanded in query_rewrites.items():
            if key in question_lower:
                return f"{question} {expanded}"
        return question
    
    return rewrite_query

def add_confidence_scoring():
    """Add confidence scoring to answers"""
    
    def score_answer_confidence(question, documents, answer):
        confidence = 0.5  # Base confidence
        
        # Higher confidence for exact matches
        if any(word in answer.lower() for word in question.lower().split()):
            confidence += 0.2
        
        # Higher confidence for structured data
        if any(doc.metadata.get('extraction_type') == 'structured' for doc in documents):
            confidence += 0.2
        
        # Higher confidence for specific dates/numbers
        import re
        if re.search(r'\d{2}-\d{2}-\d{4}|\d+', answer):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    return score_answer_confidence

def add_answer_validation():
    """Validate answers before returning"""
    
    def validate_answer(question, answer, documents):
        # Check if answer actually comes from documents
        doc_content = " ".join([doc.page_content for doc in documents])
        
        # Extract key facts from answer
        import re
        dates = re.findall(r'\d{2}-\d{2}-\d{4}', answer)
        numbers = re.findall(r'\d+', answer)
        
        # Verify facts exist in source documents
        valid = True
        for date in dates:
            if date not in doc_content:
                valid = False
                break
        
        return valid, answer if valid else "I cannot find reliable information for this question in the provided documents."
    
    return validate_answer

if __name__ == "__main__":
    print("Accuracy improvements ready to implement")