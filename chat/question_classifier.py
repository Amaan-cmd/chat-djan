"""
Question Classification Module
"""
import re
import numpy as np

class QuestionClassifier:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
    
    def classify_question_type(self, question: str) -> str:
        """Classify question using semantic search with keyword fallback"""
        # Check if question mentions specific document number
        if re.search(r'\b\d{7}\b', question):
            print(f"Document-specific question detected - classifying as gem")
            return "gem"
        
        # Check for GeM-related terms
        gem_indicators = ['bidding', 'bid', 'tender', 'procurement', 'gem', 'ministry', 'organisation', 'item category', 'documents']
        if any(indicator in question.lower() for indicator in gem_indicators):
            print(f"GeM-related question detected - classifying as gem")
            return "gem"
        
        # Try semantic classification first
        semantic_result = self._classify_semantic(question)
        if semantic_result != "unclear":
            return semantic_result
        
        # Fallback to keyword matching
        return self._classify_keywords(question)
    
    def _classify_semantic(self, question: str) -> str:
        """Semantic classification using embeddings"""
        try:
            question_embedding = self.embeddings.embed_query(question)
            
            category_descriptions = {
                "gem": "government procurement bidding tender contract ministry defence department supplier vendor purchase proposal military equipment services maintenance annual contract GeM marketplace public sector acquisition",
                "calamity": "terraria calamity mod boss weapon item crafting recipe strategy guide gaming video game yharon providence devourer astrum supreme calamitas scal draedon exo mechs",
                "general": "general knowledge facts information science history geography mathematics basic questions everyday topics"
            }
            
            similarities = {}
            for category, description in category_descriptions.items():
                category_embedding = self.embeddings.embed_query(description)
                similarity = self._cosine_similarity(question_embedding, category_embedding)
                similarities[category] = similarity
            
            best_category = max(similarities, key=similarities.get)
            best_score = similarities[best_category]
            
            if best_score > 0.55:
                print(f"Semantic classification: {best_category} (confidence: {best_score:.3f})")
                return best_category
            else:
                print(f"Semantic classification unclear (best: {best_category}, score: {best_score:.3f})")
                return "unclear"
                
        except Exception as e:
            print(f"Semantic classification failed: {e}")
            return "unclear"
    
    def _classify_keywords(self, question: str) -> str:
        """Fallback keyword classification"""
        question_lower = question.lower()
        
        gem_keywords = ['gem', 'procurement', 'bidding', 'tender', 'ministry', 'government', 'bid', 'contract', 'purchase', 'supplier', 'vendor', 'amc', 'maintenance', 'defence', 'department', 'proposal']
        calamity_keywords = ['calamity', 'terraria', 'boss', 'weapon', 'item', 'mod', 'yharon', 'supreme', 'devourer', 'providence', 'astrum']
        
        if any(keyword in question_lower for keyword in gem_keywords):
            print("Keyword classification: gem")
            return "gem"
        
        if any(keyword in question_lower for keyword in calamity_keywords):
            print("Keyword classification: calamity")
            return "calamity"
        
        print("Keyword classification: general")
        return "general"
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)