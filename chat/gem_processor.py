"""
GeM Document Processing Module
"""
import re
from langchain_core.documents import Document

class GemProcessor:
    def __init__(self, gem_db, llm):
        self.gem_db = gem_db
        self.llm = llm
    
    def setup_gem_chain(self):
        """Setup GeM procurement QA chain"""
        def gem_chain_invoke(inputs):
            question = inputs.get('input', '')
            context_docs = inputs.get('context', [])
            
            if not context_docs:
                return "No relevant documents found for this GeM query."
            
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
            direct_prompt = f"""
You are a GeM procurement document analyst. Analyze the following document content carefully:

{context_text}

Question: {question}

INSTRUCTIONS:
1. Read through ALL the content thoroughly, including paragraphs, lists, and any structured data
2. Look for information in both tabular format AND narrative text
3. If you find exact matches (dates, numbers, names), quote them precisely
4. If information is scattered across multiple sections, synthesize it coherently
5. If the answer requires interpretation of policy text or procedures, explain clearly
6. If no relevant information exists, state "This information is not available in the provided document"

Provide a comprehensive answer based on the document content:
"""
            
            response = self.llm.invoke(direct_prompt)
            return response.content
        
        class DirectChain:
            def __init__(self, func):
                self.invoke = func
        
        return DirectChain(gem_chain_invoke)
    
    def smart_gem_search(self, question: str, k: int = 8):
        """Smart GeM search for document 8046605"""
        if not self.gem_db:
            return []
        
        # Simple semantic search - no multi-document complexity
        return self.gem_db.similarity_search(question, k=k)
    
    def hybrid_gem_extraction(self, question: str, doc_number: str):
        """Hybrid extraction: Regex + Table parsing + Semantic fallback"""
        print(f"Using hybrid extraction for document {doc_number}")
        
        all_docs = self.gem_db.similarity_search(doc_number, k=50)
        doc_chunks = [doc for doc in all_docs if doc_number in doc.metadata.get('source', '')]
        
        if not doc_chunks:
            return None
        
        full_text = "\n".join([doc.page_content for doc in doc_chunks])
        
        result = self.extract_structured_field(question, full_text, doc_number)
        if result:
            return result
        
        print("Structured extraction failed, using semantic fallback")
        return self.smart_gem_search(question)
    
    def extract_structured_field(self, question: str, text: str, doc_number: str):
        """Extract structured fields using regex patterns - only for very specific queries"""
        question_lower = question.lower()
        
        # Only use structured extraction for very specific field queries
        specific_field_queries = {
            'bid opening date': r'Bid Opening Date/Time[^\n]*?([0-9]{2}-[0-9]{2}-[0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})',
            'bid opening time': r'Bid Opening Date/Time[^\n]*?([0-9]{2}-[0-9]{2}-[0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})',
            'bid end date': r'Bid End[^\n]*?([0-9]{2}-[0-9]{2}-[0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})',
            'bid end time': r'Bid End[^\n]*?([0-9]{2}-[0-9]{2}-[0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})',
        }
        
        # Only extract if the question is asking for a very specific field
        for field_query, pattern in specific_field_queries.items():
            if field_query in question_lower and len(question_lower.split()) <= 6:  # Short, specific queries only
                print(f"Detected specific field query: {field_query}")
                
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    print(f"Extracted value: {value}")
                    
                    response = f"According to GeM-Bidding-{doc_number}, the {field_query.title()} is **{value}**."
                    return [Document(
                        page_content=response, 
                        metadata={"source": f"GeM-Bidding-{doc_number}.pdf", "extraction_type": "structured"}
                    )]
        
        # For all other queries (like "terms and conditions"), return None to use semantic search
        print(f"Not a specific field query - using semantic search for: {question}")
        return None