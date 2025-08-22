"""
GeM Document Processing Module
"""
import re
from langchain.schema import Document

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
            
            # Check if this is a multi-document query
            multi_doc_indicators = ['each pdf', 'all pdf', 'all documents', 'each document', 'systematic manner', 'compare', 'list all', 'all the bid', 'all bid documents', 'in all']
            is_multi_doc = any(indicator in question.lower() for indicator in multi_doc_indicators)
            
            if is_multi_doc:
                doc_sources = set()
                for doc in context_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    if source != 'Unknown':
                        doc_sources.add(source)
                
                sources_info = f"\nDocument sources found: {', '.join(sorted(doc_sources))}\n" if doc_sources else ""
                
                direct_prompt = f"""
You are analyzing multiple GeM procurement documents to extract bid opening times. The content below comes from different PDF documents:

{sources_info}
{context_text}

Question: {question}

INSTRUCTIONS FOR BID OPENING TIME EXTRACTION:
1. Search through ALL the content for "Bid Opening Date/Time" or similar phrases
2. Look for date-time patterns like "DD-MM-YYYY HH:MM:SS" (e.g., "14-08-2025 12:30:00")
3. For each document source mentioned, find its corresponding bid opening time
4. Present results in a clear table format with columns: Document Source | Bid Opening Date/Time
5. If you find multiple documents, list ALL of them systematically
6. Ignore Hindi text and focus on English date/time information
7. If no bid opening time is found for a document, state "Not found in provided content"

Create a comprehensive table showing bid opening times from ALL available documents:
"""
            else:
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
        """Smart GeM search with document-specific filtering"""
        if not self.gem_db:
            return []
        
        # Check for multi-document queries
        multi_doc_indicators = ['each pdf', 'all pdf', 'all documents', 'each document', 'systematic manner', 'compare', 'list all', 'all the bid', 'all bid documents', 'in all']
        is_multi_doc_query = any(indicator in question.lower() for indicator in multi_doc_indicators)
        
        if is_multi_doc_query:
            print("Multi-document query detected - searching across all PDFs")
            return self.multi_document_search(question, k=50)
        
        # Extract document number if mentioned
        doc_pattern = r'\b(\d{7})\b'
        doc_match = re.search(doc_pattern, question)
        
        if doc_match:
            doc_number = doc_match.group(1)
            print(f"Searching specifically in document: {doc_number}")
            
            search_strategies = [question, doc_number]
            
            if any(term in question.lower() for term in ['document', 'required', 'seller', 'upload', 'eligibility']):
                search_strategies.extend(["documents required", "seller documents", "eligibility"])
            elif any(term in question.lower() for term in ['bid', 'opening', 'date', 'time']):
                search_strategies.extend(["Bid Opening Date/Time", "bid opening", "Bid Details"])
            elif any(term in question.lower() for term in ['validity', 'period', 'duration']):
                search_strategies.extend(["Bid Offer Validity", "validity period"])
            else:
                search_strategies.extend(["terms and conditions", "specifications", "requirements"])
            
            all_docs = []
            seen_content = set()
            
            for strategy in search_strategies:
                docs = self.gem_db.similarity_search(strategy, k=15)
                for doc in docs:
                    source = doc.metadata.get('source', '')
                    content_hash = hash(doc.page_content[:100])
                    
                    if (doc_number in source and content_hash not in seen_content and len(all_docs) < k):
                        all_docs.append(doc)
                        seen_content.add(content_hash)
                        print(f"Found chunk from {source} using strategy '{strategy}'")
            
            if all_docs:
                print(f"Found {len(all_docs)} diverse chunks from document {doc_number}")
                return all_docs
            else:
                print(f"No chunks found for document {doc_number}, using general search")
        
        return self.gem_db.similarity_search(question, k=k)
    
    def multi_document_search(self, question: str, k: int = 50):
        """Search across ALL GeM documents ensuring complete coverage"""
        if not self.gem_db:
            return []
        
        # All 8 document numbers we need to cover
        required_docs = ['7893321', '7908419', '7975925', '7987151', '8046605', '8089475', '8102343', '8127013']
        doc_coverage = {}
        
        # Strategy 1: Direct search for bid opening information
        bid_opening_terms = [
            'Bid Opening Date/Time', 'bid opening', 'opening date', 'opening time', 
            'Bid Details', '09:30:00', '10:30:00', '11:30:00', '12:30:00', '14:30:00'
        ]
        
        for term in bid_opening_terms:
            docs = self.gem_db.similarity_search(term, k=60)
            for doc in docs:
                source = doc.metadata.get('source', 'unknown')
                if source not in doc_coverage:
                    doc_coverage[source] = []
                if len(doc_coverage[source]) < 6:  # Take up to 6 chunks per document
                    doc_coverage[source].append(doc)
        
        # Strategy 2: Ensure ALL 8 documents are represented
        for doc_num in required_docs:
            doc_found = any(doc_num in source for source in doc_coverage.keys())
            if not doc_found:
                print(f"Missing document {doc_num}, searching aggressively...")
                
                # Try multiple search approaches for missing documents
                search_strategies = [
                    doc_num,
                    f"GeM-Bidding-{doc_num}",
                    f"{doc_num} bid opening",
                    "summary",  # Look for summary chunks
                    "bid_opening"  # Look for specific chunk types
                ]
                
                for strategy in search_strategies:
                    missing_docs = self.gem_db.similarity_search(strategy, k=40)
                    for doc in missing_docs:
                        source = doc.metadata.get('source', '')
                        if doc_num in source:
                            if source not in doc_coverage:
                                doc_coverage[source] = []
                            if len(doc_coverage[source]) < 4:
                                doc_coverage[source].append(doc)
                                print(f"Found missing document {doc_num} using strategy '{strategy}'")
                    
                    # Check if we found the document
                    if any(doc_num in source for source in doc_coverage.keys()):
                        break
        
        # Strategy 3: If still missing documents, do a final sweep
        covered_docs = [doc_num for doc_num in required_docs if any(doc_num in source for source in doc_coverage.keys())]
        missing_docs = [doc_num for doc_num in required_docs if doc_num not in covered_docs]
        
        if missing_docs:
            print(f"Still missing {len(missing_docs)} documents: {missing_docs}")
            # Get ANY chunks from missing documents
            all_chunks = self.gem_db.similarity_search("", k=200)  # Get many chunks
            for doc in all_chunks:
                source = doc.metadata.get('source', '')
                for missing_doc in missing_docs:
                    if missing_doc in source:
                        if source not in doc_coverage:
                            doc_coverage[source] = []
                        if len(doc_coverage[source]) < 2:
                            doc_coverage[source].append(doc)
        
        # Compile results with priority for bid opening chunks
        all_docs = []
        for source, docs in doc_coverage.items():
            # Sort chunks by relevance to bid opening
            sorted_docs = sorted(docs, key=lambda d: (
                'bid opening' in d.page_content.lower(),
                'summary' in d.metadata.get('chunk_type', ''),
                'bid_opening' in d.metadata.get('chunk_type', '')
            ), reverse=True)
            
            all_docs.extend(sorted_docs)
            print(f"Added {len(sorted_docs)} chunks from {source}")
        
        final_count = len(doc_coverage)
        print(f"Multi-document search: {len(all_docs)} chunks from {final_count}/8 documents")
        
        # Ensure we have representation from all 8 documents
        if final_count < 8:
            print(f"WARNING: Only found {final_count}/8 documents. Missing coverage!")
        
        return all_docs[:k]
    
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