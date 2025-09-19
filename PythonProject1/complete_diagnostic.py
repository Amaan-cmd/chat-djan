"""
Complete System Diagnostic
"""
from chat.chatbot_service import chatbot_service

def complete_diagnostic():
    print("=== COMPLETE SYSTEM DIAGNOSTIC ===\n")
    
    # 1. Check available PDF IDs
    print("1. CHECKING AVAILABLE PDF IDs:")
    if hasattr(chatbot_service, 'gem_db') and chatbot_service.gem_db:
        all_docs = chatbot_service.gem_db.similarity_search("", k=200)
        pdf_ids = set()
        for doc in all_docs:
            pdf_id = doc.metadata.get('pdf_id')
            if pdf_id:
                pdf_ids.add(pdf_id)
        
        print(f"Available PDF IDs: {sorted(pdf_ids)}")
        
        # Check if our test IDs exist
        test_ids = ['8103021', '8152141']
        for test_id in test_ids:
            if test_id in pdf_ids:
                print(f"✅ {test_id} EXISTS in vector store")
            else:
                print(f"❌ {test_id} NOT FOUND in vector store")
    
    print("\n" + "="*50)
    
    # 2. Test retrieval for both documents
    print("2. TESTING RETRIEVAL:")
    test_cases = [
        ("8103021", "bid end date"),
        ("8152141", "bid end date"),
        ("8103021", "bid offer validity"),
        ("8152141", "bid offer validity")
    ]
    
    for pdf_id, question in test_cases:
        print(f"\nTesting: {pdf_id} - '{question}'")
        docs = chatbot_service.scoped_gem_search(question, pdf_id=pdf_id, k=5)
        print(f"Retrieved: {len(docs)} documents")
        
        if docs:
            for i, doc in enumerate(docs[:2]):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
                print(f"  Metadata: {doc.metadata}")
        else:
            print("  No documents retrieved!")
    
    print("\n" + "="*50)
    
    # 3. Check vector store structure
    print("3. VECTOR STORE STRUCTURE:")
    if hasattr(chatbot_service, 'gem_db') and chatbot_service.gem_db:
        sample_docs = chatbot_service.gem_db.similarity_search("bid", k=10)
        
        extraction_types = {}
        sources = {}
        
        for doc in sample_docs:
            ext_type = doc.metadata.get('extraction_type', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            extraction_types[ext_type] = extraction_types.get(ext_type, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        print(f"Extraction types: {extraction_types}")
        print(f"Sources (top 5): {dict(list(sources.items())[:5])}")
    
    print("\n" + "="*50)
    
    # 4. Test specific content search
    print("4. CONTENT SEARCH TEST:")
    search_terms = ["Bid End Date/Time", "Bid Offer Validity", "8103021", "8152141"]
    
    for term in search_terms:
        docs = chatbot_service.gem_db.similarity_search(term, k=3)
        print(f"'{term}': {len(docs)} results")
        if docs:
            pdf_ids_found = [doc.metadata.get('pdf_id') for doc in docs]
            print(f"  PDF IDs: {pdf_ids_found}")

if __name__ == "__main__":
    complete_diagnostic()