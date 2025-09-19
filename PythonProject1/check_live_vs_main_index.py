"""
Check Live vs Main Index Issue
"""
import os
from chat.chatbot_service import chatbot_service

def check_index_issue():
    print("=== CHECKING INDEX ISSUE ===\n")
    
    # Check which index is being used
    print("1. CHECKING ACTIVE INDEX:")
    if hasattr(chatbot_service, 'gem_db'):
        print("✅ Main GeM index loaded")
    else:
        print("❌ Main GeM index NOT loaded")
    
    # Check if live index exists
    live_path = "faiss_gem_live"
    if os.path.exists(live_path):
        print("⚠️  Live index exists - might be interfering")
        
        # Load live index and check contents
        from langchain_community.vectorstores import FAISS
        try:
            live_db = FAISS.load_local(live_path, chatbot_service.embeddings, allow_dangerous_deserialization=True)
            live_docs = live_db.similarity_search("", k=50)
            live_pdf_ids = set(doc.metadata.get('pdf_id') for doc in live_docs if doc.metadata.get('pdf_id'))
            print(f"Live index PDF IDs: {sorted(live_pdf_ids)}")
        except Exception as e:
            print(f"Error loading live index: {e}")
    else:
        print("✅ No live index found")
    
    print("\n2. CHECKING MAIN INDEX CONTENTS:")
    main_docs = chatbot_service.gem_db.similarity_search("", k=100)
    main_pdf_ids = set(doc.metadata.get('pdf_id') for doc in main_docs if doc.metadata.get('pdf_id'))
    print(f"Main index PDF IDs: {sorted(main_pdf_ids)}")
    
    # Check if our test PDFs are in main index
    test_ids = ['8103021', '8152141']
    for test_id in test_ids:
        if test_id in main_pdf_ids:
            print(f"✅ {test_id} in main index")
            # Count chunks for this PDF
            chunks = [doc for doc in main_docs if doc.metadata.get('pdf_id') == test_id]
            print(f"   {len(chunks)} chunks found")
        else:
            print(f"❌ {test_id} NOT in main index")
    
    print("\n3. TESTING DIRECT SEARCH:")
    for test_id in test_ids:
        print(f"\nDirect search for {test_id}:")
        docs = chatbot_service.gem_db.similarity_search(test_id, k=5)
        print(f"Found {len(docs)} documents")
        for doc in docs[:2]:
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  PDF ID: {doc.metadata.get('pdf_id')}")

if __name__ == "__main__":
    check_index_issue()