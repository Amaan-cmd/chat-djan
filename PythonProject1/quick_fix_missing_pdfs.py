"""
Quick Fix: Add Missing PDFs to Main Index
"""
import os
import shutil

def quick_fix():
    """Move live index content to main index"""
    
    # 1. Check if live index has the missing PDF
    live_path = "faiss_gem_live"
    main_path = "faiss_gem_clean"
    
    if os.path.exists(live_path):
        print("Found live index with 8152141")
        
        # Load both indexes
        from chat.chatbot_service import chatbot_service
        from langchain_community.vectorstores import FAISS
        
        live_db = FAISS.load_local(live_path, chatbot_service.embeddings, allow_dangerous_deserialization=True)
        
        # Merge live into main
        chatbot_service.gem_db.merge_from(live_db)
        
        # Save updated main index
        chatbot_service.gem_db.save_local(main_path)
        
        # Remove live index to prevent conflicts
        shutil.rmtree(live_path)
        
        print("✅ Merged live index into main index")
        print("✅ Removed live index to prevent conflicts")
        
        # Verify
        updated_docs = chatbot_service.gem_db.similarity_search("", k=200)
        pdf_ids = set(doc.metadata.get('pdf_id') for doc in updated_docs if doc.metadata.get('pdf_id'))
        
        if '8152141' in pdf_ids:
            print("✅ 8152141 now available in main index")
        
        print(f"Total PDFs now: {len(pdf_ids)}")
    else:
        print("No live index found")

if __name__ == "__main__":
    quick_fix()