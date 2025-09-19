"""
Fix Missing PDFs in Vector Store
"""
from chat.chatbot_service import chatbot_service
from chat.gem_downloader import GemDownloader
from langchain_community.vectorstores import FAISS
import os

def add_missing_pdfs():
    """Add missing PDFs to the main vector store"""
    print("=== ADDING MISSING PDFs ===\n")
    
    # Check current PDFs
    main_docs = chatbot_service.gem_db.similarity_search("", k=200)
    existing_pdf_ids = set(doc.metadata.get('pdf_id') for doc in main_docs if doc.metadata.get('pdf_id'))
    print(f"Existing PDFs: {sorted(existing_pdf_ids)}")
    
    # PDFs we need
    needed_pdfs = ['8103021', '8152141']
    missing_pdfs = [pdf_id for pdf_id in needed_pdfs if pdf_id not in existing_pdf_ids]
    
    if not missing_pdfs:
        print("✅ All PDFs already exist!")
        return
    
    print(f"Missing PDFs: {missing_pdfs}")
    
    # Download and process missing PDFs
    downloader = GemDownloader()
    
    for pdf_id in missing_pdfs:
        print(f"\nProcessing PDF {pdf_id}...")
        
        try:
            # Download PDF
            pdf_path = downloader.download_pdf(pdf_id)
            if not pdf_path:
                print(f"❌ Failed to download {pdf_id}")
                continue
            
            # Extract text
            content = downloader.extract_text_from_pdf(pdf_path)
            if not content:
                print(f"❌ Failed to extract text from {pdf_id}")
                continue
            
            # Create chunks
            documents = downloader.chunk_content(content, pdf_id)
            if not documents:
                print(f"❌ Failed to create chunks for {pdf_id}")
                continue
            
            print(f"✅ Created {len(documents)} chunks for {pdf_id}")
            
            # Add to main vector store
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create temporary FAISS index
            temp_db = FAISS.from_texts(texts, chatbot_service.embeddings, metadatas=metadatas)
            
            # Merge with main index
            chatbot_service.gem_db.merge_from(temp_db)
            print(f"✅ Added {pdf_id} to main vector store")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_id}: {e}")
    
    # Save updated index
    print("\nSaving updated vector store...")
    chatbot_service.gem_db.save_local("faiss_gem_clean")
    print("✅ Vector store updated and saved!")

if __name__ == "__main__":
    add_missing_pdfs()