"""
Migration script to populate multi-vector retriever with existing GeM data
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
import django
django.setup()

from chat.chatbot_service import get_chatbot_service
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def migrate_existing_data():
    """Migrate existing FAISS data to multi-vector retriever"""
    print("🚀 Starting migration to multi-vector retriever...")
    
    # Get chatbot service
    chatbot_service = get_chatbot_service()
    
    # Check if existing GeM data exists
    gem_paths = ["faiss_gem_clean", "faiss_gem_index", "faiss_hybrid_index"]
    source_path = None
    
    for path in gem_paths:
        if os.path.exists(path):
            source_path = path
            break
    
    if not source_path:
        print("❌ No existing GeM vector store found")
        return False
    
    print(f"📂 Found existing data at: {source_path}")
    
    try:
        # Load existing FAISS store
        existing_store = FAISS.load_local(
            source_path,
            chatbot_service.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get all documents from existing store
        print("📄 Extracting documents from existing store...")
        
        # Get document count
        doc_count = existing_store.index.ntotal
        print(f"📊 Found {doc_count} documents in existing store")
        
        if doc_count == 0:
            print("⚠️ No documents found in existing store")
            return False
        
        # Search for all documents (using broad query)
        all_docs = existing_store.similarity_search("", k=min(doc_count, 1000))  # Limit for safety
        
        if not all_docs:
            # Try alternative extraction method
            print("🔄 Trying alternative extraction...")
            all_docs = existing_store.similarity_search("GeM bidding procurement", k=min(doc_count, 1000))
        
        print(f"📋 Extracted {len(all_docs)} documents")
        
        if not all_docs:
            print("❌ Could not extract documents from existing store")
            return False
        
        # Clear existing multi-vector store
        print("🧹 Clearing existing multi-vector store...")
        chatbot_service.multi_vector_retriever.clear_store()
        
        # Add documents to multi-vector retriever in batches
        batch_size = 50
        total_docs = len(all_docs)
        
        for i in range(0, total_docs, batch_size):
            batch = all_docs[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} docs)...")
            
            # Add batch to multi-vector retriever
            # Disable summaries for large migrations to save API calls
            generate_summaries = len(batch) <= 10
            chatbot_service.multi_vector_retriever.add_documents(
                batch, 
                generate_summaries=generate_summaries
            )
        
        # Get final stats
        stats = chatbot_service.multi_vector_retriever.get_stats()
        print(f"✅ Migration completed!")
        print(f"📊 Final stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_vector_retrieval():
    """Test the multi-vector retriever with sample queries"""
    print("\n🧪 Testing multi-vector retrieval...")
    
    chatbot_service = get_chatbot_service()
    
    test_queries = [
        "bid opening date",
        "ministry name",
        "total quantity",
        "delivery schedule",
        "technical specifications"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        try:
            results = chatbot_service.enhanced_gem_search(query, k=3)
            print(f"   ✅ Found {len(results)} results")
            
            if results:
                # Show first result preview
                first_result = results[0]
                preview = first_result.page_content[:200] + "..." if len(first_result.page_content) > 200 else first_result.page_content
                print(f"   📄 Preview: {preview}")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Vector Retriever Migration")
    print("=" * 60)
    
    # Run migration
    success = migrate_existing_data()
    
    if success:
        # Test the new system
        test_multi_vector_retrieval()
        print("\n🎉 Migration and testing completed!")
    else:
        print("\n❌ Migration failed!")