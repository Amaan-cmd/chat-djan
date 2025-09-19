"""
Pre-test validation script to check all imports and basic setup
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def validate_imports():
    """Validate all required imports"""
    print("🔍 Validating imports...")
    
    try:
        # Test Django setup
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
        import django
        django.setup()
        print("   ✅ Django setup successful")
        
        # Test core imports
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("   ✅ Text splitters import successful")
        
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        print("   ✅ MultiVectorRetriever import successful")
        
        from langchain.storage import InMemoryByteStore
        print("   ✅ InMemoryByteStore import successful")
        
        from langchain_community.vectorstores import FAISS
        print("   ✅ FAISS import successful")
        
        # Test our modules
        from chat.chatbot_service import get_chatbot_service
        print("   ✅ Chatbot service import successful")
        
        from chat.live_multi_vector_processor import LiveMultiVectorProcessor
        print("   ✅ Live multi-vector processor import successful")
        
        from chat.live_gem_views_clean import get_live_answer
        print("   ✅ Live GeM views import successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_chatbot_service():
    """Validate chatbot service initialization"""
    print("\n🤖 Validating chatbot service...")
    
    try:
        from chat.chatbot_service import get_chatbot_service
        
        chatbot_service = get_chatbot_service()
        print("   ✅ Chatbot service initialized")
        
        # Check if embeddings work
        if hasattr(chatbot_service, 'embeddings'):
            print("   ✅ Embeddings available")
        else:
            print("   ⚠️ Embeddings not found")
            
        # Check if LLM works
        if hasattr(chatbot_service, 'llm'):
            print("   ✅ LLM available")
        else:
            print("   ⚠️ LLM not found")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Chatbot service validation failed: {e}")
        return False

def validate_live_processor():
    """Validate live multi-vector processor"""
    print("\n📄 Validating live processor...")
    
    try:
        from chat.chatbot_service import get_chatbot_service
        from chat.live_multi_vector_processor import LiveMultiVectorProcessor
        
        chatbot_service = get_chatbot_service()
        
        # Try to initialize processor
        processor = LiveMultiVectorProcessor(
            embeddings=chatbot_service.embeddings,
            llm=chatbot_service.llm
        )
        print("   ✅ Live processor initialized")
        
        # Check stats
        stats = processor.get_live_stats()
        print(f"   📊 Initial stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Live processor validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validations"""
    print("🚀 Pre-Test Validation Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Run validations
    if not validate_imports():
        all_passed = False
        
    if not validate_chatbot_service():
        all_passed = False
        
    if not validate_live_processor():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All validations passed! Ready for testing.")
        print("\nNext steps:")
        print("1. Run: python test_live_multi_vector.py")
        print("2. Test live GeM extraction in your web interface")
    else:
        print("❌ Some validations failed. Fix issues before testing.")
        
    return all_passed

if __name__ == "__main__":
    main()