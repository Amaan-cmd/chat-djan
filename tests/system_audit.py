#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete System Audit - Check for Issues Before Production
"""
import os
import sys
import django

# Encoding fix
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
django.setup()

def audit_system():
    """Complete system audit"""
    
    print("=== SYSTEM AUDIT FOR PRODUCTION READINESS ===\n")
    
    issues = []
    warnings = []
    
    # 1. Check FAISS index
    print("1. CHECKING FAISS INDEX...")
    if os.path.exists("faiss_gem_clean"):
        print("   ✅ Clean FAISS index exists")
        
        # Check index files
        if os.path.exists("faiss_gem_clean/index.faiss") and os.path.exists("faiss_gem_clean/index.pkl"):
            print("   ✅ Index files present")
        else:
            issues.append("FAISS index files missing")
    else:
        issues.append("Clean FAISS index not found")
    
    # 2. Check chatbot service configuration
    print("\n2. CHECKING CHATBOT SERVICE...")
    try:
        from chat.chatbot_service import chatbot_service
        print("   ✅ Chatbot service imports successfully")
        
        # Check if gem_db is loaded
        if hasattr(chatbot_service, 'gem_db') and chatbot_service.gem_db:
            print("   ✅ GeM database loaded")
        else:
            issues.append("GeM database not loaded in chatbot service")
            
        # Check if scoped retriever exists
        if hasattr(chatbot_service, '_scoped') and chatbot_service._scoped:
            print("   ✅ Scoped retriever initialized")
        else:
            issues.append("Scoped retriever not initialized")
            
    except Exception as e:
        issues.append(f"Chatbot service error: {e}")
    
    # 3. Check environment variables
    print("\n3. CHECKING ENVIRONMENT...")
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "PASTE_YOUR_NEW_API_KEY_HERE":
        print("   ✅ Google API key configured")
    else:
        issues.append("Google API key not properly configured")
    
    # 4. Check Django settings
    print("\n4. CHECKING DJANGO CONFIGURATION...")
    try:
        from django.conf import settings
        print("   ✅ Django settings loaded")
    except Exception as e:
        issues.append(f"Django configuration error: {e}")
    
    # 5. Test actual retrieval
    print("\n5. TESTING RETRIEVAL FUNCTIONALITY...")
    try:
        docs = chatbot_service.scoped_gem_search("consignee", pdf_id="7908419", k=3)
        if docs and len(docs) > 0:
            print(f"   ✅ Retrieval working ({len(docs)} docs found)")
        else:
            issues.append("Retrieval returns no documents")
    except Exception as e:
        issues.append(f"Retrieval error: {e}")
    
    # 6. Test classification
    print("\n6. TESTING QUESTION CLASSIFICATION...")
    try:
        classification = chatbot_service.classify_question_type("Who is the consignee for document 7908419?")
        if classification == "gem":
            print("   ✅ Classification working correctly")
        else:
            warnings.append(f"Classification returned '{classification}' instead of 'gem'")
    except Exception as e:
        issues.append(f"Classification error: {e}")
    
    # 7. Check graph workflow
    print("\n7. CHECKING GRAPH WORKFLOW...")
    try:
        from chat.chatbot_graph import create_graph
        from chat.chatbot_service import memory_saver
        
        graph = create_graph(memory_saver)
        print("   ✅ Graph workflow created successfully")
    except Exception as e:
        issues.append(f"Graph workflow error: {e}")
    
    # 8. Test gem processor
    print("\n8. TESTING GEM PROCESSOR...")
    try:
        if hasattr(chatbot_service, 'gem_processor') and chatbot_service.gem_processor:
            print("   ✅ GeM processor initialized")
        else:
            issues.append("GeM processor not initialized")
    except Exception as e:
        issues.append(f"GeM processor error: {e}")
    
    # 9. Check for potential memory issues
    print("\n9. CHECKING MEMORY CONFIGURATION...")
    try:
        from chat.chatbot_service import DatabaseManager
        conn = DatabaseManager.get_connection()
        if conn:
            print("   ✅ Database connection working")
        else:
            warnings.append("Database connection issue")
    except Exception as e:
        warnings.append(f"Database connection warning: {e}")
    
    # 10. Test end-to-end workflow
    print("\n10. TESTING END-TO-END WORKFLOW...")
    try:
        # Simulate a complete question-answer cycle
        test_state = {
            "question": "Who is the consignee for document 7908419?",
            "chat_history": [],
            "documents": [],
            "answer": "",
            "generation_source": "",
            "question_type": "",
            "user_choice": "",
            "active_doc": ""
        }
        
        from chat.chatbot_graph import classify_question, retrieve_documents
        
        # Test classification
        result = classify_question(test_state)
        if result.get("question_type") == "gem":
            print("   ✅ End-to-end classification working")
        else:
            warnings.append("End-to-end classification issue")
            
        # Test retrieval
        test_state.update(result)
        result = retrieve_documents(test_state)
        if result.get("documents"):
            print("   ✅ End-to-end retrieval working")
        else:
            warnings.append("End-to-end retrieval issue")
            
    except Exception as e:
        issues.append(f"End-to-end workflow error: {e}")
    
    # Report results
    print("\n" + "="*60)
    print("AUDIT RESULTS")
    print("="*60)
    
    if not issues and not warnings:
        print("🟢 SYSTEM READY FOR PRODUCTION")
        print("✅ All systems operational")
        print("✅ No critical issues found")
        print("✅ Ready for questioning!")
        return True
    
    if issues:
        print("🔴 CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"   ❌ {issue}")
    
    if warnings:
        print("\n🟡 WARNINGS:")
        for warning in warnings:
            print(f"   ⚠️  {warning}")
    
    if issues:
        print("\n❌ SYSTEM NOT READY - Fix critical issues first")
        return False
    else:
        print("\n🟡 SYSTEM MOSTLY READY - Warnings can be ignored")
        return True

if __name__ == '__main__':
    ready = audit_system()
    if ready:
        print("\n🚀 GREEN LIGHT: Start your server and begin questioning!")
    else:
        print("\n🛑 RED LIGHT: Fix issues before proceeding")