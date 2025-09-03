#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Document Check - Test ALL fields for document 7908419
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

from chat.chatbot_graph import create_graph
from chat.chatbot_service import memory_saver

def complete_document_test():
    """Test ALL possible fields for document 7908419"""
    
    print("=== COMPLETE DOCUMENT 7908419 CHECK ===\n")
    
    graph = create_graph(memory_saver)
    
    # ALL possible questions for document 7908419
    all_questions = [
        "What is the Bid End Date for document 7908419?",
        "What is the Bid Opening Date for document 7908419?",
        "What is the Bid Offer Validity for document 7908419?",
        "What is the Ministry Name for document 7908419?",
        "What is the Department Name for document 7908419?",
        "What is the Organisation Name for document 7908419?",
        "What is the Office Name for document 7908419?",
        "What is the Buyer Email for document 7908419?",
        "What is the Total Quantity for document 7908419?",
        "What is the Item Category for document 7908419?",
        "What are the Searched Strings for document 7908419?",
        "What Documents are required from seller for document 7908419?",
        "What is the Type of Bid for document 7908419?",
        "What is the Primary product category for document 7908419?",
        "What are the Payment Timelines for document 7908419?",
        "What is the Evaluation Method for document 7908419?",
        "What is the Arbitration Clause for document 7908419?",
        "What is the Mediation Clause for document 7908419?",
        "What is the EMD Detail for document 7908419?",
        "What is the ePBG Detail for document 7908419?",
        "What is the MII Purchase Preference for document 7908419?",
        "What is the MSE Purchase Preference for document 7908419?",
        "Who is the Consignee Reporting Officer for document 7908419?",
        "What is the address for document 7908419?",
        "What is the delivery time for document 7908419?",
        "Is Bid to RA enabled for document 7908419?",
        "What is the RA Qualification Rule for document 7908419?",
        "What is the Technical Clarification Time for document 7908419?",
    ]
    
    results = {}
    success_count = 0
    
    for i, question in enumerate(all_questions, 1):
        print(f"[{i:2d}/{len(all_questions)}] Testing: {question}")
        
        initial_state = {
            "question": question,
            "chat_history": [],
            "documents": [],
            "answer": "",
            "generation_source": "",
            "question_type": "",
            "user_choice": "",
            "active_doc": "7908419"
        }
        
        try:
            config = {"configurable": {"thread_id": f"test_{i}"}}
            result = graph.invoke(initial_state, config)
            
            answer = result.get('answer', 'No answer')
            
            # Evaluate answer quality
            if len(answer) > 20 and "not available" not in answer.lower() and "error" not in answer.lower():
                status = "✅ GOOD"
                success_count += 1
            elif len(answer) > 10:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ POOR"
            
            results[question] = {
                'answer': answer,
                'status': status
            }
            
            print(f"     {status}: {answer[:80]}...")
            
        except Exception as e:
            results[question] = {
                'answer': f"ERROR: {e}",
                'status': "❌ ERROR"
            }
            print(f"     ❌ ERROR: {e}")
        
        print()
    
    # Summary Report
    print("="*100)
    print("COMPLETE DOCUMENT 7908419 SUMMARY REPORT")
    print("="*100)
    
    total_questions = len(all_questions)
    success_rate = (success_count / total_questions) * 100
    
    print(f"Total Questions: {total_questions}")
    print(f"Good Answers: {success_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    # Detailed results
    for question, result in results.items():
        field_name = question.split(" for document")[0].replace("What is the ", "").replace("What are the ", "").replace("Who is the ", "").replace("Is ", "")
        print(f"{result['status']} {field_name}")
        if result['status'] == "❌ POOR" or result['status'] == "❌ ERROR":
            print(f"    Answer: {result['answer'][:100]}...")
        print()
    
    # Recommendations
    print("="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    poor_results = [q for q, r in results.items() if "❌" in r['status']]
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Document 7908419 has comprehensive coverage!")
    elif success_rate >= 75:
        print("👍 GOOD: Most fields covered, minor improvements needed")
    else:
        print("⚠️  NEEDS WORK: Several fields need attention")
    
    if poor_results:
        print(f"\nFields needing improvement ({len(poor_results)}):")
        for question in poor_results[:5]:  # Show first 5
            field = question.split(" for document")[0].replace("What is the ", "").replace("What are the ", "")
            print(f"  - {field}")
    
    return results, success_rate

if __name__ == '__main__':
    results, rate = complete_document_test()
    print(f"\n🏁 FINAL SCORE: {rate:.1f}% coverage for document 7908419")