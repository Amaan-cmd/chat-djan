"""
Clean Live GeM Integration Views - Completely Separate from Local System
"""
import json
import os
import shutil
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.shortcuts import render
from .chatbot_service import get_chatbot_service
from .gem_downloader import GemDownloader
from .live_multi_vector_processor import LiveMultiVectorProcessor


def live_gem_page(request):
    """Render the live GeM integration page"""
    return render(request, 'chat/live_gem.html')


@csrf_exempt
@require_POST
def live_gem_extract(request):
    """Extract single bid - completely clean"""
    try:
        data = json.loads(request.body)
        bid_id = str(data.get("bid_id", "")).strip()
        
        if not bid_id or len(bid_id) != 7 or not bid_id.isdigit():
            return JsonResponse({"success": False, "error": "Invalid bid ID. Must be 7 digits."})
        
        # Process PDF with complete index clearing
        success, error_msg, chunk_count = process_single_bid_clean(bid_id)
        
        if success:
            return JsonResponse({
                "success": True, 
                "chunks": chunk_count,
                "message": f"Successfully processed bid {bid_id}"
            })
        else:
            return JsonResponse({"success": False, "error": error_msg})
            
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Server error: {str(e)}"})


@csrf_exempt
@require_POST
def live_gem_chat(request):
    """Chat with ONLY live extracted data"""
    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()
        
        if not question:
            return JsonResponse({"error": "Question is required"})
        
        # Get answer using ONLY live index
        answer, chunks_used = get_live_answer(question)
        
        return JsonResponse({
            "answer": answer,
            "chunks_used": chunks_used,
            "source": "live_gem_only"
        })
        
    except Exception as e:
        return JsonResponse({"error": f"Chat error: {str(e)}"})


def process_single_bid_clean(bid_id: str) -> tuple[bool, str, int]:
    """Process single bid with complete index clearing"""
    try:
        print(f"\n=== PROCESSING BID {bid_id} ===")
        
        # Initialize downloader
        downloader = GemDownloader()
        
        # Try to download PDF first
        print(f"Step 1: Attempting to download PDF for bid {bid_id}...")
        pdf_path = downloader.download_pdf(bid_id)
        
        if not pdf_path:
            print(f"[ERROR] PDF download failed for bid {bid_id}")
            return False, f"Failed to download PDF for bid {bid_id}. The bid may not exist, may be expired, or may require authentication.", 0
        
        print(f"[SUCCESS] PDF downloaded successfully: {pdf_path}")
        
        # Try to extract text
        print(f"Step 2: Extracting text from PDF...")
        content = downloader.extract_text_from_pdf(pdf_path)
        
        if not content or len(content.strip()) < 100:
            print(f"[ERROR] Text extraction failed or content too short: {len(content) if content else 0} chars")
            return False, f"Failed to extract meaningful content from PDF. Content length: {len(content) if content else 0} characters.", 0
        
        print(f"[SUCCESS] Text extracted successfully: {len(content)} characters")
        
        # Get chatbot service for multi-vector processing
        print(f"Step 3: Setting up multi-vector processing...")
        chatbot_service = get_chatbot_service()
        
        # Get singleton live multi-vector processor
        live_processor = LiveMultiVectorProcessor(
            embeddings=chatbot_service.embeddings,
            llm=chatbot_service.llm
        )
        
        # Clear previous live index
        live_processor.clear_live_index()
        print(f"[SUCCESS] Cleared previous PDF from live index")
        
        # Process with multi-vector approach
        print(f"Step 4: Processing with multi-vector retrieval...")
        chunk_count = live_processor.process_live_pdf(content, bid_id)
        
        if chunk_count == 0:
            print(f"[ERROR] Multi-vector processing failed")
            return False, f"Failed to process PDF with multi-vector approach.", 0
        
        print(f"[SUCCESS] Multi-vector processing complete: {chunk_count} parent chunks")
        
        print(f"[SUCCESS] Live multi-vector index ready with {chunk_count} parent chunks from PDF {bid_id}")
        print(f"[INFO] Previous PDFs cleared - live index is now single-PDF only")
        print(f"=== BID {bid_id} PROCESSING COMPLETE ===")
        
        return True, "", chunk_count
        
    except Exception as e:
        print(f"[ERROR] Error processing bid {bid_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Processing error: {str(e)}", 0


def get_live_answer(question: str) -> tuple[str, int]:
    """Get answer using enhanced multi-vector retrieval"""
    try:
        print(f"\n❓ QUESTION: {question}")
        
        chatbot_service = get_chatbot_service()
        
        # Get singleton live multi-vector processor
        live_processor = LiveMultiVectorProcessor(
            embeddings=chatbot_service.embeddings,
            llm=chatbot_service.llm
        )
        
        # Use multi-vector search for better accuracy
        docs = live_processor.search_live_content(question, k=8)
        
        if not docs:
            print(f"⚠️ No documents found for question: {question}")
            return "No relevant information found in the extracted document.", 0
        
        # Enhanced keyword matching for better retrieval
        question_lower = question.lower()
        filtered_docs = docs  # Initialize with all docs
        
        print(f"📂 Found {len(docs)} documents, showing top scored result:")
        if filtered_docs:
            preview = filtered_docs[0].page_content[:150].replace('\n', ' ')
            print(f"   Preview: {preview}...")
        
        # Define field-specific keywords
        field_keywords = {
            'category': ['item category', 'category', 'cctv', 'surveillance', 'system'],
            'email': ['email', 'buyer email', '@', '.gov.in', '.com'],
            'ministry': ['ministry', 'organisation', 'department'],
            'quantity': ['quantity', 'qty', 'units'],
            'date': ['date', 'time', 'opening', 'closing'],
            'officer': ['officer', 'consignee', 'contact']
        }
        
        # Score documents based on keyword relevance
        scored_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            score = 1  # Base score
            
            # Boost score for keyword matches
            for field, keywords in field_keywords.items():
                if any(kw in question_lower for kw in keywords):
                    for kw in keywords:
                        if kw in content_lower:
                            score += 3
            
            # Extra boost for exact phrase matches
            if 'item category' in question_lower and 'item category' in content_lower:
                score += 5
            if 'buyer email' in question_lower and ('email' in content_lower or '@' in content_lower):
                score += 5
                
            scored_docs.append((score, doc))
        
        # Sort by score and take top results
        if scored_docs:
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            filtered_docs = [doc for score, doc in scored_docs[:6]]
            
            # Show if we found the specific field
            if 'item category' in question_lower:
                for doc in filtered_docs[:3]:
                    if 'item category' in doc.page_content.lower():
                        print(f"   ✅ Found 'item category' in chunk")
                        break
        
        # Step 3: Generate answer using GeM chain
        answer = chatbot_service.gem_chain.invoke({
            "input": question,
            "context": filtered_docs,
            "chat_history": []
        })
        
        print(f"\n💬 ANSWER: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"📄 Used {len(filtered_docs)} chunks")
        
        return answer, len(filtered_docs)
        
    except Exception as e:
        return f"Error: {str(e)}", 0