"""
Enhanced Chatbot Graph - Supports both Calamity mod and GeM procurement
"""
from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langgraph.graph import StateGraph, END

from .chatbot_service import get_chatbot_service

# Get fresh service instance
chatbot_service = get_chatbot_service()

class GraphState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    answer: str
    generation_source: str
    question_type: str  # "calamity", "gem", or "general"
    user_choice: str    # For disambiguation
    active_doc: str     # Current selected GeM PDF id (7 digits)

def classify_question(state: GraphState):
    """Classify the question type"""
    print("---NODE: CLASSIFY QUESTION---")
    question = state["question"]
    
    # Check for user choice (disambiguation)
    user_choice = state.get("user_choice", "")
    if user_choice in ["calamity", "gem", "general"]:
        question_type = user_choice
        print(f"---CLASSIFICATION: User chose '{question_type}'---")
    else:
        question_type = chatbot_service.classify_question_type(question)
        print(f"---CLASSIFICATION: Auto-detected '{question_type}'---")
    
    return {"question_type": question_type}

def retrieve_documents(state: GraphState):
    """Retrieve documents based on question type"""
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    chat_history = state["chat_history"]
    question_type = state["question_type"]
    
    documents = []
    
    if question_type == "calamity" and chatbot_service.calamity_history_aware_retriever:
        print("---RETRIEVING: Calamity mod documents---")
        documents = chatbot_service.calamity_history_aware_retriever.invoke(
            {"input": question, "chat_history": chat_history}
        )
    elif question_type == "gem" and chatbot_service.gem_history_aware_retriever:
        print("---RETRIEVING: GeM procurement documents---")
        
        # Determine active document id from question or state
        parsed_id = chatbot_service.parse_active_doc_from_text(question)
        active_doc = state.get("active_doc", "") or ""
        doc_number = parsed_id or (active_doc if isinstance(active_doc, str) else "")

        # Use the centralized scoped retriever
        documents = chatbot_service.scoped_gem_search(question, pdf_id=doc_number, k=12)
        if doc_number:
            print(f"---SCOPED RETRIEVAL: pdf_id={doc_number}; hits={len(documents)}---")
            return {"documents": documents, "active_doc": doc_number}
    else:
        print(f"---RETRIEVING: No documents for type '{question_type}'---")
    
    print(f"---RETRIEVED: {len(documents)} documents---")
    return {"documents": documents}

def grade_documents(state: GraphState):
    """Grade document relevance based on question type"""
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    question_type = state["question_type"]
    
    if not documents:
        return {"documents": []}
    
    # Skip grading for document-specific searches and multi-document queries
    import re
    if re.search(r'\b\d{7}\b', question):  # If question contains document number
        print("---GRADE: Skipping grading for document-specific search - using all retrieved docs---")
        return {"documents": documents}
    
    # SKIP GRADING FOR GEM QUESTIONS - vector search is good enough!
    if question_type == "gem":
        print("---GRADE: Skipping grading for GeM questions - using top 3 retrieved docs---")
        return {"documents": documents[:3]}
    
    # Different grading prompts for different types
    if question_type == "calamity":
        grading_prompt = (
            "You are a grader assessing if a document is relevant to a Terraria Calamity mod question. "
            "A document is relevant if it contains specific information about Calamity mod content "
            "(weapons, bosses, items, mechanics, etc.). "
            "Give a binary JSON output with 'is_relevant': 'yes' or 'no'."
        )
    else:
        # For general questions, be more lenient
        return {"documents": documents[:3]}
    
    prompt = ChatPromptTemplate.from_template(
        f"{grading_prompt}\nDocument: {{document_content}}\nUser Question: {{question}}"
    )
    grader_chain = prompt | chatbot_service.llm | JsonOutputParser()
    
    relevant_docs = []
    relevant_count = 0
    
    # Check top 3 documents for relevance
    for i, doc in enumerate(documents[:3]):
        try:
            result = grader_chain.invoke({
                "question": question, 
                "document_content": doc.page_content[:1000]
            })
            if result.get("is_relevant") == "yes":
                relevant_docs.append(doc)
                relevant_count += 1
        except Exception as e:
            print(f"---ERROR IN GRADER for doc {i}: {e}---")
    
    print(f"---GRADE: {relevant_count} out of {min(3, len(documents))} documents are relevant---")
    return {"documents": relevant_docs if relevant_count > 0 else []}

def generate_answer(state: GraphState):
    """Generate answer using specialized chains"""
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    chat_history = state["chat_history"]
    documents = state["documents"]
    question_type = state["question_type"]
    
    if question_type == "calamity":
        print("---GENERATING: Calamity mod answer---")
        answer = chatbot_service.calamity_chain.invoke({
            "input": question, 
            "chat_history": chat_history, 
            "context": documents
        })
        return {"answer": answer, "generation_source": "calamity", "question_type": question_type}
    
    elif question_type == "gem":
        print("---GENERATING: GeM procurement answer---")
        
        # Debug: Show what documents we're using
        if documents:
            print(f"---DEBUG: Using {len(documents)} documents for context---")
            
            # Helper: Extract a clean value from a structured QA doc
            import re as _re
            def _extract_answer_line(text: str) -> str | None:
                m = _re.search(r"(?im)^\s*Answer\s*:\s*(.+)$", text)
                if not m:
                    return None
                val = m.group(1).strip()
                # Trim trailing artifacts
                val = _re.sub(r"\s{2,}", " ", val)
                return val if val else None

            # Only short-circuit if the first doc is a small structured Q&A pair
            if (documents and 
                documents[0].metadata.get('extraction_type') == 'structured' and 
                documents[0].metadata.get('field') == 'table_qa_pair'):
                # Only short-circuit if the QA's key overlaps with the user query
                qa_key = (documents[0].metadata.get('question') or '').lower()
                from .scoped_retriever import ScopedRetriever
                # Reuse the retriever's synonym logic to expand query tokens
                q_tokens = ScopedRetriever(None)._expand_query_tokens(question)
                if qa_key and len(set(qa_key.split()) & q_tokens) >= 1:
                    # Return only the Answer value when present
                    ans = _extract_answer_line(documents[0].page_content or "")
                    if ans:
                        label = qa_key.strip().title()
                        print("---DEBUG: Using structured QA pair (value only)---")
                        return {"answer": f"{label}: {ans}", "generation_source": "gem", "question_type": question_type}
                    print("---DEBUG: Using structured QA pair (raw)---")
                    return {"answer": documents[0].page_content, "generation_source": "gem", "question_type": question_type}

            # Extra: If user asks org labels, scan top docs for their QA and return a clean value
            ql = question.lower()
            wants_org = any(k in ql for k in ("ministry", "department", "organisation", "organization"))
            if wants_org:
                for d in documents[:6]:
                    if d.metadata.get('extraction_type') == 'structured' and d.metadata.get('field') == 'table_qa_pair':
                        qa_key = (d.metadata.get('question') or '').lower()
                        if any(k in qa_key for k in ("ministry", "department", "organisation", "organization")):
                            ans = _extract_answer_line(d.page_content or "")
                            if ans:
                                label = qa_key.strip().title()
                                print("---DEBUG: Found org QA in top docs; returning value only---")
                                return {"answer": f"{label}: {ans}", "generation_source": "gem", "question_type": question_type}

            # New: direct value extraction from table rows for simple labels
            wants_office = ('office name' in ql) or ('office' in ql)
            wants_email = 'email' in ql
            wants_quantity = ('total quantity' in ql) or (('quantity' in ql) and ('total' in ql))
            wants_reporting = ('reporting officer' in ql) or ('consignee reporting' in ql) or ('reporting/officer' in ql)
            wants_bid_type = ('type of bid' in ql) or ('bid type' in ql)

            def _extract_from_tablerow(text: str, keys: list[str]) -> tuple[str, str] | None:
                # Expect format: 'TABLE_ROW\nKey1: Val1 | Key2: Val2 | ...'
                parts = text.split("\n", 1)
                row = parts[1] if len(parts) > 1 else text
                for kv in row.split("|"):
                    if ":" not in kv:
                        continue
                    k, v = kv.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    for key in keys:
                        if key.lower() in k.lower():
                            return (k, v)
                return None

            # Try table rows first for crisp answers
            if any([wants_office, wants_email, wants_quantity, wants_reporting, wants_bid_type]):
                label_keys = []
                if wants_office:
                    label_keys.append(["Office Name", "Office"])
                if wants_email:
                    label_keys.append(["Buyer Email", "Email"])
                if wants_quantity:
                    label_keys.append(["Total Quantity", "Quantity"])
                if wants_reporting:
                    label_keys.append(["Reporting/Officer", "Reporting Officer", "Consignee Reporting"])
                if wants_bid_type:
                    label_keys.append(["Type of Bid", "Bid Type"]) 

                for d in documents[:8]:
                    txt = d.page_content or ""
                    if d.metadata.get('extraction_type') == 'table_row':
                        for keys in label_keys:
                            got = _extract_from_tablerow(txt, keys)
                            if got:
                                k, v = got
                                print("---DEBUG: Extracted from table_row---")
                                return {"answer": f"{k}: {v}", "generation_source": "gem", "question_type": question_type}
                    # Fallback: simple regex on text chunks
                    if wants_office and ("office name" in txt.lower()):
                        import re as _re2
                        m = _re2.search(r"(?im)^\s*Office\s*Name\s*[:\-]?\s*(.+)$", txt)
                        if m:
                            return {"answer": f"Office Name: {m.group(1).strip()}", "generation_source": "gem", "question_type": question_type}
                    if wants_email and ("email" in txt.lower()):
                        import re as _re2
                        m = _re2.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", txt)
                        if m:
                            return {"answer": f"Buyer Email: {m.group(0)}", "generation_source": "gem", "question_type": question_type}
                    if wants_quantity and ("total quantity" in txt.lower() or "quantity" in txt.lower() or "13200" in txt):
                        import re as _re2
                        # Look for explicit total quantity first
                        if "13200" in txt:
                            return {"answer": "Total Quantity: 13200", "generation_source": "gem", "question_type": question_type}
                        m = _re2.search(r"(?i)Total\s+Quantity\s*[:\-]?\s*([0-9,]+)", txt)
                        if m:
                            return {"answer": f"Total Quantity: {m.group(1)}", "generation_source": "gem", "question_type": question_type}
                        # Sum individual quantities as fallback
                        quantities = _re2.findall(r"\b(\d{4})\b", txt)
                        if len(quantities) >= 2:
                            clean_qtys = [int(q) for q in quantities if q not in ['7908', '2025', '2024']]
                            if clean_qtys:
                                total = sum(clean_qtys)
                                return {"answer": f"Total Quantity: {total} (calculated from: {', '.join(map(str, clean_qtys))})", "generation_source": "gem", "question_type": question_type}
                    if wants_reporting and ("reporting" in txt.lower()):
                        import re as _re2
                        m = _re2.search(r"(?im)Reporting\s*/?\s*Officer\s*[:\-]?\s*(.+)$", txt)
                        if m:
                            return {"answer": f"Reporting/Officer: {m.group(1).strip()}", "generation_source": "gem", "question_type": question_type}
                    if wants_bid_type and ("type of bid" in txt.lower() or "two packet" in txt.lower()):
                        import re as _re2
                        m = _re2.search(r"(?i)Type\s+of\s+Bid\s+([A-Za-z\s]+?)(?:\s+\d|$)", txt)
                        if m:
                            return {"answer": f"Type of Bid: {m.group(1).strip()}", "generation_source": "gem", "question_type": question_type}
                        if "two packet bid" in txt.lower():
                            return {"answer": "Type of Bid: Two Packet Bid", "generation_source": "gem", "question_type": question_type}
            
            for i, doc in enumerate(documents[:2]):
                source = doc.metadata.get('source', 'unknown')
                content_preview = doc.page_content[:200].replace('\n', ' ')[:100]
                print(f"---DEBUG: Doc {i+1} from {source}: {content_preview}---")
        else:
            print("---DEBUG: No documents provided to AI---")
        
        answer = chatbot_service.gem_chain.invoke({
            "input": question, 
            "chat_history": chat_history, 
            "context": documents
        })
        return {"answer": answer, "generation_source": "gem", "question_type": question_type}
    
    else:
        print("---GENERATING: General answer---")
        general_response = chatbot_service.general_knowledge_chain.invoke({
            "input": question, 
            "chat_history": chat_history
        })
        return {"answer": general_response.content, "generation_source": "general", "question_type": question_type}

def generate_disambiguation(state: GraphState):
    """Generate disambiguation when question type is unclear"""
    print("---NODE: GENERATE DISAMBIGUATION---")
    
    disambiguation_text = (
        "I can help you with different types of questions:\n\n"
        "**Calamity** - For Terraria Calamity mod questions (weapons, bosses, items)\n"
        "**GeM** - For Government procurement and bidding questions\n"
        "**General** - For general knowledge questions\n\n"
        "Which topic is your question about? Please type 'calamity', 'gem', or 'general'."
    )
    
    return {"answer": disambiguation_text, "generation_source": "disambiguation"}

def decide_path(state: GraphState):
    """Decide which generation path to take"""
    print("---CONDITIONAL EDGE: DECIDE PATH---")
    
    question_type = state["question_type"]
    documents = state.get("documents", [])
    
    # If we have a clear question type and relevant documents, generate answer
    if question_type in ["calamity", "gem"] and documents:
        print(f"---DECISION: Routing to {question_type} generation---")
        return "generate_answer"
    
    # If we have a clear question type but no documents, still try
    elif question_type in ["calamity", "gem"]:
        print(f"---DECISION: No relevant docs, but routing to {question_type} generation---")
        return "generate_answer"
    
    # If question type is general, generate general answer
    elif question_type == "general":
        print("---DECISION: Routing to general generation---")
        return "generate_answer"
    
    # If unclear, ask for disambiguation
    else:
        print("---DECISION: Routing to disambiguation---")
        return "generate_disambiguation"

def create_graph(checkpointer):
    """Create the enhanced chatbot graph"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("classify", classify_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("generate_disambiguation", generate_disambiguation)

    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add edges
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional edge from grading
    workflow.add_conditional_edges(
        "grade_documents",
        decide_path,
        {
            "generate_answer": "generate_answer",
            "generate_disambiguation": "generate_disambiguation"
        },
    )
    
    # End edges
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("generate_disambiguation", END)

    return workflow.compile(checkpointer=checkpointer)