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
        
        # Use hybrid extraction for specific document queries
        import re
        doc_match = re.search(r'\b(\d{7})\b', question)
        if doc_match:
            doc_number = doc_match.group(1)
            print(f"---USING: Hybrid extraction for document {doc_number}---")
            documents = chatbot_service.hybrid_gem_extraction(question, doc_number)
            
            if documents:
                print(f"---HYBRID EXTRACTION SUCCESS: {len(documents)} results---")
            else:
                print("---HYBRID FAILED: Falling back to smart search---")
                documents = chatbot_service.smart_gem_search(question)
        else:
            # Check for multi-document queries first
            multi_doc_indicators = ['all documents', 'each document', 'all pdf', 'each pdf', 'for all', 'systematic manner', 'compare', 'list all']
            is_multi_doc = any(indicator in question.lower() for indicator in multi_doc_indicators)
            
            if is_multi_doc:
                print("---MULTI-DOC QUERY DETECTED: Using smart search across all PDFs---")
                documents = chatbot_service.smart_gem_search(question, k=50)
            else:
                # Use regular history-aware retrieval for single queries
                documents = chatbot_service.gem_history_aware_retriever.invoke(
                    {"input": question, "chat_history": chat_history}
                )
                print(f"---REGULAR SEARCH RETURNED: {len(documents)} documents---")
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
    
    # Skip grading for multi-document queries to preserve all documents
    multi_doc_indicators = ['all documents', 'each document', 'all pdf', 'each pdf', 'for all', 'systematic manner']
    if any(indicator in question.lower() for indicator in multi_doc_indicators):
        print("---GRADE: Skipping grading for multi-document query - using all retrieved docs---")
        return {"documents": documents}
    
    # Different grading prompts for different types
    if question_type == "calamity":
        grading_prompt = (
            "You are a grader assessing if a document is relevant to a Terraria Calamity mod question. "
            "A document is relevant if it contains specific information about Calamity mod content "
            "(weapons, bosses, items, mechanics, etc.). "
            "Give a binary JSON output with 'is_relevant': 'yes' or 'no'."
        )
    elif question_type == "gem":
        grading_prompt = (
            "You are a grader assessing if a document is relevant to a GeM procurement question. "
            "A document is relevant if it contains information about government bidding, procurement processes, "
            "requirements, or procedures. "
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
            
            # Check if this is a structured extraction result
            if documents and documents[0].metadata.get('extraction_type') == 'structured':
                print("---DEBUG: Using structured extraction result---")
                # Return the pre-formatted response directly
                return {"answer": documents[0].page_content, "generation_source": "gem", "question_type": question_type}
            
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