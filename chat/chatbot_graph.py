# chat/chatbot_graph.py

from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langgraph.graph import StateGraph, END

from .chatbot_logic import chatbot_service
class GraphState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    answer: str
    generation_source: str
    is_ambiguous: bool
    user_choice: str
def retrieve_documents(state: GraphState):
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    chat_history = state["chat_history"]
    
    # Multi-strategy search
    all_docs = []
    
    # Direct search
    direct_docs = chatbot_service.retriever.invoke(question)
    all_docs.extend(direct_docs)
    
    # Term-based search
    key_terms = [word.lower() for word in question.split() if len(word) > 3]
    for term in key_terms:
        term_docs = chatbot_service.retriever.invoke(term)
        all_docs.extend(term_docs)
    
    # Remove duplicates
    seen_ids = set()
    unique_docs = []
    for doc in all_docs:
        doc_id = getattr(doc, 'id', str(hash(doc.page_content[:100])))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
    
    final_docs = unique_docs[:5]
    print(f"---RETRIEVED: {len(final_docs)} documents---")
    
    return {"documents": final_docs}

def grade_documents(state: GraphState):
    print("---NODE: GRADE DOCUMENTS---")
    documents = state["documents"]
    return {"documents": documents[:3]}  # Take top 3

def detect_ambiguity(state: GraphState):
    print("---NODE: DETECT AMBIGUITY---")
    question = state["question"].lower()
    documents = state.get("documents", [])
    
    print(f"---QUESTION: '{question}'---")
    print(f"---DOCUMENTS FOUND: {len(documents)}---")
    print(f"---USER CHOICE: {state.get('user_choice', 'None')}---")
    
    # Skip if user already made a choice
    if state.get("user_choice"):
        print("---USER ALREADY CHOSE, SKIPPING AMBIGUITY---")
        return {"is_ambiguous": False}
    
    # Ambiguous terms that could mean Calamity or general concepts
    ambiguous_terms = ["abyss", "armor", "weapons", "boss", "biome"]
    
    # Check if question contains ambiguous terms AND we have documents
    has_ambiguous_term = any(term in question for term in ambiguous_terms)
    has_calamity_docs = len(documents) > 0
    
    print(f"---HAS AMBIGUOUS TERM: {has_ambiguous_term}---")
    print(f"---HAS CALAMITY DOCS: {has_calamity_docs}---")
    
    if has_ambiguous_term and has_calamity_docs:
        found_term = next((term for term in ambiguous_terms if term in question), "this topic")
        
        disambiguation_prompt = f"I found information about '{found_term}'. Which would you prefer:\n\n🎮 Calamity mod {found_term}\n📚 General {found_term} information\n\nPlease click your choice:"
        
        print(f"---AMBIGUITY DETECTED FOR: {found_term}---")
        return {
            "is_ambiguous": True,
            "answer": disambiguation_prompt,
            "generation_source": "disambiguation"
        }
    else:
        print("---NO AMBIGUITY DETECTED---")
        return {"is_ambiguous": False}


def generate_rag_answer(state: GraphState):
    print("---NODE: GENERATE RAG ANSWER---")
    question = state["question"]
    chat_history = state["chat_history"]
    documents = state["documents"]
    user_choice = state.get("user_choice", "")
    
    # If user chose Calamity after disambiguation, enhance the question context
    if user_choice == "calamity":
        enhanced_question = f"Tell me about the Calamity mod {question.lower().replace('what is', '').replace('tell me about', '').strip()}"
        print(f"---ENHANCED QUESTION: {enhanced_question}---")
        answer = chatbot_service.strict_Youtube_chain.invoke({"input": enhanced_question, "chat_history": chat_history, "context": documents})
    else:
        answer = chatbot_service.strict_Youtube_chain.invoke({"input": question, "chat_history": chat_history, "context": documents})
    
    return {"answer": answer, "generation_source": "rag"}

def generate_general_answer(state: GraphState):
    print("---NODE: GENERATE GENERAL ANSWER---")
    question = state["question"]
    chat_history = state["chat_history"]
    general_response = chatbot_service.general_knowledge_chain.invoke({"input": question, "chat_history": chat_history})
    return {"answer": general_response.content, "generation_source": "general"}
def decide_after_ambiguity(state: GraphState):
    print("---CONDITIONAL EDGE: AFTER AMBIGUITY---")
    
    # If disambiguation needed, stop here
    if state.get("is_ambiguous", False):
        print("---DECISION: Ambiguous, showing choices to user---")
        return "END"
    
    # Route based on user choice or documents
    user_choice = state.get("user_choice", "")
    if user_choice == "calamity":
        print("---DECISION: User chose Calamity, routing to RAG with context---")
        return "generate_rag"
    elif user_choice == "general":
        print("---DECISION: User chose General, routing to General---")
        return "generate_general"
    
    # Normal routing based on documents
    if state.get("documents", []):
        print("---DECISION: Documents found, routing to RAG---")
        return "generate_rag"
    else:
        print("---DECISION: No documents, routing to General---")
        return "generate_general"


def create_graph(checkpointer):
    """Builds and compiles the chatbot state graph with a checkpointer."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("detect_ambiguity", detect_ambiguity)
    workflow.add_node("generate_rag", generate_rag_answer)
    workflow.add_node("generate_general", generate_general_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "detect_ambiguity")
    workflow.add_conditional_edges(
        "detect_ambiguity",
        decide_after_ambiguity,
        {"END": END, "generate_rag": "generate_rag", "generate_general": "generate_general"},
    )
    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_general", END)

    # Compile the graph WITH the checkpointer
    return workflow.compile(checkpointer=checkpointer)
