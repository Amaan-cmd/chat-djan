# chat/chatbot_graph.py

from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langgraph.graph import StateGraph, END

from .chatbot_logic import chatbot_service
from .cache_service import ChatCacheService
from langchain.schema import Document

class GraphState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    answer: str
    generation_source: str
    is_correction: bool
def retrieve_documents(state: GraphState):
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    chat_history = state["chat_history"]
    
    # Check cache first
    cached_docs = ChatCacheService.get_cached_documents(question)
    if cached_docs:
        print(f"---CACHE HIT: {len(cached_docs)} documents---")
        # Reconstruct Document objects
        documents = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in cached_docs]
        return {"documents": documents}
    
    # Cache miss - retrieve from source
    documents = chatbot_service.history_aware_retriever.invoke(
        {"input": question, "chat_history": chat_history}
    )
    
    # Cache the results
    ChatCacheService.cache_documents(question, documents)
    print(f"---RETRIEVED: {len(documents)} documents (cached)---")
    return {"documents": documents}

def grade_documents(state: GraphState):
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing if a document is relevant to a Terraria Calamity mod question.
        A document is relevant if it contains specific information about Calamity mod content (weapons, bosses, items, mechanics, etc.).
        It is NOT relevant if it only mentions general Terraria content or unrelated topics.
        Give a binary JSON output with a single key 'is_relevant' and a value of 'yes' or 'no'.
        Document: {document_content}\nUser Question: {question}"""
    )
    grader_chain = prompt | chatbot_service.llm | JsonOutputParser()
    
    if not documents:
        return {"documents": []}
    
    relevant_docs = []
    relevant_count = 0
    max_docs = min(len(documents), 3)
    
    for i, doc in enumerate(documents[:max_docs]):
        try:
            content = doc.page_content[:800] if doc.page_content else ""
            result = grader_chain.invoke({"question": question, "document_content": content})
            if result.get("is_relevant") == "yes":
                relevant_docs.append(doc)
                relevant_count += 1
        except Exception as e:
            print(f"---ERROR IN GRADER for doc {i}: {str(e)[:100]}---")
            continue
    
    print(f"---GRADE: {relevant_count} out of {max_docs} documents are relevant---")
    return {"documents": relevant_docs if relevant_count > 0 else []}

def generate_rag_answer(state: GraphState):
    print("---NODE: GENERATE RAG ANSWER---")
    question = state["question"]
    chat_history = state["chat_history"]
    documents = state["documents"]
    answer = chatbot_service.strict_Youtube_chain.invoke({"input": question, "chat_history": chat_history, "context": documents})
    return {"answer": answer, "generation_source": "rag"}

def generate_general_answer(state: GraphState):
    print("---NODE: GENERATE GENERAL ANSWER---")
    question = state["question"]
    chat_history = state["chat_history"]
    general_response = chatbot_service.general_knowledge_chain.invoke({"input": question, "chat_history": chat_history})
    return {"answer": general_response.content, "generation_source": "general"}

def decide_generation_path(state: GraphState):
    print("---CONDITIONAL EDGE: DECIDE PATH---")
    
    if state.get("is_correction", False):
        if state.get("documents", []):
            print("---DECISION: Correction detected, routing to General instead of RAG.---")
            return "generate_general"
        else:
            print("---DECISION: Correction detected, routing to RAG instead of General.---")
            return "generate_rag"
    
    if state.get("documents", []):
        print("---DECISION: Graded as relevant, routing to RAG.---")
        return "generate_rag"
    else:
        print("---DECISION: Graded as not relevant, routing to General.---")
        return "generate_general"


def create_graph(checkpointer):
    """Builds and compiles the chatbot state graph with a checkpointer."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_rag", generate_rag_answer)
    workflow.add_node("generate_general", generate_general_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_generation_path,
        {"generate_rag": "generate_rag", "generate_general": "generate_general"},
    )
    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_general", END)

    return workflow.compile(checkpointer=checkpointer)
