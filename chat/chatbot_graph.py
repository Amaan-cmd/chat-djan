# chat/chatbot_graph.py

from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langgraph.graph import StateGraph, END

from .chatbot_logic import chatbot_service
#GraphState is like a carrier or a data storage thing which passes the query through various nodes
class GraphState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    answer: str
    generation_source: str
#now there are multiple nodes:- ret_docs is there to get the doc from faiss index
def retrieve_documents(state: GraphState):
    # ... (code for this function)
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    chat_history = state["chat_history"]
    documents = chatbot_service.history_aware_retriever.invoke(
        {"input": question, "chat_history": chat_history}
    )
    return {"documents": documents}
#grader is there to qualify how accurate the answer actually is bhai
def grade_documents(state: GraphState):
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing if a document is relevant to a user's question.
        A document is relevant if it contains information that can substantively answer the user's question.
        It is NOT relevant if it simply shares a keyword but discusses a different topic.
        For example, a document about the 'Sega Saturn' video game console is NOT relevant to a question about the 'planet Saturn'.
        Give a binary JSON output with a single key 'is_relevant' and a value of 'yes' or 'no'.
        Document: {document_content}\nUser Question: {question}"""
    )
    grader_chain = prompt | chatbot_service.llm | JsonOutputParser()
    is_relevant = "no"
    if documents:
        doc_content = documents[0].page_content
        try:
            result = grader_chain.invoke({"question": question, "document_content": doc_content})
            if result.get("is_relevant") == "yes":
                is_relevant = "yes"
        except Exception as e:
            print(f"---ERROR IN GRADER: {e}---")
            is_relevant = "no"
    print(f"---GRADE: Documents are '{is_relevant}' for the question---")
    return {"documents": documents if is_relevant == "yes" else []}
#this node specifically generates a answer to the query using knowledge base
def generate_rag_answer(state: GraphState):
    # ... (code for this function)
    print("---NODE: GENERATE RAG ANSWER---")
    question = state["question"]
    chat_history = state["chat_history"]
    documents = state["documents"]
    answer = chatbot_service.strict_Youtube_chain.invoke({"input": question, "chat_history": chat_history, "context": documents})
    return {"answer": answer, "generation_source": "rag"}
#and here an answer from the base model is created
def generate_general_answer(state: GraphState):
    # ... (code for this function)
    print("---NODE: GENERATE GENERAL ANSWER---")
    question = state["question"]
    chat_history = state["chat_history"]
    general_response = chatbot_service.general_knowledge_chain.invoke({"input": question, "chat_history": chat_history})
    return {"answer": general_response.content, "generation_source": "general"}
#ooh decision maker so if docs are relevant answer from the docs are given otherwise answer from gen model are passed
def decide_generation_path(state: GraphState):
    print("---CONDITIONAL EDGE: DECIDE PATH---")
    if state["documents"]:
        print("---DECISION: Graded as relevant, routing to RAG.---")
        return "generate_rag"
    else:
        print("---DECISION: Graded as not relevant, routing to General.---")
        return "generate_general"
#take all the nodes and the conditions defined and wires them together into a complete, runnable workflow
def create_graph():
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
    return workflow.compile()
