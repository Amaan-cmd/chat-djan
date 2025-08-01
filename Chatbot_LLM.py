import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, List, Dict
from langchain_core.documents import Document

# --- Configuration ---
# NEW: Add all the PDF files you want to use into this list
PDF_FILE_PATHS = [
    r"C:\Users\dbleg\Downloads\BE AIDS Syllabus .pdf",
    r"C:\Users\dbleg\Downloads\SISCON_The_atomic_bomb.pdf",  # <-- Add the name of your second PDF here
    r"C:\Users\dbleg\Downloads\agriculture.pdf",
    r"C:\Users\dbleg\Downloads\motor_law-59.pdf"
]
VECTOR_STORE_PATH = "faiss_index_combined"

# Gemini Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCnOGFSGOaJCQDkUouvv_1Nh-clZ684vM4"


# --- Callback Handler (remains the same) ---
class Bcolors:
    OKBLUE, OKCYAN, OKGREEN, WARNING, ENDC = '\033[94m', '\033[96m', '\033[92m', '\033[93m', '\033[0m'


class MyCustomCallbackHandler(BaseCallbackHandler):
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any): print(
        f"\n{Bcolors.OKBLUE}[Retriever Start] Query: '{query}'{Bcolors.ENDC}")

    def on_retriever_end(self, documents: List[Document], **kwargs: Any): print(
        f"{Bcolors.OKCYAN}[Retriever End] Found {len(documents)} docs.{Bcolors.ENDC}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any): print(
        f"\n{Bcolors.OKGREEN}[LLM Start] Prompt: {Bcolors.WARNING}{prompts[0][:500]}...{Bcolors.ENDC}")

    def on_llm_end(self, response: Any, **kwargs: Any): print(
        f"{Bcolors.OKGREEN}[LLM End] Response received.{Bcolors.ENDC}")


def get_vector_store():
    """Handles creation or loading of a vector store from MULTIPLE PDFs."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Loading existing vector store from: {VECTOR_STORE_PATH}")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vector store from all PDFs in the list...")
        all_docs = []
        # NEW: Loop through all PDF paths and load them
        for pdf_path in PDF_FILE_PATHS:
            if os.path.exists(pdf_path):
                print(f"  - Loading {pdf_path}...")
                loader = PyPDFLoader(pdf_path)
                all_docs.extend(loader.load())
            else:
                print(f"  - Warning: File not found at {pdf_path}, skipping.")

        if not all_docs:
            print("Error: No valid PDF documents were loaded. Exiting.")
            sys.exit(1)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(all_docs)

        print("Creating and saving new combined vector store...")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(VECTOR_STORE_PATH)
        print("Vector store saved.")
        return db


def main():
    """Main function to run the PDF chatbot."""
    db = get_vector_store()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    retriever = db.as_retriever(search_kwargs={"k": 10})

    # Chains and chat loop remain the same
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history... formulate a standalone question..."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a factual assistant... Use the context...\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, youtube_chain)

    chat_history = []
    print("\nKnowledge base processed. You can now ask questions.")
    while True:
        print("\n---------------------------")
        question = input("Ask a question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        result = qa_chain.invoke(
            {"input": question, "chat_history": chat_history},
            config={"callbacks": [MyCustomCallbackHandler()]}
        )

        print("\nAI Answer:")
        print(result['answer'])

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result['answer']))


if __name__ == "__main__":
    main()