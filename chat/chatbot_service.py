"""
Main Chatbot Service - Refactored and Clean
"""
import os
import sqlite3
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from .question_classifier import QuestionClassifier
from .gem_processor import GemProcessor

load_dotenv(override=True)

CALAMITY_VECTOR_STORE_PATH = "faiss_index_combined"
GEM_VECTOR_STORE_PATH = "faiss_gem_index"

class ChatbotService:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatbotService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print("Initializing ChatbotService...")
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "PASTE_YOUR_NEW_API_KEY_HERE":
            raise ValueError("Please set a valid GOOGLE_API_KEY in your .env file")
            
        self._initialized = True

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=api_key
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
            transport="rest"
        )

        # Load vector stores
        self._load_vector_stores()
        
        # Initialize processors
        self.classifier = QuestionClassifier(self.embeddings, self.llm)
        self.gem_processor = GemProcessor(self.gem_db, self.llm)
        
        # Setup chains
        self.calamity_history_aware_retriever = self._setup_history_aware_retriever(self.calamity_retriever)
        self.gem_history_aware_retriever = self._setup_history_aware_retriever(self.gem_retriever)
        
        self.calamity_chain = self._setup_calamity_chain()
        self.gem_chain = self.gem_processor.setup_gem_chain()
        self.general_knowledge_chain = self._setup_general_chain()
        
        print("All chains initialized successfully")
        print("ChatbotService initialized successfully.")

    def _load_vector_stores(self):
        """Load both Calamity and GeM vector stores"""
        try:
            calamity_db = FAISS.load_local(
                CALAMITY_VECTOR_STORE_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            self.calamity_retriever = calamity_db.as_retriever(search_kwargs={"k": 5})
            print("SUCCESS: Calamity mod vector store loaded")
        except Exception as e:
            print(f"WARNING: Could not load Calamity vector store: {e}")
            self.calamity_retriever = None

        try:
            self.gem_db = FAISS.load_local(
                GEM_VECTOR_STORE_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            self.gem_retriever = self.gem_db.as_retriever(search_kwargs={"k": 5})
            print("SUCCESS: GeM procurement vector store loaded")
        except Exception as e:
            print(f"WARNING: Could not load GeM vector store: {e}")
            self.gem_retriever = None
            self.gem_db = None

    def _setup_history_aware_retriever(self, retriever):
        """Setup history-aware retriever for any vector store"""
        if not retriever:
            return None
            
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Reformulate the user question based on chat history to be a standalone question. Do not answer it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

    def _setup_calamity_chain(self):
        """Setup Calamity mod QA chain"""
        system_prompt = (
            "You are a Terraria Calamity mod expert assistant. "
            "CRITICAL RULES:\n"
            "1. Answer ONLY using information from the provided context\n"
            "2. If the context doesn't contain enough information, say 'I don't have enough information about that in my knowledge base'\n"
            "3. Focus specifically on Calamity mod content (weapons, bosses, items, mechanics)\n"
            "4. Be precise and factual - no speculation or general Terraria advice\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, prompt)

    def _setup_general_chain(self):
        """Setup general knowledge chain"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Be conversational but not overly casual. Answer questions clearly and add a touch of personality when appropriate."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return prompt | self.llm

    def classify_question_type(self, question: str) -> str:
        """Classify question type using the classifier"""
        return self.classifier.classify_question_type(question)
    
    def smart_gem_search(self, question: str, k: int = 8):
        """Smart GeM search using the processor"""
        return self.gem_processor.smart_gem_search(question, k)
    
    def hybrid_gem_extraction(self, question: str, doc_number: str):
        """Hybrid extraction using the processor"""
        return self.gem_processor.hybrid_gem_extraction(question, doc_number)

# Force fresh instance creation for Django
def get_chatbot_service():
    """Get fresh chatbot service instance"""
    return ChatbotService()

# Create service instance
chatbot_service = get_chatbot_service()
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory_saver = SqliteSaver(conn=conn)