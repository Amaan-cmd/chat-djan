# chat/chatbot_logic.py

import os
import sqlite3
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv(override=True)

VECTOR_STORE_PATH = "faiss_index_combined"


class ChatbotService:
    def __init__(self):
        print("Initializing ChatbotService and loading components...")
        api_key = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=api_key
        )

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
            transport="rest"
        )

        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        # Fetch top 3 most relevant chunks for better quality
        self.retriever = db.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.7})

        self.history_aware_retriever = self._setup_history_aware_retriever()
        self.strict_Youtube_chain = self._setup_strict_qa_chain()
        self.detailed_chain = self._setup_detailed_qa_chain()
        self.general_knowledge_chain = self._setup_general_chain()
        print("ChatbotService initialized successfully.")

    def _setup_history_aware_retriever(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Reformulate the user question based on chat history to be a standalone question. Do not answer it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt)

    def _setup_strict_qa_chain(self):
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

    def _setup_detailed_qa_chain(self):
        system_prompt = (
            "You are a Terraria Calamity mod expert providing DETAILED and COMPREHENSIVE answers. "
            "CRITICAL RULES:\n"
            "1. Provide extensive, in-depth information from the context\n"
            "2. Include specific details like stats, mechanics, crafting recipes, strategies\n"
            "3. Cover multiple aspects of the topic (usage, obtaining, related items, tips)\n"
            "4. Use examples and specific scenarios when possible\n"
            "5. If context is limited, clearly state what information is available\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, prompt)

    def _setup_general_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question based on your own knowledge."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return prompt | self.llm


chatbot_service = ChatbotService()
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory_saver = SqliteSaver(conn=conn)