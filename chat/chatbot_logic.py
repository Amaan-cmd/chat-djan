import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

VECTOR_STORE_PATH = "faiss_index_combined"
#service class is a container for all the tools that will be used
class ChatbotService:

    def __init__(self):
        print("Initializing ChatbotService and loading components...")

        # Initialize the core Language Model
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                          temperature=0.1,
                                          google_api_key=os.getenv("GOOGLE_API_KEY"))

        # Initialize the document retriever
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        self.retriever = db.as_retriever(search_kwargs={"k": 1})

        # Initialize the history-aware retriever (for rephrasing questions)
        self.history_aware_retriever = self._setup_history_aware_retriever()

        # Initialize the two main answer-generation chains
        self.strict_Youtube_chain = self._setup_strict_qa_chain()
        self.general_knowledge_chain = self._setup_general_chain()

        print("ChatbotService initialized successfully.")

    def _setup_history_aware_retriever(self):
        """Creates the chain that reformulates a question based on chat history."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Reformulate the user question based on chat history to be a standalone question. Do not answer it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt)

    def _setup_strict_qa_chain(self):
        """Creates the chain that answers ONLY from the provided context."""
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Your single most important instruction is to answer the user's question ONLY from the provided context.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, prompt)

    def _setup_general_chain(self):
        """Creates the chain that answers based on general knowledge."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question based on your own knowledge."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return prompt | self.llm

chatbot_service = ChatbotService()