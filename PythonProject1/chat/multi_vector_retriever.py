"""
Multi-Vector Retrieval System - Enhanced accuracy with parent-child document strategy
"""
import os
import uuid
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class EnhancedMultiVectorRetriever:
    """
    Advanced retrieval system using parent-child document strategy
    - Parent docs: Full context (large chunks)
    - Child docs: Precise retrieval (small chunks) 
    - Summaries: Semantic understanding
    """
    
    def __init__(self, embeddings, llm, persist_directory="faiss_multi_vector"):
        self.embeddings = embeddings
        self.llm = llm
        self.persist_directory = persist_directory
        
        # Text splitters for different granularities
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        # Storage components
        self.byte_store = InMemoryByteStore()
        self.id_key = "doc_id"
        
        # Initialize vector store and retriever
        self._setup_retriever()
        
        # Summary generation chain
        self.summary_chain = self._create_summary_chain()
    
    def _setup_retriever(self):
        """Initialize the multi-vector retriever"""
        try:
            # Try to load existing vector store
            if os.path.exists(self.persist_directory):
                self.vectorstore = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded existing multi-vector store from {self.persist_directory}")
            else:
                # Create new vector store
                # Initialize with dummy document to avoid empty store issues
                dummy_doc = Document(page_content="initialization", metadata={"type": "dummy"})
                self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
                print(f"✅ Created new multi-vector store")
        except Exception as e:
            print(f"⚠️ Error setting up vector store: {e}")
            # Fallback: create fresh store
            dummy_doc = Document(page_content="initialization", metadata={"type": "dummy"})
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
        
        # Create multi-vector retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.byte_store,
            id_key=self.id_key,
            search_kwargs={"k": 8}
        )
    
    def _create_summary_chain(self):
        """Create LLM chain for document summarization"""
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following document chunk, focusing on key information "
            "that would be useful for question answering:\n\n{doc}\n\nSummary:"
        )
        return (
            {"doc": lambda x: x.page_content}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def add_documents(self, documents: List[Document], generate_summaries: bool = True):
        """
        Add documents using multi-vector strategy
        
        Args:
            documents: List of documents to add
            generate_summaries: Whether to generate summaries for better retrieval
        """
        print(f"📚 Adding {len(documents)} documents to multi-vector store...")
        
        # Step 1: Create parent documents (larger chunks for context)
        parent_docs = self.parent_splitter.split_documents(documents)
        
        # Step 2: Generate unique IDs for parent documents
        doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
        
        # Step 3: Create child documents (smaller chunks for precise retrieval)
        child_docs = []
        for i, parent_doc in enumerate(parent_docs):
            # Split parent into children
            children = self.child_splitter.split_documents([parent_doc])
            for child in children:
                # Link child to parent via ID
                child.metadata[self.id_key] = doc_ids[i]
                child_docs.append(child)
        
        print(f"📄 Created {len(parent_docs)} parent docs and {len(child_docs)} child docs")
        
        # Step 4: Generate summaries if requested
        if generate_summaries and len(parent_docs) <= 20:  # Limit for API costs
            try:
                print("🤖 Generating summaries...")
                summaries = self.summary_chain.batch(parent_docs[:10])  # Limit batch size
                
                # Create summary documents
                summary_docs = []
                for i, summary in enumerate(summaries):
                    if i < len(doc_ids):  # Safety check
                        summary_doc = Document(
                            page_content=summary,
                            metadata={self.id_key: doc_ids[i], "type": "summary"}
                        )
                        summary_docs.append(summary_doc)
                
                # Add summaries to vector store
                if summary_docs:
                    self.vectorstore.add_documents(summary_docs)
                    print(f"✅ Added {len(summary_docs)} summaries")
                
            except Exception as e:
                print(f"⚠️ Summary generation failed: {e}")
        
        # Step 5: Add child documents to vector store (for retrieval)
        if child_docs:
            self.vectorstore.add_documents(child_docs)
            print(f"✅ Added {len(child_docs)} child documents to vector store")
        
        # Step 6: Store parent documents in byte store (for context)
        if parent_docs and doc_ids:
            self.retriever.docstore.mset(list(zip(doc_ids, parent_docs)))
            print(f"✅ Stored {len(parent_docs)} parent documents in byte store")
        
        # Step 7: Save vector store
        try:
            self.vectorstore.save_local(self.persist_directory)
            print(f"💾 Saved multi-vector store to {self.persist_directory}")
        except Exception as e:
            print(f"⚠️ Failed to save vector store: {e}")
    
    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        """
        Retrieve documents using multi-vector strategy
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of parent documents (full context)
        """
        try:
            # Update search parameters
            self.retriever.search_kwargs = {"k": k}
            
            # Retrieve using multi-vector strategy
            # This searches child docs but returns parent docs
            results = self.retriever.invoke(query)
            
            print(f"🔍 Multi-vector retrieval: {len(results)} documents found")
            return results
            
        except Exception as e:
            print(f"❌ Multi-vector retrieval failed: {e}")
            # Fallback to direct vector store search
            try:
                fallback_results = self.vectorstore.similarity_search(query, k=k)
                print(f"🔄 Fallback retrieval: {len(fallback_results)} documents")
                return fallback_results
            except Exception as fallback_error:
                print(f"❌ Fallback retrieval also failed: {fallback_error}")
                return []
    
    def clear_store(self):
        """Clear the vector store and byte store"""
        try:
            # Clear byte store
            self.byte_store = InMemoryByteStore()
            
            # Recreate vector store
            dummy_doc = Document(page_content="initialization", metadata={"type": "dummy"})
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            
            # Recreate retriever
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                byte_store=self.byte_store,
                id_key=self.id_key,
                search_kwargs={"k": 8}
            )
            
            print("🧹 Multi-vector store cleared")
            
        except Exception as e:
            print(f"⚠️ Error clearing store: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the multi-vector store"""
        try:
            # Count documents in vector store
            vector_count = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 0
            
            # Count documents in byte store  
            byte_count = len(self.byte_store.store) if hasattr(self.byte_store, 'store') else 0
            
            return {
                "vector_store_docs": vector_count,
                "byte_store_docs": byte_count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}