"""
Live Multi-Vector Processor for GeM Integration
Enhances live PDF processing with multi-vector retrieval
"""
import os
import uuid
import shutil
from typing import List, Optional
from pathlib import Path
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LiveMultiVectorProcessor:
    """
    Enhanced processor for live GeM PDFs using multi-vector retrieval
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, embeddings=None, llm=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, embeddings, llm):
        if self._initialized:
            return
            
        self.embeddings = embeddings
        self.llm = llm
        self.live_index_path = "faiss_gem_live_multi"
        
        # Text splitters optimized for GeM documents
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Smaller for GeM docs
            chunk_overlap=150
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,   # Precise for field extraction
            chunk_overlap=30
        )
        
        # Storage components - use persistent storage
        self.byte_store_path = f"{self.live_index_path}_bytestore"
        self.byte_store = self._get_persistent_byte_store()
        self.id_key = "doc_id"
        
        # Initialize fresh for each PDF
        self._setup_fresh_retriever()
        self._initialized = True
    
    def _get_persistent_byte_store(self):
        """Get persistent byte store using file system"""
        try:
            from langchain.storage import LocalFileStore
            return LocalFileStore(self.byte_store_path)
        except:
            # Fallback to in-memory if LocalFileStore not available
            return InMemoryByteStore()
    
    def _setup_fresh_retriever(self):
        """Setup fresh retriever for new PDF processing"""
        # Try to load existing components first
        if os.path.exists(self.live_index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.live_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("📂 Loaded existing live index")
            except:
                self._create_fresh_components()
        else:
            self._create_fresh_components()
        
        # Setup byte store
        self.byte_store = self._get_persistent_byte_store()
        
        # Create retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.byte_store,
            id_key=self.id_key,
            search_kwargs={"k": 8}
        )
        
        print("🔄 Multi-vector retriever ready")
    
    def _create_fresh_components(self):
        """Create fresh vector store components"""
        dummy_doc = Document(page_content="init", metadata={"type": "init"})
        self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
        print("🆕 Created fresh vector store")
    
    def process_live_pdf(self, content: str, bid_id: str) -> int:
        """
        Process live PDF content with multi-vector strategy
        
        Args:
            content: Extracted PDF text
            bid_id: GeM bid ID
            
        Returns:
            Number of chunks created
        """
        print(f"📄 Processing PDF {bid_id} with multi-vector approach...")
        
        # Create base document
        base_doc = Document(
            page_content=content,
            metadata={
                "bid_id": bid_id,
                "source": "live_gem",
                "extraction_type": "full_document"
            }
        )
        
        # Step 1: Create parent documents (context chunks)
        parent_docs = self.parent_splitter.split_documents([base_doc])
        
        # Step 2: Generate unique IDs
        doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
        
        # Step 3: Create child documents (retrieval chunks)
        child_docs = []
        for i, parent_doc in enumerate(parent_docs):
            children = self.child_splitter.split_documents([parent_doc])
            for child in children:
                child.metadata.update({
                    self.id_key: doc_ids[i],
                    "bid_id": bid_id,
                    "source": "live_gem",
                    "extraction_type": "child_chunk"
                })
                child_docs.append(child)
        
        print(f"📊 Created {len(parent_docs)} parent + {len(child_docs)} child chunks")
        
        # Step 4: Add to vector store and byte store
        if child_docs:
            self.vectorstore.add_documents(child_docs)
        
        if parent_docs and doc_ids:
            self.retriever.docstore.mset(list(zip(doc_ids, parent_docs)))
        
        # Step 5: Save to persistent storage
        self.vectorstore.save_local(self.live_index_path)
        
        print(f"✅ Multi-vector processing complete for {bid_id}")
        return len(parent_docs)
    
    def search_live_content(self, query: str, k: int = 6) -> List[Document]:
        """
        Search live content using multi-vector retrieval
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of parent documents with full context
        """
        try:
            # Load live index if exists
            if os.path.exists(self.live_index_path):
                self.vectorstore = FAISS.load_local(
                    self.live_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Reload byte store
                self.byte_store = self._get_persistent_byte_store()
                
                # Recreate retriever with loaded components
                self.retriever = MultiVectorRetriever(
                    vectorstore=self.vectorstore,
                    byte_store=self.byte_store,
                    id_key=self.id_key,
                    search_kwargs={"k": k}
                )
            
            # Try multi-vector search first
            try:
                results = self.retriever.invoke(query)
                if results and len(results) > 0:
                    # Filter out dummy init documents
                    results = [doc for doc in results if doc.page_content != "init"]
            except Exception as e:
                print(f"Multi-vector search failed: {e}")
                results = []
            
            # Fallback to direct vector search if no results
            if not results:
                results = self.vectorstore.similarity_search(query, k=k)
                results = [doc for doc in results if doc.page_content != "init"]
            print(f"🔍 Live multi-vector search: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"⚠️ Multi-vector search failed: {e}")
            # Fallback to direct vector search
            try:
                fallback = self.vectorstore.similarity_search(query, k=k)
                print(f"🔄 Fallback search: {len(fallback)} results")
                return fallback
            except:
                return []
    
    def clear_live_index(self):
        """Clear the live multi-vector index"""
        if os.path.exists(self.live_index_path):
            shutil.rmtree(self.live_index_path)
        if os.path.exists(self.byte_store_path):
            shutil.rmtree(self.byte_store_path)
        
        # Create fresh components
        self._create_fresh_components()
        self.byte_store = self._get_persistent_byte_store()
        
        # Recreate retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.byte_store,
            id_key=self.id_key,
            search_kwargs={"k": 8}
        )
        
        print("🧹 Live multi-vector index cleared")