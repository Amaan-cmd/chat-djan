"""
PDF Upload and Processing Module
Handles user-uploaded PDFs for Q&A
"""
import os
import tempfile
import hashlib
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from django.core.cache import cache
from django.conf import settings

class PDFUploadProcessor:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        # Create uploads directory if it doesn't exist
        self.upload_dir = os.path.join(settings.BASE_DIR, 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def process_uploaded_pdf(self, pdf_file, session_id: str) -> dict:
        """
        Process an uploaded PDF file and create a vector store
        Returns: dict with status and vector_store_id
        """
        try:
            # Generate unique ID for this PDF
            pdf_content = pdf_file.read()
            pdf_file.seek(0)  # Reset file pointer
            
            pdf_hash = hashlib.md5(pdf_content).hexdigest()
            vector_store_id = f"upload_{session_id}_{pdf_hash[:8]}"
            
            # Check if already processed
            cached_store = cache.get(f"pdf_store_{vector_store_id}")
            if cached_store:
                return {
                    'status': 'success',
                    'vector_store_id': vector_store_id,
                    'message': 'PDF already processed and ready for questions'
                }
            
            # Save uploaded file temporarily
            temp_path = os.path.join(self.upload_dir, f"{vector_store_id}.pdf")
            with open(temp_path, 'wb') as f:
                f.write(pdf_content)
            
            # Process the PDF
            documents = self._extract_and_chunk_pdf(temp_path, pdf_file.name)
            
            if not documents:
                return {
                    'status': 'error',
                    'message': 'Could not extract content from PDF'
                }
            
            # Create vector store
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Cache the vector store (30 minutes)
            cache.set(f"pdf_store_{vector_store_id}", vector_store, 1800)
            
            # Store metadata
            metadata = {
                'filename': pdf_file.name,
                'num_chunks': len(documents),
                'vector_store_id': vector_store_id
            }
            cache.set(f"pdf_meta_{vector_store_id}", metadata, 1800)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return {
                'status': 'success',
                'vector_store_id': vector_store_id,
                'filename': pdf_file.name,
                'num_chunks': len(documents),
                'message': f'Successfully processed {pdf_file.name} into {len(documents)} chunks'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing PDF: {str(e)}'
            }
    
    def _extract_and_chunk_pdf(self, pdf_path: str, filename: str) -> List[Document]:
        """Extract text from PDF and create document chunks"""
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Combine all pages
            full_text = "\n\n".join([page.page_content for page in pages])
            
            # Clean text
            full_text = self._clean_text(full_text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only keep substantial chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': filename,
                            'chunk_id': i,
                            'type': 'uploaded_pdf'
                        }
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\/\%\$]', ' ', text)
        
        # Remove excessive dots/dashes
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)
        
        return text.strip()
    
    def get_vector_store(self, vector_store_id: str) -> Optional[FAISS]:
        """Retrieve cached vector store"""
        return cache.get(f"pdf_store_{vector_store_id}")
    
    def get_pdf_metadata(self, vector_store_id: str) -> Optional[dict]:
        """Get metadata for processed PDF"""
        return cache.get(f"pdf_meta_{vector_store_id}")
    
    def search_uploaded_pdf(self, vector_store_id: str, question: str, k: int = 5) -> List[Document]:
        """Search in uploaded PDF"""
        vector_store = self.get_vector_store(vector_store_id)
        if not vector_store:
            return []
        
        try:
            return vector_store.similarity_search(question, k=k)
        except Exception as e:
            print(f"Error searching PDF: {e}")
            return []
    
    def create_pdf_chain(self):
        """Create a chain for answering questions from uploaded PDFs"""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.chains.combine_documents import create_stuff_documents_chain
        
        system_prompt = (
            "You are an AI assistant that answers questions based on uploaded PDF documents. "
            "INSTRUCTIONS:\n"
            "1. Answer ONLY using information from the provided context\n"
            "2. If the context doesn't contain enough information, say 'I don't have enough information about that in the uploaded document'\n"
            "3. Be precise and factual - quote relevant sections when possible\n"
            "4. If the question is outside the scope of the document, clearly state that\n\n"
            "Context from uploaded PDF:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        return create_stuff_documents_chain(self.llm, prompt)
