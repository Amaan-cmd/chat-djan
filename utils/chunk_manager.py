#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chunk Manager - Add/Update specific chunks in FAISS index
"""
import os
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# Encoding fix
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv()

class ChunkManager:
    def __init__(self, index_path="faiss_gem_clean"):
        self.index_path = index_path
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vectorstore = None
        self._load_index()
    
    def _load_index(self):
        """Load the FAISS index"""
        try:
            self.vectorstore = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded index from {self.index_path}")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
    
    def add_chunk(self, pdf_id, field_name, value, page=1, extraction_type="structured"):
        """Add a single chunk to the index"""
        
        if not self.vectorstore:
            print("❌ Index not loaded")
            return False
        
        # Create clean content
        content = f"{field_name}: {value}"
        
        # Create document
        doc = Document(
            page_content=content,
            metadata={
                'pdf_id': str(pdf_id),
                'page': page,
                'extraction_type': extraction_type,
                'field': field_name.lower().replace(' ', '_'),
                'source': 'key_value'
            }
        )
        
        # Add to index
        self.vectorstore.add_documents([doc])
        print(f"✅ Added: {content} (PDF: {pdf_id})")
        
        return True
    
    def add_multiple_chunks(self, chunks_data):
        """Add multiple chunks at once"""
        
        if not self.vectorstore:
            print("❌ Index not loaded")
            return False
        
        documents = []
        
        for chunk in chunks_data:
            content = f"{chunk['field']}: {chunk['value']}"
            
            doc = Document(
                page_content=content,
                metadata={
                    'pdf_id': str(chunk['pdf_id']),
                    'page': chunk.get('page', 1),
                    'extraction_type': chunk.get('extraction_type', 'structured'),
                    'field': chunk['field'].lower().replace(' ', '_'),
                    'source': 'key_value'
                }
            )
            documents.append(doc)
        
        # Add all documents
        self.vectorstore.add_documents(documents)
        
        print(f"✅ Added {len(documents)} chunks:")
        for chunk in chunks_data:
            print(f"   - {chunk['field']}: {chunk['value']} (PDF: {chunk['pdf_id']})")
        
        return True
    
    def save_index(self):
        """Save the updated index"""
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)
            print(f"✅ Index saved to {self.index_path}")
            return True
        return False
    
    def test_chunk(self, query, pdf_id=None, k=3):
        """Test if a chunk can be found"""
        if not self.vectorstore:
            print("❌ Index not loaded")
            return
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        if pdf_id:
            docs = [doc for doc in docs if doc.metadata.get('pdf_id') == str(pdf_id)]
        
        print(f"\n=== Test Query: '{query}' ===")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.page_content}")
            print(f"   PDF: {doc.metadata.get('pdf_id')}, Page: {doc.metadata.get('page')}")

if __name__ == '__main__':
    print("Chunk Manager - Utility for adding missing chunks to FAISS index")
    print("\nUsage:")
    print("manager = ChunkManager()")
    print("manager.add_chunk('7908419', 'EMD Required', 'No')")
    print("manager.save_index()")