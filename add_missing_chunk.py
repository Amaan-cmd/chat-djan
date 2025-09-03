#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add Missing Total Quantity Chunk to Clean Index
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

def add_total_quantity_chunk():
    """Add the missing total quantity chunk"""
    
    print("=== ADDING MISSING TOTAL QUANTITY CHUNK ===\n")
    
    # Load existing index
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    try:
        vectorstore = FAISS.load_local("faiss_gem_clean", embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded existing clean index")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return
    
    # Create the missing chunk for total quantity
    total_quantity_doc = Document(
        page_content="Total Quantity: 13200",
        metadata={
            'pdf_id': '7908419',
            'page': 1,
            'extraction_type': 'structured',
            'field': 'total_quantity',
            'source': 'key_value'
        }
    )
    
    # Add the document to the existing index
    vectorstore.add_documents([total_quantity_doc])
    
    # Save the updated index
    vectorstore.save_local("faiss_gem_clean")
    
    print("✅ Added Total Quantity: 13200 chunk for document 7908419")
    print("✅ Updated index saved")
    
    # Test the addition
    print("\n=== TESTING UPDATED INDEX ===")
    test_docs = vectorstore.similarity_search("total quantity 7908419", k=5)
    
    for i, doc in enumerate(test_docs):
        if doc.metadata.get('pdf_id') == '7908419':
            print(f"{i+1}. {doc.page_content}")
            print(f"   Metadata: {doc.metadata}")

if __name__ == '__main__':
    add_total_quantity_chunk()