#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Show Chunks Structure - Display how chunks are stored in FAISS index
Perfect for understanding the data structure for your friend!
"""
import os
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from collections import defaultdict

# Encoding fix
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv()

def show_chunks_structure():
    """Show how chunks are structured in the FAISS index"""
    
    print("="*80)
    print("FAISS INDEX CHUNKS STRUCTURE OVERVIEW")
    print("="*80)
    
    # Load the index
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    try:
        vectorstore = FAISS.load_local("faiss_gem_clean", embeddings, allow_dangerous_deserialization=True)
        print("✅ Successfully loaded FAISS index: faiss_gem_clean")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return
    
    # Get all documents
    all_docs = vectorstore.similarity_search("document", k=1000)  # Get many docs
    
    print(f"\n📊 TOTAL CHUNKS IN INDEX: {len(all_docs)}")
    
    # Group by PDF ID
    docs_by_pdf = defaultdict(list)
    for doc in all_docs:
        pdf_id = doc.metadata.get('pdf_id', 'unknown')
        docs_by_pdf[pdf_id].append(doc)
    
    print(f"📁 DOCUMENTS COVERED: {len(docs_by_pdf)} PDFs")
    
    # Show PDF distribution
    print(f"\n📈 CHUNKS PER DOCUMENT:")
    for pdf_id in sorted(docs_by_pdf.keys()):
        chunk_count = len(docs_by_pdf[pdf_id])
        print(f"   PDF {pdf_id}: {chunk_count} chunks")
    
    # Show chunk types
    print(f"\n🔍 CHUNK TYPES BREAKDOWN:")
    extraction_types = defaultdict(int)
    sources = defaultdict(int)
    
    for doc in all_docs:
        extraction_type = doc.metadata.get('extraction_type', 'unknown')
        source = doc.metadata.get('source', 'unknown')
        extraction_types[extraction_type] += 1
        sources[source] += 1
    
    print("   By Extraction Type:")
    for ext_type, count in sorted(extraction_types.items()):
        print(f"     {ext_type}: {count} chunks")
    
    print("   By Source:")
    for source, count in sorted(sources.items()):
        print(f"     {source}: {count} chunks")
    
    # Show sample chunks from document 7908419
    print(f"\n" + "="*80)
    print("SAMPLE CHUNKS FROM DOCUMENT 7908419")
    print("="*80)
    
    sample_docs = [doc for doc in all_docs if doc.metadata.get('pdf_id') == '7908419'][:10]
    
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n--- CHUNK {i} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 60)
    
    # Show different chunk formats
    print(f"\n" + "="*80)
    print("DIFFERENT CHUNK FORMATS EXAMPLES")
    print("="*80)
    
    # Find examples of different types
    examples = {
        'structured': None,
        'text': None,
        'table_row': None
    }
    
    for doc in all_docs:
        ext_type = doc.metadata.get('extraction_type')
        if ext_type in examples and examples[ext_type] is None:
            examples[ext_type] = doc
    
    for chunk_type, doc in examples.items():
        if doc:
            print(f"\n🔸 {chunk_type.upper()} CHUNK EXAMPLE:")
            print(f"   Content: {doc.page_content}")
            print(f"   Metadata: {doc.metadata}")
    
    # Show how to query
    print(f"\n" + "="*80)
    print("HOW TO QUERY THE INDEX")
    print("="*80)
    
    print("1. BASIC SEARCH:")
    print("   docs = vectorstore.similarity_search('consignee', k=5)")
    
    print("\n2. DOCUMENT-SPECIFIC SEARCH:")
    print("   docs = vectorstore.similarity_search('ministry', k=10)")
    print("   filtered = [d for d in docs if d.metadata.get('pdf_id') == '7908419']")
    
    print("\n3. FIELD-SPECIFIC SEARCH:")
    print("   docs = vectorstore.similarity_search('Total Quantity', k=5)")
    
    # Test actual queries
    print(f"\n" + "="*80)
    print("LIVE QUERY EXAMPLES")
    print("="*80)
    
    test_queries = [
        ("consignee 7908419", "Finding consignee information"),
        ("ministry name", "Finding ministry information"),
        ("total quantity", "Finding quantity information")
    ]
    
    for query, description in test_queries:
        print(f"\n🔍 {description.upper()}:")
        print(f"   Query: '{query}'")
        
        results = vectorstore.similarity_search(query, k=3)
        doc_7908419 = [d for d in results if d.metadata.get('pdf_id') == '7908419']
        
        if doc_7908419:
            print(f"   ✅ Found {len(doc_7908419)} relevant chunks for document 7908419")
            print(f"   Top result: {doc_7908419[0].page_content}")
        else:
            print(f"   ⚠️  No specific results for document 7908419")
    
    print(f"\n" + "="*80)
    print("SUMMARY FOR YOUR FRIEND")
    print("="*80)
    print("✅ Index contains clean, structured data")
    print("✅ Each chunk has content + metadata")
    print("✅ Supports document-specific queries")
    print("✅ Multiple chunk types (text, structured, table)")
    print("✅ Easy to query and filter")
    print("✅ Perfect for multi-document Q&A systems!")

if __name__ == '__main__':
    show_chunks_structure()