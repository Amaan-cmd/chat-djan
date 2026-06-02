#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rebuild FAISS Index - Clean English-Only with Perfect Key-Value Pairs
"""
import os
import sys
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Encoding fix
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv()

class CleanIndexBuilder:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.documents = []
    
    def clean_text(self, text: str) -> str:
        """Remove Hindi and clean text for perfect English extraction"""
        if not text:
            return ""
        
        # Remove Hindi/Devanagari characters
        text = re.sub(r'[\u0900-\u097F]', '', text)
        
        # Remove PDF artifacts
        text = re.sub(r'\(cid:\d+\)', '', text)
        
        # Clean special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\-\.,:/()@#%&]', ' ', text)
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_key_value_pairs(self, text: str, pdf_id: str, page_num: int) -> List[Document]:
        """Extract clean key-value pairs from text"""
        
        documents = []
        clean_text = self.clean_text(text)
        
        # Key patterns to extract
        key_patterns = {
            'Consignee Reporting Officer': r'Consignee\s+Reporting[/\\]?Officer[:\s]*([^|/]+?)(?:\s*Address|$)',
            'Address': r'Address[:\s]*([^|/]+?)(?:\s*Quantity|$)',
            'Quantity': r'Quantity[:\s]*(\d+)',
            'Delivery Days': r'Delivery\s+Days[:\s]*(\d+)',
            'Ministry': r'Ministry[:\s]*([^|/]+?)(?:\s*Department|$)',
            'Department': r'Department[:\s]*([^|/]+?)(?:\s*Organisation|$)',
            'Organisation': r'Organisation[:\s]*([^|/]+?)(?:\s*Office|$)',
            'Office Name': r'Office\s+Name[:\s]*([^|/]+?)(?:\s*Buyer|$)',
            'Bid End Date': r'Bid\s+End\s+Date[/\\]?Time[:\s]*([^|/]+?)(?:\s*Bid\s+Opening|$)',
            'Bid Opening Date': r'Bid\s+Opening\s+Date[/\\]?Time[:\s]*([^|/]+?)(?:\s*Item|$)',
            'Item Category': r'Item\s+Category[:\s]*([^|/]+?)(?:\s*Documents|$)',
            'EMD Required': r'EMD[:\s]*(Required|Not Required|Yes|No)',
            'MSE Purchase Preference': r'MSE\s+Purchase\s+Preference[:\s]*(Yes|No)',
        }
        
        for field_name, pattern in key_patterns.items():
            matches = re.finditer(pattern, clean_text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if len(value) > 1 and value.lower() not in ['yes', 'no', 'required']:
                    # Clean the value further
                    value = re.sub(r'\s+', ' ', value)
                    value = value[:200]  # Limit length
                    
                    if len(value) > 3:  # Only meaningful values
                        doc = Document(
                            page_content=f"{field_name}: {value}",
                            metadata={
                                'pdf_id': pdf_id,
                                'page': page_num,
                                'extraction_type': 'structured',
                                'field': field_name.lower().replace(' ', '_'),
                                'source': 'key_value'
                            }
                        )
                        documents.append(doc)
        
        return documents
    
    def extract_table_rows(self, text: str, pdf_id: str, page_num: int) -> List[Document]:
        """Extract and format table rows as clean key-value pairs"""
        
        documents = []
        clean_text = self.clean_text(text)
        
        # Look for table-like structures with multiple fields
        lines = clean_text.split('\n')
        
        for line_num, line in enumerate(lines):
            # Skip short lines
            if len(line.strip()) < 20:
                continue
            
            # Look for lines with consignee information
            if 'consignee' in line.lower() and ('officer' in line.lower() or 'address' in line.lower()):
                
                # Extract structured data from the line
                extracted_data = {}
                
                # Try to parse as delimited data (|, /, :)
                if '|' in line:
                    parts = line.split('|')
                elif '/' in line and line.count('/') > 2:
                    parts = line.split('/')
                else:
                    parts = [line]
                
                for part in parts:
                    part = part.strip()
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if len(key) > 2 and len(value) > 1:
                            extracted_data[key] = value
                
                # Create individual documents for each key-value pair
                for key, value in extracted_data.items():
                    if len(value) > 2:
                        doc = Document(
                            page_content=f"{key}: {value}",
                            metadata={
                                'pdf_id': pdf_id,
                                'page': page_num,
                                'extraction_type': 'table_row',
                                'field': key.lower().replace(' ', '_'),
                                'source': 'table',
                                'row_index': line_num
                            }
                        )
                        documents.append(doc)
        
        return documents
    
    def extract_narrative_text(self, text: str, pdf_id: str, page_num: int) -> List[Document]:
        """Extract clean narrative text chunks"""
        
        clean_text = self.clean_text(text)
        
        # Skip if too short or mostly structured data
        if len(clean_text) < 100:
            return []
        
        # Split into meaningful chunks
        sentences = re.split(r'[.!?]+', clean_text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += sentence + ". "
            else:
                if len(current_chunk.strip()) > 50:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add final chunk
        if len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'pdf_id': pdf_id,
                    'page': page_num,
                    'extraction_type': 'text',
                    'source': 'narrative',
                    'chunk_index': i
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF and extract all document types"""
        
        pdf_id = os.path.basename(pdf_path).replace('GeM-Bidding-', '').replace('.pdf', '')
        print(f"Processing PDF {pdf_id}...")
        
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            all_docs = []
            
            for page_num, page in enumerate(pages, 1):
                content = page.page_content
                
                # Extract different types of content
                kv_docs = self.extract_key_value_pairs(content, pdf_id, page_num)
                table_docs = self.extract_table_rows(content, pdf_id, page_num)
                text_docs = self.extract_narrative_text(content, pdf_id, page_num)
                
                all_docs.extend(kv_docs)
                all_docs.extend(table_docs)
                all_docs.extend(text_docs)
            
            print(f"  Extracted {len(all_docs)} documents from {pdf_id}")
            return all_docs
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return []
    
    def build_index(self):
        """Build the complete clean index"""
        
        print("=== BUILDING CLEAN FAISS INDEX ===\n")
        
        # Process all PDFs
        docs_dir = "documents"
        pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
        
        all_documents = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(docs_dir, pdf_file)
            docs = self.process_pdf(pdf_path)
            all_documents.extend(docs)
        
        print(f"\nTotal documents extracted: {len(all_documents)}")
        
        # Show sample of extracted data
        print("\n=== SAMPLE EXTRACTED DATA ===")
        consignee_docs = [doc for doc in all_documents if 'consignee' in doc.page_content.lower()]
        
        for i, doc in enumerate(consignee_docs[:5]):
            print(f"{i+1}. {doc.page_content}")
            print(f"   Metadata: {doc.metadata}")
            print()
        
        if len(all_documents) == 0:
            print("ERROR: No documents extracted!")
            return
        
        # Build FAISS index
        print("Building FAISS index...")
        
        # Create new clean index
        vectorstore = FAISS.from_documents(all_documents, self.embeddings)
        
        # Save the index
        output_path = "faiss_gem_clean"
        vectorstore.save_local(output_path)
        
        print(f"✅ Clean index saved to {output_path}")
        print(f"✅ Total chunks: {len(all_documents)}")
        
        # Test the index
        self.test_index(vectorstore)
    
    def test_index(self, vectorstore):
        """Test the new clean index"""
        
        print("\n=== TESTING CLEAN INDEX ===")
        
        test_queries = [
            "consignee reporting officer",
            "address",
            "Koil Sudahar Krishnan",
            "delivery days"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            docs = vectorstore.similarity_search(query, k=3)
            
            for i, doc in enumerate(docs):
                print(f"  {i+1}. {doc.page_content}")
                print(f"     PDF: {doc.metadata.get('pdf_id')}, Page: {doc.metadata.get('page')}")

if __name__ == '__main__':
    builder = CleanIndexBuilder()
    builder.build_index()