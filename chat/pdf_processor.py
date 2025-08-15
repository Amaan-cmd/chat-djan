"""
GeM PDF Processing Pipeline
Handles extraction, cleaning, and chunking of GeM bidding documents
"""
import os
import re
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class GeMPDFProcessor:
    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean GeM document text"""
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Convert \r\n to \n
        text = re.sub(r'\r\n', '\n', text)
        
        # Remove Hindi duplicates - more aggressive approach
        # Split by lines and keep only English lines or lines with numbers/dates
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Keep lines that are primarily English or contain important data
            if (re.search(r'[A-Za-z]', line) and 
                not re.search(r'^[\u0900-\u097F\s]+$', line)):
                cleaned_lines.append(line)
            elif re.search(r'\d{2}-\d{2}-\d{4}|\d+|GEM/\d+', line):
                # Keep lines with dates, numbers, or GEM IDs
                cleaned_lines.append(line)
        
        # Join back and clean up
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove asterisks and special formatting
        text = re.sub(r'\*+', '', text)
        
        return text.strip()
    
    def extract_metadata(self, filename: str, content: str) -> Dict:
        """Extract metadata from GeM document"""
        metadata = {"source": filename}
        
        # Extract bid number
        bid_match = re.search(r'GEM/\d{4}/B/\d+', content)
        if bid_match:
            metadata["bid_number"] = bid_match.group()
        
        # Extract ministry
        ministry_match = re.search(r'Ministry Of ([^\n\r]+)', content)
        if ministry_match:
            metadata["ministry"] = ministry_match.group(1).strip()
        
        # Extract department
        dept_match = re.search(r'Department Of ([^\n\r]+)', content)
        if dept_match:
            metadata["department"] = dept_match.group(1).strip()
        
        # Extract item category
        category_match = re.search(r'Item Category/[^\n\r]*\n([^\n\r]+)', content)
        if category_match:
            metadata["category"] = category_match.group(1).strip()
        
        return metadata
    
    def process_single_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single GeM PDF"""
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Combine all pages
            full_text = "\n".join([page.page_content for page in pages])
            
            # Clean text
            cleaned_text = self.clean_text(full_text)
            
            # Extract metadata
            filename = os.path.basename(pdf_path)
            metadata = self.extract_metadata(filename, cleaned_text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk_id"] = i
                doc_metadata["total_chunks"] = len(chunks)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            print(f"  SUCCESS: Extracted {len(chunks)} chunks")
            return documents
            
        except Exception as e:
            print(f"  ERROR: Error processing {pdf_path}: {str(e)}")
            return []
    
    def process_all_pdfs(self) -> List[Document]:
        """Process all PDFs in the documents directory"""
        all_documents = []
        
        pdf_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files")
        print("=" * 50)
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.documents_dir, pdf_file)
            documents = self.process_single_pdf(pdf_path)
            all_documents.extend(documents)
        
        print("=" * 50)
        print(f"Total documents created: {len(all_documents)}")
        return all_documents
    
    def get_processing_summary(self, documents: List[Document]) -> Dict:
        """Get summary of processed documents"""
        summary = {
            'total_chunks': len(documents),
            'documents': {},
            'ministries': set(),
            'categories': set()
        }
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in summary['documents']:
                summary['documents'][source] = 0
            summary['documents'][source] += 1
            
            if 'ministry' in doc.metadata:
                summary['ministries'].add(doc.metadata['ministry'])
            if 'category' in doc.metadata:
                summary['categories'].add(doc.metadata['category'])
        
        return summary

