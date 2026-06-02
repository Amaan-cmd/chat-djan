import pdfplumber
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
import os

class HybridPDFExtractor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_all_content(self, pdf_path):
        """Extract both text and tables from PDF"""
        documents = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text_content = self._extract_text_content(page, page_num)
                if text_content:
                    documents.extend(text_content)
                
                # Extract tables
                table_content = self._extract_table_content(page, page_num)
                if table_content:
                    documents.extend(table_content)
        
        return documents
    
    def _extract_text_content(self, page, page_num):
        """Extract and process text content"""
        text = page.extract_text()
        if not text or len(text.strip()) < 50:
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 30:
                # Extract structured info from chunk
                metadata = self._extract_metadata(chunk)
                metadata.update({
                    'source': 'text',
                    'page': page_num + 1,
                    'chunk_id': f"text_{page_num}_{i}"
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
        
        return documents
    
    def _extract_table_content(self, page, page_num):
        """Extract and process table content"""
        tables = page.extract_tables()
        documents = []
        
        for table_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue
            
            # Convert table to structured text
            table_text = self._table_to_text(table)
            
            if len(table_text.strip()) > 30:
                metadata = self._extract_metadata(table_text)
                metadata.update({
                    'source': 'table',
                    'page': page_num + 1,
                    'table_id': table_idx,
                    'chunk_id': f"table_{page_num}_{table_idx}"
                })
                
                documents.append(Document(
                    page_content=table_text,
                    metadata=metadata
                ))
        
        return documents
    
    def _table_to_text(self, table):
        """Convert table to readable text format"""
        if not table:
            return ""
        
        # Get headers (first row)
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        
        # Process data rows
        text_parts = []
        for row in table[1:]:
            row_data = []
            for i, cell in enumerate(row):
                if cell and str(cell).strip():
                    header = headers[i] if i < len(headers) and headers[i] else f"Column_{i}"
                    row_data.append(f"{header}: {str(cell).strip()}")
            
            if row_data:
                text_parts.append(" | ".join(row_data))
        
        return "\n".join(text_parts)
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\/\%\₹]', '', text)
        
        # Remove Hindi/Devanagari characters
        text = re.sub(r'[\u0900-\u097F]+', '', text)
        
        return text.strip()
    
    def _extract_metadata(self, content):
        """Extract structured metadata from content"""
        metadata = {}
        
        # Document number
        doc_match = re.search(r'(?:Document|Tender|RFP|NIT).*?(\d{7})', content, re.IGNORECASE)
        if doc_match:
            metadata['document_number'] = doc_match.group(1)
        
        # Ministry/Department
        ministry_match = re.search(r'Ministry\s+of\s+([^,\n]+)', content, re.IGNORECASE)
        if ministry_match:
            metadata['ministry'] = ministry_match.group(1).strip()
        
        dept_match = re.search(r'Department\s+of\s+([^,\n]+)', content, re.IGNORECASE)
        if dept_match:
            metadata['department'] = dept_match.group(1).strip()
        
        # Bid type
        bid_match = re.search(r'(Two\s+Packet\s+Bid|Single\s+Packet\s+Bid|Technical\s+Bid)', content, re.IGNORECASE)
        if bid_match:
            metadata['bid_type'] = bid_match.group(1)
        
        # Item category
        item_match = re.search(r'(?:Item|Product|Service|Category).*?([A-Z][^,\n]{10,50})', content, re.IGNORECASE)
        if item_match:
            metadata['item_category'] = item_match.group(1).strip()
        
        # Dates
        date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', content)
        if date_match:
            metadata['date'] = date_match.group(1)
        
        # Amount/Value
        amount_match = re.search(r'(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{2})?)', content)
        if amount_match:
            metadata['amount'] = amount_match.group(1)
        
        return metadata

def create_hybrid_index():
    """Create comprehensive index using hybrid extraction"""
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    
    # Extract content using hybrid method
    extractor = HybridPDFExtractor()
    pdf_path = r"c:\Users\dbleg\PycharmProjects\PythonProject1\documents\GeM-Bidding-8046605.pdf"
    
    print("Extracting content using hybrid method...")
    documents = extractor.extract_all_content(pdf_path)
    
    print(f"Extracted {len(documents)} chunks")
    
    # Show sample of what we extracted
    for i, doc in enumerate(documents[:3]):
        print(f"\nChunk {i+1} ({doc.metadata.get('source', 'unknown')}):")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
    
    # Create vector store
    print("\nCreating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the index
    index_path = "faiss_hybrid_index"
    vectorstore.save_local(index_path)
    print(f"Saved hybrid index to {index_path}")
    
    return vectorstore

if __name__ == "__main__":
    create_hybrid_index()