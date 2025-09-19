"""
GeM PDF Downloader and Processor
"""
import os
import re
import requests
import pdfplumber
import time
from pathlib import Path
from typing import List, Optional, Dict
from langchain.schema import Document


class GemDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.download_dir = Path("gem_downloads")
        self.download_dir.mkdir(exist_ok=True)
    
    def get_gem_session_and_csrf(self):
        """Get authenticated session with CSRF token - multiple attempts"""
        import random
        
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ]
        
        for attempt in range(3):
            try:
                session = requests.Session()
                user_agent = random.choice(user_agents)
                
                print(f"Attempt {attempt + 1}: Getting session from GeM portal...")
                
                # Try different entry points
                entry_urls = [
                    "https://bidplus.gem.gov.in/all-bids",
                    "https://bidplus.gem.gov.in/",
                    "https://gem.gov.in/"
                ]
                
                for url in entry_urls:
                    try:
                        response = session.get(url, headers={"User-Agent": user_agent}, timeout=20)
                        if response.status_code == 200:
                            print(f"✅ Connected to {url}")
                            break
                    except:
                        continue
                
                # Extract CSRF token
                csrf_token = session.cookies.get("csrf_gem_cookie")
                
                headers = {
                    "User-Agent": user_agent,
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "Origin": "https://bidplus.gem.gov.in",
                    "Referer": "https://bidplus.gem.gov.in/all-bids",
                    "X-Requested-With": "XMLHttpRequest"
                }
                
                return session, headers, csrf_token
                
            except Exception as e:
                print(f"Session attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(random.uniform(2, 5))
                    
        raise Exception("Failed to establish session with GeM portal")
    
    def download_pdf(self, bid_id: str) -> Optional[str]:
        """Download PDF using authenticated session with CSRF token"""
        filepath = self.download_dir / f"bid_{bid_id}.pdf"
        
        try:
            print(f"[DEBUG] Starting download for bid {bid_id}")
            
            # Get authenticated session
            session, headers, csrf_token = self.get_gem_session_and_csrf()
            print(f"[DEBUG] Session established, CSRF token: {csrf_token[:10] if csrf_token else 'None'}...")
            
            # Try multiple URL patterns like mentor's project
            urls_to_try = [
                f"https://bidplus.gem.gov.in/showbidDocument/{bid_id}",
                f"https://mkp.gem.gov.in/showbidDocument/{bid_id}",
                f"https://gem.gov.in/showbidDocument/{bid_id}"
            ]
            
            response = None
            for i, url in enumerate(urls_to_try):
                print(f"[DEBUG] Attempt {i+1}: {url}")
                try:
                    response = session.get(url, headers=headers, timeout=30)  # Increased timeout
                    print(f"[DEBUG] Response status: {response.status_code}, Size: {len(response.content)} bytes")
                    
                    # Check if it's actually a PDF
                    if response.status_code == 200:
                        if response.content.startswith(b'%PDF'):
                            print(f"[SUCCESS] Found PDF at: {url}")
                            break
                        else:
                            # Check if it's HTML (error page)
                            content_preview = response.content[:200].decode('utf-8', errors='ignore')
                            print(f"[DEBUG] Not PDF, content preview: {content_preview}")
                    else:
                        print(f"[DEBUG] HTTP error {response.status_code}")
                        
                except Exception as e:
                    print(f"[DEBUG] Exception with {url}: {e}")
                    continue
            
            if not response or not response.content.startswith(b'%PDF'):
                print(f"[ERROR] No valid PDF found from any URL")
                return None
            
            # Save the PDF
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"[SUCCESS] Downloaded PDF to {filepath}")
            return str(filepath)
                
        except Exception as e:
            print(f"[ERROR] Download exception: {e}")
            import traceback
            traceback.print_exc()
            return None
        

    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text with enhanced table processing"""
        try:
            import pdfplumber
            
            full_text = ""
            table_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract regular text
                    page_text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
                    if page_text:
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    # Extract tables separately
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_text = f"\n--- Table {table_idx + 1} on Page {page_num + 1} ---\n"
                            for row in table:
                                if row and any(cell for cell in row if cell):
                                    # Join cells with proper spacing
                                    row_text = " | ".join(str(cell or "").strip() for cell in row)
                                    table_text += row_text + "\n"
                            table_data.append(table_text)
            
            # Combine text and tables
            combined_text = full_text + "\n\n--- EXTRACTED TABLES ---\n" + "\n".join(table_data)
            
            return self.clean_pdf_content_mentor_style(combined_text)
            
        except Exception as e:
            print(f"Enhanced text extraction error: {e}")
            # Fallback to original method
            try:
                from langchain_community.document_loaders import PDFPlumberLoader
                loader = PDFPlumberLoader(pdf_path, text_kwargs={"layout": True, "x_tolerance": 3, "y_tolerance": 3})
                documents = loader.load()
                full_text = "\n".join([doc.page_content for doc in documents])
                return self.clean_pdf_content_mentor_style(full_text)
            except:
                return ""
    
    def clean_pdf_content_mentor_style(self, content: str) -> str:
        """Clean PDF content using mentor's exact method"""
        lines = content.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Remove (cid:...) codes
            line = re.sub(r"\(cid:\d+\)", "", line)
            
            # Remove Hindi / non-ASCII but keep spaces, alignment, and email symbols
            line = ''.join(ch if ord(ch) < 128 or ch.isspace() or ch in '@.' else '' for ch in line)
            
            # Remove page numbers like "7 / 8", "12 / 15", etc.
            if re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", line.strip()):
                continue
            
            # Remove "Thank You" or similar boilerplate lines (case-insensitive)
            if re.search(r"-{2,}\s*thank\s*you\s*-{2,}", line, flags=re.IGNORECASE):
                continue
            
            # Strip only trailing spaces (don't collapse middle spaces!)
            cleaned_lines.append(line.rstrip())
        
        # Rejoin while keeping alignment intact
        cleaned_content = "\n".join(cleaned_lines)
        
        # Cleanup: normalize slashes with spacing
        cleaned_content = re.sub(r"\s*/\s*", " / ", cleaned_content)
        
        return cleaned_content.strip()