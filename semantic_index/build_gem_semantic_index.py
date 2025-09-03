#!/usr/bin/env python
"""
Build a clean FAISS index for GeM PDFs with minimal noise and full fidelity:
- Parse text and tables separately (pdfplumber for both; optional Camelot if available)
- Semantic text chunking by headings/clauses/bullets (line-preserving)
- Table rows as individual chunks with header context
- OpenAI embeddings (text-embedding-3-large)
- Rich metadata: pdf_id, page, section, table headers, row index, etc.
"""
from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except Exception:
    pdfplumber = None  # type: ignore
    _HAS_PDFPLUMBER = False

# Optional Camelot for better table extraction if available
try:
    import camelot  # type: ignore
    _HAS_CAMELOT = True
except Exception:
    camelot = None  # type: ignore
    _HAS_CAMELOT = False

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# Load env so GOOGLE_API_KEY is available without new env setup
load_dotenv(override=True)

# OpenAI embeddings (LangChain-style interface)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Fallback embeddings using existing Google Generative AI (already in project)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings as _GoogleEmb
    _HAS_GOOGLE_EMB = True
except Exception:
    _GoogleEmb = None  # type: ignore
    _HAS_GOOGLE_EMB = False


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


class OpenAIEmbeddingsLC:
    """Minimal LangChain Embeddings-compatible wrapper for OpenAI 1.x SDK."""
    def __init__(self, model: str = "text-embedding-3-large", api_key: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package not installed.")
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not (api_key or os.getenv("OPENAI_API_KEY")):
            raise RuntimeError("OPENAI_API_KEY is not set. Please set it and retry.")

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch in groups to stay within token limits
        out: List[List[float]] = []
        B = 64
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out


class GoogleEmbeddingsLC:
    """LangChain Embeddings-compatible wrapper for Google Generative AI embeddings already used in the app."""
    def __init__(self, model: str = "models/text-embedding-004", api_key: Optional[str] = None):
        if not _HAS_GOOGLE_EMB:
            raise RuntimeError("langchain-google-genai not available.")
        self.emb = _GoogleEmb(model=model, google_api_key=api_key or os.getenv("GOOGLE_API_KEY"))

    def embed_query(self, text: str) -> List[float]:
        return self.emb.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.emb.embed_documents(texts)


class PDFExtractor:
    """Extracts text (page-wise) and tables (row-wise) from PDF using pdfplumber/Camelot."""
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        self.pdf_path = pdf_path
        m = re.search(r"GeM-Bidding-(\d+)\.pdf", os.path.basename(pdf_path), re.I)
        self.pdf_id = m.group(1) if m else None

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\(cid:\d+\)", "", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        # Preserve newlines; collapse excessive spaces
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def extract_pages(self) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        if _HAS_PDFPLUMBER:
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    txt = page.extract_text(x_tolerance=1.5, y_tolerance=3.0) or ""
                    out.append((i, self._clean_text(txt)))
            return out
        # Fallback: use LangChain's PyPDFLoader (already present in project)
        try:
            from langchain_community.document_loaders import PyPDFLoader  # type: ignore
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            for i, p in enumerate(pages, 1):
                out.append((i, self._clean_text(p.page_content or "")))
        except Exception:
            pass
        return out

    def extract_tables(self) -> List[Tuple[int, List[List[str]]]]:
        rows: List[Tuple[int, List[List[str]]]] = []
        # Prefer Camelot if available (lattice+stream), else pdfplumber tables
        if _HAS_CAMELOT:
            try:
                tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='lattice')
                for t in tables:
                    data = [list(map(lambda x: (x or '').strip(), row)) for row in t.df.values.tolist()]
                    page = t.page or 1
                    rows.append((page, data))
            except Exception:
                pass
        # Fallback/augment with pdfplumber
        if _HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(self.pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages, 1):
                        for table in page.extract_tables() or []:
                            data = [list(map(lambda x: (x or '').strip(), row)) for row in table]
                            rows.append((i, data))
            except Exception:
                pass
        return rows


class SemanticChunker:
    """Line-preserving semantic chunking by headings and bullets with small windows."""
    HEADING_HINTS = [
        r"^\s*(Bid\s+Number)\b",
        r"^\s*(Bid\s+Opening\s+Date/Time)\b",
        r"^\s*(Bid\s+End\s+Date/Time)\b",
        r"^\s*(Bid\s+Offer\s+Validity)\b",
        r"^\s*(Ministry/State\s+Name)\b",
        r"^\s*(Department\s+Name)\b",
        r"^\s*(Organisat(?:ion|ion)\s+Name)\b",
        r"^\s*(Documents\s+required\s+from\s+seller)\b",
        r"^\s*(Past\s+Performance)\b",
        r"^\s*(Evaluation\s+Method)\b",
        r"^\s*(Payment\s+Terms)\b",
        r"^\s*(Option\s+Clause)\b",
        r"^\s*(Purchase\s+Preference)\b",
        r"^\s*(Consignee|Delivery\s+To)\b",
        r"^\s*(Delivery\s+Days|Delivery\s+Period)\b",
        r"^\s*(Item\s+Category|Total\s+Quantity)\b",
        r"^\s*(Buyer\s+Added\s+Bid\s+Specific\s+ATC|Buyer\s+added)\b",
        r"^\s*(Compliance)\b",
        r"^\s*(Technical\s+Specifications|Specification)\b",
    ]

    def __init__(self, max_lines: int = 22, overlap: int = 5):
        self.max_lines = max_lines
        self.overlap = overlap
        self._heading_re = re.compile("|".join(self.HEADING_HINTS), re.I)

    def split_page(self, page_text: str) -> List[Tuple[str, str]]:
        """Return list of (section_name, section_text) per page."""
        lines = [ln.rstrip() for ln in page_text.split("\n")]
        sections: List[Tuple[str, List[str]]] = []
        current_name = "Page_Section"
        current_buf: List[str] = []

        def flush():
            nonlocal current_name, current_buf
            if current_buf:
                sections.append((current_name, current_buf))
            current_name = "Page_Section"
            current_buf = []

        for ln in lines:
            if self._heading_re.search(ln) or (ln.isupper() and 3 <= len(ln) <= 80) or ln.strip().endswith(":"):
                flush()
                current_name = re.sub(r"\s+", "_", ln.strip())[:80]
                current_buf = [ln]
            else:
                # Keep bullets tightly with prior line
                if re.match(r"^\s*(?:[-*•]\s+|\d+\.)", ln):
                    current_buf.append(ln)
                else:
                    current_buf.append(ln)
        flush()
        # Window inside each section
        out: List[Tuple[str, str]] = []
        for name, buf in sections:
            if len(buf) <= self.max_lines:
                out.append((name, "\n".join(buf).strip()))
            else:
                for i in range(0, len(buf), self.max_lines - self.overlap):
                    win = buf[i:i+self.max_lines]
                    if len(win) < 3:
                        continue
                    out.append((name, "\n".join(win).strip()))
        return out


class TableChunker:
    """Create one chunk per row, include header context in metadata and text."""
    def rows_to_chunks(self, page: int, table: List[List[str]], base_meta: Dict[str, Any]) -> List[Chunk]:
        if not table or len(table) < 2:
            return []
        header = [h.strip() for h in table[0]]
        chunks: List[Chunk] = []
        for idx, row in enumerate(table[1:], 1):
            cols = [c.strip() for c in row]
            # Build a readable row text with headers for context
            pairs = []
            for h, v in zip(header, cols):
                if h and v:
                    pairs.append(f"{h}: {v}")
            content = "TABLE_ROW\n" + " | ".join(pairs)
            meta = dict(base_meta)
            meta.update({
                "extraction_type": "table_row",
                "table_headers": header,
                "row_index": idx,
            })
            chunks.append(Chunk(text=content, metadata=meta))
        return chunks


class IndexBuilder:
    def __init__(self, embeddings: OpenAIEmbeddingsLC):
        self.embeddings = embeddings

    def build_for_pdfs(self, pdf_paths: List[str], save_dir: str) -> FAISS:
        docs: List[Document] = []
        for path in pdf_paths:
            extractor = PDFExtractor(path)
            pdf_id = extractor.pdf_id
            # Text sections
            pages = extractor.extract_pages()
            chunker = SemanticChunker()
            for page_num, text in pages:
                for section_name, section_text in chunker.split_page(text):
                    if not section_text or len(section_text) < 20:
                        continue
                    meta = {
                        "source": os.path.basename(path),
                        "pdf_id": pdf_id,
                        "page": page_num,
                        "section": section_name,
                        "extraction_type": "text",
                    }
                    docs.append(Document(page_content=section_text, metadata=meta))

            # Tables
            tchunker = TableChunker()
            for page_num, table in extractor.extract_tables():
                base_meta = {
                    "source": os.path.basename(path),
                    "pdf_id": pdf_id,
                    "page": page_num,
                    "section": "Table",
                }
                for ch in tchunker.rows_to_chunks(page_num, table, base_meta):
                    docs.append(Document(page_content=ch.text, metadata=ch.metadata))

        if not docs:
            raise RuntimeError("No documents extracted to index.")

        store = FAISS.from_documents(docs, self.embeddings)
        os.makedirs(save_dir, exist_ok=True)
        store.save_local(save_dir)
        return store


def discover_pdfs(doc_dir: str = "documents") -> List[str]:
    out: List[str] = []
    if not os.path.isdir(doc_dir):
        return out
    for fn in os.listdir(doc_dir):
        if fn.lower().startswith("gem-bidding-") and fn.lower().endswith(".pdf"):
            out.append(os.path.join(doc_dir, fn))
    out.sort()
    return out


def build_index(save_dir: str = "faiss_gem_semantic") -> None:
    # Prefer OpenAI if available; else fall back to Google embeddings already in the project
    embeddings = None
    if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
        try:
            embeddings = OpenAIEmbeddingsLC(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            embeddings = None
    if embeddings is None and _HAS_GOOGLE_EMB and os.getenv("GOOGLE_API_KEY"):
        embeddings = GoogleEmbeddingsLC(api_key=os.getenv("GOOGLE_API_KEY"))
    if embeddings is None:
        raise RuntimeError("No embeddings configured. Set OPENAI_API_KEY or GOOGLE_API_KEY.")
    builder = IndexBuilder(embeddings)
    pdfs = discover_pdfs()
    if not pdfs:
        print("No PDFs found in documents/.")
        return
    store = builder.build_for_pdfs(pdfs, save_dir)
    print(f"✅ Built and saved FAISS index to {save_dir} with {len(store.docstore._dict)} chunks.")


def query_index(query: str, k: int = 8, index_dir: str = "faiss_gem_semantic") -> List[Document]:
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddingsLC(api_key=api_key)
    store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return store.similarity_search(query, k=k)


if __name__ == "__main__":
    # CLI: python semantic_index/build_gem_semantic_index.py build|query "your question"
    import argparse
    p = argparse.ArgumentParser(description="Build or query GeM semantic FAISS index")
    p.add_argument("action", choices=["build", "query"], help="build the index or run a test query")
    p.add_argument("query", nargs="?", help="query string for 'query' action")
    p.add_argument("--save", dest="save", default="faiss_gem_semantic")
    args = p.parse_args()
    if args.action == "build":
        build_index(save_dir=args.save)
    else:
        if not args.query:
            print("Provide a query.")
            sys.exit(2)
        docs = query_index(args.query, k=5, index_dir=args.save)
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            print(f"\n[{i}] {meta.get('source')} p{meta.get('page')} sec={meta.get('section')} type={meta.get('extraction_type')}")
            print((d.page_content or "").split("\n")[0][:200])
