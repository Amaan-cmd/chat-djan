"""
ScopedRetriever: Centralized retrieval with pdf_id scoping and structured-first ordering.
"""
from collections import Counter
import re
from typing import List, Optional
from langchain.schema import Document


class ScopedRetriever:
    def __init__(self, vector_db):
        self.db = vector_db
        # Canonical label synonyms for overlap/ranking
        self.label_synonyms = {
            "organisation": {"organization", "org", "organisation"},
            "organization": {"organization", "org", "organisation"},
            "bid opening date/time": {"bid opening date", "opening date", "opening time"},
            "bid end date/time": {"bid end date", "closing date", "closing time"},
            "delivery days": {"delivery period", "delivery schedule"},
            "payment terms": {"payment timeline", "payment timelines", "payment"},
            "past performance": {"pastperformance", "experience", "project experience"},
            "evaluation method": {"evaluation", "evaluation methodology"},
            "availability of service centres": {"service centers", "service centres", "service centre availability"},
            "escalation matrix for service support": {"escalation matrix", "service support escalation"},
            "type of bid": {"bid type"},
            "option clause": {"option", "optionclause"},
            # New labels
            "office name": {"office", "office name"},
            "buyer email": {"email", "buyer email", "email address", "mail id", "e-mail"},
            "total quantity": {"quantity", "total quantity"},
            "consignee reporting officer": {"reporting officer", "consignee officer", "reporting/Officer"},
        }

    def _expand_query_tokens(self, query: str) -> set:
        tokens = set(t.lower() for t in query.split())
        q = query.lower()
        for canon, syns in self.label_synonyms.items():
            if canon in q or any(s in q for s in syns):
                tokens |= set(' '.join(canon.split()).split())
                for s in syns:
                    tokens |= set(s.split())
        return tokens

    def _prefer_structured(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        structured = [d for d in docs if d.metadata.get('extraction_type') == 'structured']
        if not structured:
            return docs
        qa = [d for d in structured if d.metadata.get('field') == 'table_qa_pair']
        timing = [d for d in structured if d.metadata.get('field') == 'timing']
        other_structured = [d for d in structured if d not in qa and d not in timing]
        remaining = [d for d in docs if d not in structured]
        return qa + timing + other_structured + remaining

    def _dominant_doc(self, docs: List[Document]) -> Optional[str]:
        ids = [d.metadata.get('pdf_id') for d in docs if d.metadata.get('pdf_id')]
        if not ids:
            return None
        return Counter(ids).most_common(1)[0][0]

    def _rerank_by_query_overlap(self, query: str, docs: List[Document]) -> List[Document]:
        """Simple token-overlap re-rank for small structured docs."""
        if not docs:
            return docs
        qtokens = self._expand_query_tokens(query)
        def score(doc: Document) -> int:
            text = (doc.page_content or "")
            # For structured QA, consider the question field if present
            q = doc.metadata.get('question') or ''
            tkns = set((q + ' ' + text).lower().split())
            return len(qtokens & tkns)
        return sorted(docs, key=score, reverse=True)

    def _inject_targeted_structured(self, query: str, pdf_id: Optional[str], docs: List[Document]) -> List[Document]:
        """For precise label questions (timing), inject the exact structured chunk(s) from the active pdf.

        This helps when vector similarity brings related content but misses the exact 'Bid End/Opening' row.
        """
        if not pdf_id or not hasattr(self.db, 'docstore') or not hasattr(self.db.docstore, '_dict'):
            return docs

        ql = query.lower()
        wants_timing = any(s in ql for s in (
            'bid end date', 'closing date', 'closing time', 'bid opening date', 'opening time', 'offer validity'
        )) or 'bid end date/time' in ql or 'bid opening date/time' in ql
        wants_org = any(s in ql for s in (
            'ministry', 'ministry/state name', 'department name', 'organisation name', 'organization name', 'office name'
        ))
        # Additional simple labels found in tables
        wants_office = ('office name' in ql) or ('office' in ql)
        wants_email = 'email' in ql
        wants_quantity = ('total quantity' in ql) or (('quantity' in ql) and ('total' in ql))
        wants_reporting = (
            'reporting officer' in ql or 'consignee reporting' in ql or 'reporting/officer' in ql or 'reporting officer' in ql
        )

        if not (wants_timing or wants_org or wants_office or wants_email or wants_quantity or wants_reporting):
            return docs

        # Gather candidates from the docstore for this pdf_id
        all_docs = [d for d in self.db.docstore._dict.values() if isinstance(d, Document)]
        cands = []
        for d in all_docs:
            if d.metadata.get('pdf_id') != pdf_id:
                continue
            txt = (d.page_content or '')
            field = str(d.metadata.get('field', '')).lower()
            section = str(d.metadata.get('section', '')).lower()
            # Timing candidates
            if wants_timing and (field == 'timing' or (
                'Bid End Date/Time' in txt or 'Bid Opening Date/Time' in txt or 'Bid Offer Validity' in txt
            )):
                cands.append(d)
            # Organization candidates
            if wants_org and (
                section == 'organization_details'.lower() or
                'Ministry/State Name' in txt or 'Department Name' in txt or 'Organisation Name' in txt or
                (field == 'table_qa_pair' and any(k in (d.metadata.get('question','') or '').lower() for k in ('ministry','department','organisation','organization')))
            ):
                cands.append(d)
            # Table-row specific labels
            if d.metadata.get('extraction_type') == 'table_row':
                low = txt.lower()
                if wants_office and ('office name' in low or 'office:' in low):
                    cands.append(d)
                if wants_email and ('email' in low):
                    cands.append(d)
                if wants_quantity and ('total quantity' in low or re.search(r'(?:^|\b)quantity\b', low)):
                    cands.append(d)
                if wants_reporting and ('reporting' in low and ('officer' in low or 'reporting/officer' in low)):
                    cands.append(d)

        if not cands:
            return docs

        # Prioritize based on query intent
        def pscore(d: Document) -> int:
            t = (d.page_content or '')
            s = 0
            if 'bid end' in ql or 'closing' in ql:
                s += 2 if 'Bid End Date/Time' in t else 0
            if 'opening' in ql:
                s += 2 if 'Bid Opening Date/Time' in t else 0
            if 'validity' in ql:
                s += 2 if 'Bid Offer Validity' in t else 0
            if wants_org:
                s += 2 if ('Ministry/State Name' in t or 'Department Name' in t or 'Organisation Name' in t) else 0
            if wants_office:
                s += 2 if ('Office Name' in t or 'office name' in t) else 0
            if wants_email:
                s += 2 if ('Email' in t or 'email' in t) else 0
            if wants_quantity:
                s += 2 if ('Total Quantity' in t or 'quantity' in t) else 0
            if wants_reporting:
                s += 2 if ('Reporting' in t and ('Officer' in t or 'reporting/ officer' in t)) else 0
            s += 1 if d.metadata.get('extraction_type') == 'structured' else 0
            return s

        cands = sorted({id(d): d for d in cands}.values(), key=pscore, reverse=True)

        # Inject unique candidates at the front
        seen_ids = set(id(x) for x in cands)
        tail = [d for d in docs if id(d) not in seen_ids]
        return cands + tail

    def search(self, query: str, pdf_id: Optional[str] = None, k: int = 12, prefer_structured: bool = True) -> List[Document]:
        if not self.db:
            return []

        # Use MMR for diversity in initial fetch
        try:
            # If a pdf_id is provided, fetch more to increase odds of scoped hits
            fetch_k = max(k * (8 if pdf_id else 3), 64 if pdf_id else 20)
            raw = self.db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        except Exception:
            raw = self.db.similarity_search(query, k=k)

        if not raw:
            return []

        # Scoped by pdf_id when provided
        if pdf_id:
            scoped = [d for d in raw if d.metadata.get('pdf_id') == pdf_id]
            if not scoped:
                scoped = [d for d in raw if pdf_id in str(d.metadata.get('source', ''))]
            docs = scoped
        else:
            # No scope: choose dominant document cluster
            chosen = self._dominant_doc(raw)
            docs = [d for d in raw if d.metadata.get('pdf_id') == chosen] if chosen else raw

        # Fallbacks when scoping yields nothing: try again and finally rank all docs from target pdf
        if pdf_id and not docs:
            try:
                # Retry with a much larger fetch_k
                raw2 = self.db.max_marginal_relevance_search(query, k=max(k, 12), fetch_k=500)
            except Exception:
                raw2 = self.db.similarity_search(query, k=200)
            scoped2 = [d for d in raw2 if d.metadata.get('pdf_id') == pdf_id] or [
                d for d in raw2 if pdf_id in str(d.metadata.get('source', ''))
            ]
            docs = scoped2

            # Final safety: rank documents directly from docstore for the given pdf_id
            if not docs and hasattr(self.db, 'docstore') and hasattr(self.db.docstore, '_dict'):
                all_docs = [d for d in self.db.docstore._dict.values() if isinstance(d, Document)]
                candidates = [d for d in all_docs if d.metadata.get('pdf_id') == pdf_id]
                if candidates:
                    # Lightweight scoring: prefer structured timing/QA, then token overlap
                    qtokens = self._expand_query_tokens(query)
                    def s(doc: Document) -> tuple:
                        field = str(doc.metadata.get('field', '')).lower()
                        et = doc.metadata.get('extraction_type') == 'structured'
                        text = (doc.page_content or '').lower()
                        overlap = len(qtokens & set(text.split()))
                        timing_boost = 2 if ('bid end' in text or 'bid opening' in text or 'offer validity' in text or field == 'timing') else 0
                        qa_boost = 1 if field in ('table_qa_pair',) else 0
                        return (1 if et else 0) + timing_boost + qa_boost, overlap
                    candidates.sort(key=s, reverse=True)
                    docs = candidates[:max(k, 8)]

        # Inject targeted structured timing docs when the query is precise
        docs = self._inject_targeted_structured(query, pdf_id, docs)

        if prefer_structured:
            docs = self._prefer_structured(docs)
            # Light rerank within the first few structured docs
            head = docs[:12]
            tail = docs[12:]
            head = self._rerank_by_query_overlap(query, head)
            docs = head + tail

        return docs[:8]
