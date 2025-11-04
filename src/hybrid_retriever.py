from typing import List, Dict, Optional, Any
import re
import logging

try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

log = logging.getLogger("hybrid_retriever")

class HybridRetriever:
    """Combine Chroma vector similarity search with simple keyword/regex matching.

    This is a lightweight, dependency-free hybrid retriever suitable for the
    repository's upgrade plan. It accepts two Chroma "vectorstore" instances:
    one for primary docs (vs) and an optional web vectorstore (web_vs).
    """
    def __init__(self, vs: Any, web_vs: Optional[Any] = None, k: int = 4, score_weight: float = 0.7):
        self.vs = vs
        self.web_vs = web_vs
        self.k = k
        self.score_weight = score_weight  # weight for vector scores vs keyword scores

    def _keyword_score(self, text: str, query: str) -> float:
        # simple scoring: count occurrences of tokens
        q_tokens = re.findall(r"\w+", query.lower())
        if not q_tokens:
            return 0.0
        text_l = text.lower()
        score = 0
        for t in q_tokens:
            score += text_l.count(t)
        return float(score) / (len(q_tokens) + 1)

    def _gather_all_docs(self, which: str = "local") -> List[Document]:
        # Attempt to get all documents from the store. Adapter-specific; best-effort.
        try:
            data = None
            if which == "local":
                data = self.vs.get() if hasattr(self.vs, 'get') else None
            else:
                data = self.web_vs.get() if (self.web_vs and hasattr(self.web_vs, 'get')) else None

            if isinstance(data, dict) and "documents" in data:
                docs = []
                for i, txt in enumerate(data.get("documents", [])):
                    meta = data.get("metadatas", [])[i] if i < len(data.get("metadatas", [])) else {}
                    docs.append(Document(page_content=txt, metadata=meta))
                return docs

            # Some adapters return a list of Document-like objects
            if isinstance(data, list):
                converted = []
                for item in data:
                    if hasattr(item, 'page_content'):
                        converted.append(item)
                    elif isinstance(item, dict) and 'page_content' in item:
                        converted.append(Document(page_content=item.get('page_content', ''), metadata=item.get('metadata', {})))
                if converted:
                    return converted
        except Exception as e:
            log.debug("hybrid_retriever: primary get() call failed: %s", str(e))

        # Fallback: inspect underlying collection object if available (langchain adapter or chromadb)
        try:
            col = None
            if which == "local":
                col = getattr(self.vs, '_collection', None)
            else:
                col = getattr(self.web_vs, '_collection', None) if self.web_vs is not None else None

            if col and hasattr(col, 'get'):
                try:
                    raw = col.get()
                    if isinstance(raw, dict) and 'documents' in raw:
                        docs = []
                        for i, txt in enumerate(raw.get('documents', [])):
                            meta = raw.get('metadatas', [])[i] if i < len(raw.get('metadatas', [])) else {}
                            docs.append(Document(page_content=txt, metadata=meta))
                        return docs
                except Exception as ce:
                    log.debug("hybrid_retriever: underlying collection.get() failed: %s", str(ce))

            # As a last resort, try to access any stored 'documents' attribute
            if col and hasattr(col, 'documents'):
                docs_attr = getattr(col, 'documents')
                if isinstance(docs_attr, list) and docs_attr:
                    converted = []
                    for i, txt in enumerate(docs_attr):
                        meta = getattr(col, 'metadatas', [{}])[i] if hasattr(col, 'metadatas') else {}
                        converted.append(Document(page_content=txt, metadata=meta))
                    return converted
        except Exception as e:
            log.debug("hybrid_retriever: fallback collection inspection failed: %s", str(e))

        return []

    def get_relevant_documents(self, query: str, k: Optional[int] = None, fetch_k: Optional[int] = None, use_mmr: Optional[bool] = None, mmr_lambda: float = 0.5) -> List[Document]:
        """Retrieve relevant documents.

        Parameters:
        - query: user query
        - k: final number of documents to return
        - fetch_k: initial recall size (larger than k when using MMR)
        - use_mmr: whether to apply MMR diversification
        - mmr_lambda: not used in naive MMR but kept for API compatibility
        """
        if k is None:
            k = self.k
        if fetch_k is None:
            fetch_k = max(k, getattr(self, 'k', k))
        if use_mmr is None:
            use_mmr = True

        results = []
        vector_results = []
        try:
            # similarity_search_with_score may be supported by LangChain/adapter
            if hasattr(self.vs, 'similarity_search_with_score'):
                vector_results = self.vs.similarity_search_with_score(query, k=fetch_k)
            elif hasattr(self.vs, 'similarity_search'):
                docs = self.vs.similarity_search(query, k=fetch_k)
                vector_results = [(d, 1.0) for d in docs]
        except Exception as e:
            log.debug("hybrid_retriever: vector search failed: %s", str(e))

        log.debug("hybrid_retriever: vector_results count=%d", len(vector_results) if vector_results is not None else 0)

        # turn vector_results into dict doc->score
        doc_scores = {}
        for item in vector_results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
                score = 1.0
            doc_scores[getattr(doc, 'metadata', {}).get('sha', id(doc))] = (doc, float(score))

        # perform simple keyword search over all local docs
        keyword_docs = []
        all_docs = self._gather_all_docs(which="local")
        log.debug("hybrid_retriever: gathered all_docs count=%d", len(all_docs))
        for doc in all_docs:
            ks = self._keyword_score(doc.page_content, query)
            if ks > 0:
                keyword_docs.append((doc, ks))

        log.debug("hybrid_retriever: keyword_docs count=%d", len(keyword_docs))

        # include web docs if available
        if self.web_vs:
            try:
                web_vector_results = []
                if hasattr(self.web_vs, 'similarity_search_with_score'):
                    web_vector_results = self.web_vs.similarity_search_with_score(query, k=fetch_k)
                elif hasattr(self.web_vs, 'similarity_search'):
                    docs = self.web_vs.similarity_search(query, k=fetch_k)
                    web_vector_results = [(d, 1.0) for d in docs]
                for item in web_vector_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                    else:
                        doc = item
                        score = 1.0
                    key = getattr(doc, 'metadata', {}).get('sha', id(doc))
                    if key not in doc_scores:
                        doc_scores[key] = (doc, float(score) * 0.9)
                # keyword scan web
                web_docs = self._gather_all_docs(which="web")
                for doc in web_docs:
                    ks = self._keyword_score(doc.page_content, query)
                    if ks > 0 and getattr(doc, 'metadata', {}).get('sha') not in doc_scores:
                        keyword_docs.append((doc, ks))
            except Exception as e:
                log.debug("hybrid_retriever: web search failed: %s", str(e))

        # Merge keyword docs into doc_scores with lower weight
        for doc, ks in keyword_docs:
            key = getattr(doc, 'metadata', {}).get('sha', id(doc))
            if key in doc_scores:
                # combine
                existing = doc_scores[key]
                combined = existing[1] + (ks * (1 - self.score_weight))
                doc_scores[key] = (existing[0], combined)
            else:
                doc_scores[key] = (doc, ks * (1 - self.score_weight))

        log.debug("hybrid_retriever: doc_scores count=%d", len(doc_scores))

        # now select top candidates by score (we selected up to fetch_k earlier)
        scored = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        selected = [d for d, s in scored]

        # If MMR requested, apply naive MMR to diversify down to k
        if use_mmr and len(selected) > k:
            final = []
            final.append(selected[0])
            for cand in selected[1:]:
                if len(final) >= k:
                    break
                a = set(re.findall(r"\w+", final[0].page_content.lower()))
                b = set(re.findall(r"\w+", cand.page_content.lower()))
                if a:
                    overlap = len(a & b) / (len(a) + 1)
                    if overlap < 0.5:
                        final.append(cand)
            # fill if not enough
            if len(final) < k:
                for cand in selected:
                    if cand not in final:
                        final.append(cand)
                    if len(final) >= k:
                        break
            return final[:k]

        # no MMR: just return top-k by score
        return selected[:k]

    def get_relevant_documents_with_scores(self, query: str, k: Optional[int] = None, fetch_k: Optional[int] = None, use_mmr: Optional[bool] = None, mmr_lambda: float = 0.5) -> List[tuple]:
        """Return list of (Document, score) tuples for the query.

        This mirrors `get_relevant_documents` but preserves scores so callers
        can make numeric decisions (for gating web-augmentation, etc.).
        """
        if k is None:
            k = self.k
        if fetch_k is None:
            fetch_k = max(k, getattr(self, 'k', k))
        if use_mmr is None:
            use_mmr = True

        vector_results = []
        try:
            if hasattr(self.vs, 'similarity_search_with_score'):
                vector_results = self.vs.similarity_search_with_score(query, k=fetch_k)
            elif hasattr(self.vs, 'similarity_search'):
                docs = self.vs.similarity_search(query, k=fetch_k)
                vector_results = [(d, 1.0) for d in docs]
        except Exception as e:
            log.debug("hybrid_retriever: vector search failed: %s", str(e))

        doc_scores = {}
        for item in vector_results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
                score = 1.0
            key = getattr(doc, 'metadata', {}).get('sha', id(doc))
            doc_scores[key] = (doc, float(score))

        # Merge keyword matches as lower-weight scores
        keyword_docs = []
        all_docs = self._gather_all_docs(which="local")
        for doc in all_docs:
            ks = self._keyword_score(doc.page_content, query)
            if ks > 0:
                keyword_docs.append((doc, ks))

        if self.web_vs:
            try:
                web_vector_results = []
                if hasattr(self.web_vs, 'similarity_search_with_score'):
                    web_vector_results = self.web_vs.similarity_search_with_score(query, k=fetch_k)
                elif hasattr(self.web_vs, 'similarity_search'):
                    docs = self.web_vs.similarity_search(query, k=fetch_k)
                    web_vector_results = [(d, 1.0) for d in docs]
                for item in web_vector_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                    else:
                        doc = item
                        score = 1.0
                    key = getattr(doc, 'metadata', {}).get('sha', id(doc))
                    if key not in doc_scores:
                        doc_scores[key] = (doc, float(score) * 0.9)
                web_docs = self._gather_all_docs(which="web")
                for doc in web_docs:
                    ks = self._keyword_score(doc.page_content, query)
                    if ks > 0 and getattr(doc, 'metadata', {}).get('sha') not in doc_scores:
                        keyword_docs.append((doc, ks))
            except Exception as e:
                log.debug("hybrid_retriever: web search failed: %s", str(e))

        for doc, ks in keyword_docs:
            key = getattr(doc, 'metadata', {}).get('sha', id(doc))
            if key in doc_scores:
                existing = doc_scores[key]
                combined = existing[1] + (ks * (1 - self.score_weight))
                doc_scores[key] = (existing[0], combined)
            else:
                doc_scores[key] = (doc, ks * (1 - self.score_weight))

        # Sort by numeric score descending
        scored = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        # Return tuples (doc, score)
        return [(d, s) for d, s in scored[:fetch_k]]