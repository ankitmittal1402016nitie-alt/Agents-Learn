"""Ingest utilities for the `your-agent` app.

This module merges the more-complete ingestion utilities found at the repo root
with the local helper in `your-agent/`. It exposes the functions
``ingest_pdfs`` and ``ingest_url`` (used by ``app.py``) and provides a
lightweight, testable ingestion pipeline.

Design notes:
- Prefer the LangChain-friendly ``add_documents``/Chroma wrapper when available.
- Fall back to chromadb.PersistentClient usage when present.
- Provide helpers for PDF extraction, sample PDF creation, and a small
    deterministic embedding stub if no real provider is available.
"""

from __future__ import annotations

import os
import sys
import time
import hashlib
import logging
import traceback
from typing import List, Dict

from dotenv import load_dotenv

# Load local env (silently safe; .env is gitignored)
load_dotenv()

LOG_FILE = os.path.join(os.path.dirname(__file__), "ingest.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ingest")

# Paths
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
DB_DIR = os.path.join(ROOT, "db")
WEB_DB_DIR = os.path.join(ROOT, "webdb")

# Try optional imports
Chroma = None
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    Chroma = None

try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# chromadb Client usage (optional)
chromadb = None
Settings = None
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

# pypdf for reliable PDF extraction
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _sha_for(source: str, page: int, text: str) -> str:
    h = hashlib.sha1()
    key = f"{source}|{page}|{_normalize(text)}"
    h.update(key.encode("utf-8"))
    return h.hexdigest()


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    if not text:
        return []
    text = _normalize(text)
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = end - overlap
    return chunks


def _get_embed_dim(provider) -> int:
    """Return embedding vector length for the provider (best-effort).

    Tries provider.embed_query, provider.embed_documents, or provider.embed_texts
    with a tiny sample input. If that fails, returns a sensible default (1536).
    This helper is module-level so other functions can reuse it.
    """
    fallback_dim = int(os.environ.get("EMBEDDING_DIM_FALLBACK", "1536"))
    try:
        if hasattr(provider, 'embed_query'):
            v = provider.embed_query('test')
            return len(v) if v else fallback_dim
        if hasattr(provider, 'embed_documents'):
            v = provider.embed_documents(['test'])
            return len(v[0]) if v and isinstance(v, list) and v[0] else fallback_dim
        if hasattr(provider, 'embed_texts'):
            v = provider.embed_texts(['test'])
            return len(v[0]) if v and isinstance(v, list) and v[0] else fallback_dim
    except Exception:
        log.debug(f"_get_embed_dim: failed to probe embedding provider; using fallback {fallback_dim}")
    return fallback_dim


def _get_embed_model(provider) -> str:
    """Try to determine a short model identifier from the provider or env vars.

    Falls back to environment variables (GEMINI_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL)
    or a generic 'unknown' tag. The returned string is sanitized for use in collection names.
    """
    # Try common attributes exposed by embedding wrappers
    try:
        if hasattr(provider, 'model'):
            return str(provider.model).replace('/', '_')
        if hasattr(provider, 'model_name'):
            return str(provider.model_name).replace('/', '_')
        if hasattr(provider, '_model'):
            return str(provider._model).replace('/', '_')
    except Exception:
        pass

    # Fall back to environment variables commonly used
    for key in ("GEMINI_EMBEDDING_MODEL", "GEMINI_MODEL_EMBED", "OPENAI_EMBEDDING_MODEL", "OPENAI_MODEL_EMBED"):
        v = os.environ.get(key)
        if v:
            return v.replace('/', '_')

    return "unknown"


def _get_embeddings_provider():
    try:
        from src.gemini import GeminiEmbeddings
        log.info("ingest: using src.gemini.GeminiEmbeddings for embeddings")
        return GeminiEmbeddings()
    except Exception:
        log.debug("ingest: src.gemini.GeminiEmbeddings not available")

    try:
        from langchain_openai import OpenAIEmbeddings
        log.info("ingest: using OpenAIEmbeddings for embeddings")
        return OpenAIEmbeddings()
    except Exception:
        log.debug("ingest: OpenAIEmbeddings not available")

    class StubEmbeds:
        def embed_documents(self, texts: List[str]):
            return [[0.0] * 1536 for _ in texts]

        def embed_query(self, text: str):
            return [0.0] * 1536

    log.warning("ingest: falling back to StubEmbeds (no real embeddings available)")
    return StubEmbeds()


def _init_vectorstore(persist_directory: str, embedding_provider) -> object:
    if Chroma is None:
        raise RuntimeError("Chroma adapter not available (langchain_community.vectorstores.Chroma)")

    # _get_embed_dim is defined at module level so it can be reused elsewhere

    class EmbeddingWrapper:
        def __init__(self, provider):
            self._provider = provider

        def __call__(self, texts: List[str]):
            if hasattr(self._provider, 'embed_documents'):
                return self._provider.embed_documents(texts)
            elif hasattr(self._provider, 'embed_texts'):
                return self._provider.embed_texts(texts)
            else:
                raise RuntimeError("Provider does not support embed_documents/embed_texts")

        def embed_documents(self, texts: List[str]):
            return self(texts)

        def embed_query(self, text: str):
            if hasattr(self._provider, 'embed_query'):
                return self._provider.embed_query(text)
            res = self([text])
            return res[0] if res else []

    ew = EmbeddingWrapper(embedding_provider)
    # Namespace collection by embedding dimension to avoid chroma dimension mismatch errors
    try:
        model = _get_embed_model(embedding_provider)
        dim = _get_embed_dim(embedding_provider)
    except Exception:
        model = "unknown"
        dim = 1536
    # use model name + dim so different providers/models won't conflict
    collection_name = f"conversational_docs_{model}_{dim}"
    vs = Chroma(persist_directory=persist_directory, embedding_function=ew, collection_name=collection_name)
    return vs


def ingest_pdfs(chunk_size: int = 1000, overlap: int = 150) -> int:
    """Scan `data/` for PDFs and ingest them into the local Chroma DB at `db/`.

    Returns: number of chunks indexed.
    """
    embedding_provider = _get_embeddings_provider()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    pdfs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if not pdfs:
        log.info("ingest: no PDFs found in %s", DATA_DIR)
        return 0

    try:
        vs = _init_vectorstore(DB_DIR, embedding_provider)
    except Exception as e:
        log.error("ingest: failed to initialize vectorstore: %s", e)
        raise

    total_chunks = 0
    docs_to_add = []
    ids = []
    metadatas = []

    for pdf in pdfs:
        source = os.path.relpath(pdf)
        try:
            # try PyPDF2/pypdf first
            pages = []
            if PdfReader is not None:
                try:
                    reader = PdfReader(pdf)
                    pages = [p.extract_text() or "" for p in reader.pages]
                except Exception:
                    pages = []
            if not pages:
                # fallback to pymupdf if available
                try:
                    import fitz
                    doc = fitz.open(pdf)
                    pages = [doc.load_page(i).get_text() for i in range(doc.page_count)]
                except Exception:
                    log.exception("ingest: failed to extract text from %s", pdf)
                    pages = []

            for i, page_text in enumerate(pages, start=1):
                chunks = _chunk_text(page_text or "", chunk_size=chunk_size, overlap=overlap)
                for chunk in chunks:
                    sha = _sha_for(source, i, chunk)
                    meta = {
                        "source": source,
                        "page": i,
                        "type": "pdf",
                        "title": os.path.basename(source),
                        "sha": sha,
                        "ingested_at": int(time.time())
                    }
                    docs_to_add.append(Document(page_content=chunk, metadata=meta))
                    ids.append(sha)
                    metadatas.append(meta)
            total_chunks += sum(len(_chunk_text(p or "", chunk_size, overlap)) for p in pages)
        except Exception:
            log.exception("ingest: exception processing %s", pdf)

    # Upsert into Chroma (LangChain wrapper) if supported
    try:
        if hasattr(vs, 'add_documents'):
            vs.add_documents(docs_to_add, ids=ids)
        elif hasattr(vs, 'add_texts'):
            texts = [d.page_content for d in docs_to_add]
            vs.add_texts(texts, metadatas=metadatas, ids=ids)
        else:
            raise RuntimeError("Chroma adapter doesn't support add_documents/add_texts")
        if hasattr(vs, 'persist'):
            try:
                vs.persist()
            except Exception:
                log.debug("ingest: vs.persist() not supported or failed; continuing")
    except Exception:
        log.exception("ingest: failed to upsert documents into vectorstore")
        raise

    log.info("ingest: indexed %d chunks from %d PDFs", total_chunks, len(pdfs))
    return total_chunks


def ingest_url(url: str, chunk_size: int = 1200, overlap: int = 100) -> int:
    """Fetch a URL, extract main readable text, chunk, and add to webdb (with TTL metadata)."""
    embedding_provider = _get_embeddings_provider()
    os.makedirs(WEB_DB_DIR, exist_ok=True)

    try:
        vs = _init_vectorstore(WEB_DB_DIR, embedding_provider)
    except Exception as e:
        log.error("ingest_url: failed to init vectorstore: %s", e)
        raise

    try:
        import requests
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        log.exception("ingest_url: failed to fetch %s", url)
        raise

    text = None
    try:
        from readability import Document as ReadabilityDocument
        rd = ReadabilityDocument(html)
        text = rd.summary()
    except Exception:
        import re
        text = re.sub('<[^<]+?>', ' ', html)

    text = _normalize(text or "")
    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    docs = []
    ids = []
    now = int(time.time())
    for idx, chunk in enumerate(chunks, start=1):
        sha = _sha_for(url, idx, chunk)
        meta = {
            "source": url,
            "page": idx,
            "type": "web",
            "title": url,
            "sha": sha,
            "captured_at": now,
            "ttl_seconds": 7 * 24 * 3600,
        }
        docs.append(Document(page_content=chunk, metadata=meta))
        ids.append(sha)

    try:
        if hasattr(vs, 'add_documents'):
            vs.add_documents(docs, ids=ids)
        elif hasattr(vs, 'add_texts'):
            vs.add_texts([d.page_content for d in docs], metadatas=[d.metadata for d in docs], ids=ids)
        if hasattr(vs, 'persist'):
            try:
                vs.persist()
            except Exception:
                log.debug("ingest_url: vs.persist() not supported")
    except Exception:
        log.exception("ingest_url: failed to upsert web documents")
        raise

    log.info("ingest_url: indexed %d chunks from %s", len(chunks), url)
    return len(chunks)


# --- Additional helpers carried over from modular pipeline ---
def create_sample_pdf(path: str) -> None:
    try:
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(path)
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, "Sample PDF generated for ingestion tests.")
        c.drawString(72, 700, "This file is safe to delete.")
        c.save()
        return
    except Exception:
        pass

    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /ProcSet [/PDF /Text] /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n20 180 Td\n(Hello PDF) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000120 00000 n \n0000000240 00000 n \n0000000320 00000 n \ntrailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n400\n%%EOF\n"
    try:
        with open(path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to write sample PDF to {path}: {e}")


def create_sample_pdf_if_needed() -> str:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if pdfs:
        return ""
    sample_path = os.path.join(DATA_DIR, "sample_generated.pdf")
    create_sample_pdf(sample_path)
    return sample_path


def list_pdfs() -> List[str]:
    if not os.path.exists(DATA_DIR):
        return []
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]


def extract_text_from_pdf(path: str) -> str:
    if PdfReader is None:
        log.error("pypdf not available; cannot extract text from PDFs")
        return ""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        log.exception("Error extracting text from %s: %s", path, e)
        return ""


def ingest_pipeline() -> int:
    try:
        log.info("Pipeline: starting ingestion pipeline")
        pdfs = list_pdfs()
        docs = []
        for p in pdfs:
            docs.append({"text": extract_text_from_pdf(p), "metadata": {"source": os.path.basename(p)}})

        website_urls = []
        # website_docs = load_websites(website_urls)  # left as extension point
        # docs.extend(website_docs)

        if not docs:
            log.warning("Pipeline: no documents found under data/")
            return 0

        # Simple splitter
        chunks = []
        for d in docs:
            text = d.get("text", "")
            meta = d.get("metadata", {})
            for i in range(0, len(text), 1000):
                chunk = text[i:i+1000]
                if chunk.strip():
                    chunks.append({"text": chunk, "metadata": meta})

        # Prefer a chromadb-based embed+store if available
        if chromadb is not None:
            try:
                return _embed_and_store_chromadb(chunks, DB_DIR)
            except Exception:
                log.exception("chromadb storage failed; falling back to LangChain Chroma wrapper")

        # Fallback: use LangChain-style vs
        if Chroma is not None:
            emb = _get_embeddings_provider()
            vs = _init_vectorstore(DB_DIR, emb)
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            ids = [f"doc-{i}" for i in range(len(chunks))]
            if hasattr(vs, 'add_texts'):
                vs.add_texts(texts, metadatas=metadatas, ids=ids)
            elif hasattr(vs, 'add_documents'):
                docs_objs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
                vs.add_documents(docs_objs, ids=ids)
            if hasattr(vs, 'persist'):
                try:
                    vs.persist()
                except Exception:
                    pass
            return len(chunks)

        log.error("No storage backend available (chromadb or langchain Chroma)")
        return 0
    except Exception:
        log.exception("Ingestion pipeline failed")
        return 0


def _embed_and_store_chromadb(chunks: list, db_dir: str) -> int:
    if chromadb is None or Settings is None:
        raise RuntimeError("chromadb not available")
    try:
        client = chromadb.PersistentClient(path=db_dir)
    except Exception:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_dir))

    # choose collection name based on embedding dimension to ensure stored vectors
    # match the expected dimension for this embeddings provider
    try:
        emb_probe = None
        try:
            from src.gemini import GeminiEmbeddings
            emb_probe = GeminiEmbeddings()
        except Exception:
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    from langchain_openai import OpenAIEmbeddings
                    emb_probe = OpenAIEmbeddings()
                except Exception:
                    emb_probe = None
        model = _get_embed_model(emb_probe) if emb_probe is not None else "unknown"
        dim = _get_embed_dim(emb_probe) if emb_probe is not None else 1536
        col = client.get_or_create_collection(name=f"conversational_docs_{model}_{dim}", metadata={"hnsw:space": "cosine", "embed_model": model, "embed_dim": dim})
    except Exception:
        col = client.get_or_create_collection(name=f"conversational_docs_unknown_{1536}")

    embeddings = None
    try:
        from src.gemini import GeminiEmbeddings
        embeddings = GeminiEmbeddings()
    except Exception:
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
            except Exception:
                embeddings = None

    if embeddings is None:
        log.error("No embeddings provider available for chromadb storage")
        return 0

    docs = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"doc-{i}" for i in range(len(chunks))]

    if hasattr(embeddings, 'embed_documents'):
        doc_embeddings = embeddings.embed_documents(docs)
    elif hasattr(embeddings, 'embed_texts'):
        doc_embeddings = embeddings.embed_texts(docs)
    else:
        raise RuntimeError("Embeddings provider does not expose embed_documents/embed_texts")

    col.add(documents=docs, embeddings=doc_embeddings, metadatas=metadatas, ids=ids)
    log.info("Stored %d chunks into chromadb collection", len(docs))
    return len(docs)


if __name__ == '__main__':
    log.info("Running ingest as script")
    n = ingest_pdfs()
    print(f"Indexed {n} chunks")
