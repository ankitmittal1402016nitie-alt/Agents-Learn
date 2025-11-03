"""Ingest helper for PDFs into Chroma DB using OpenAI embeddings.

This file contains minimal code to:
- scan the `data/` folder for PDF files
- extract text (placeholder — you can plug in PyPDF2, pdfminer, or pypdf)
- create or use a Chroma collection and add embeddings

Adjust the extraction and embedding code to your preferred libraries and credentials.
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure openai.error.Timeout exists for older langchain expectations
try:
    import openai as _openai
    # If the package doesn't expose an `error` attribute or Timeout class,
    # create lightweight stubs so langchain's retry logic can reference them.
    if not hasattr(_openai, "error"):
        class _OpenAIErrorStub:
            class Timeout(Exception):
                pass
            class APIError(Exception):
                pass
            class RateLimitError(Exception):
                pass
            class ServiceUnavailableError(Exception):
                pass
            class APIConnectionError(Exception):
                pass
        _openai.error = _OpenAIErrorStub
    else:
        # ensure Timeout exists on the submodule
        if not hasattr(_openai.error, "Timeout"):
            class Timeout(Exception):
                pass
            setattr(_openai.error, "Timeout", Timeout)
except Exception:
    # If import fails, we'll surface the missing package later when used
    pass

# Ensure the openai package exposes the `error` submodule as an attribute
# (some installs don't set package attributes for submodules which older
# versions of langchain expect when they reference `openai.error.Timeout`).
try:
    import importlib
    _openai_pkg = importlib.import_module("openai")
    try:
        _openai_error = importlib.import_module("openai.error")
    except Exception:
        _openai_error = None
    if _openai_error is not None and not hasattr(_openai_pkg, "error"):
        setattr(_openai_pkg, "error", _openai_error)
except Exception:
    # if openai isn't installed, we'll surface that later when needed
    pass


# === Imports and Setup ===
import logging
import sys
import traceback
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"ChromaDB import error: {e}. Install with: pip install chromadb")
    raise
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    print(f"langchain-openai import error: {e}. Install with: pip install langchain-openai")
    OpenAIEmbeddings = None
try:
    from langchain_community.vectorstores import Chroma as LCChroma
except Exception:
    LCChroma = None
try:
    from pypdf import PdfReader
except ImportError as e:
    print(f"PyPDF import error: {e}. Install with: pip install pypdf")
    raise

# Optional: Add your website loader imports here (e.g., requests, BeautifulSoup)

# === Logging Setup ===
LOG_FILE = os.path.join(os.path.dirname(__file__), "ingest.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ingest")


# === Configurable Paths ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "db")

# === Modularized Ingestion Pipeline ===

def load_pdfs(data_dir: str) -> list:
    """Load PDF file paths from the data directory."""
    try:
        logger.info("PDFLoader: scanning directory %s. Purpose: find PDF files for ingestion.", data_dir)
        if not os.path.exists(data_dir):
            logger.warning("PDFLoader: directory %s not found. Next step: create directory or verify path.", data_dir)
            return []
        pdfs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        logger.info("PDFLoader: found %d PDF(s). Next step: extract text from each file.", len(pdfs))
        return pdfs
    except Exception as e:
        logger.error("PDFLoader: failed to scan directory. Error: %s. Next step: check directory permissions.", str(e))
        return []

def load_websites(urls: list) -> list:
    """Stub for loading website content. Extend as needed."""
    # Example: Use requests/BeautifulSoup to fetch and parse web pages
    logger.info("Website loading not implemented. Returning empty list.")
    return []

def split_documents(docs: list, chunk_size: int = 1000) -> list:
    """Split documents into smaller chunks for embedding (simple splitter)."""
    chunks = []
    for doc in docs:
        text = doc["text"]
        meta = doc["metadata"]
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append({"text": chunk, "metadata": meta})
    logger.info(f"Split into {len(chunks)} chunk(s).")
    return chunks

def embed_and_store(chunks: list, db_dir: str) -> int:
    """Embed document chunks and store them in ChromaDB.

    This function prefers the GeminiEmbeddings adapter (if available). If Gemini is
    not present, it will attempt to use OpenAIEmbeddings only when an OPENAI_API_KEY
    is available. This avoids startup-time pydantic validation errors when OpenAI
    credentials are not present.
    """
    try:
        logger.info("Storage: initializing ChromaDB. Purpose: store document vectors for search. Expected outcome: persistent vectorstore.")
        
        # Create Chroma client for version 1.0.x
        try:
            logger.info("Storage: creating Chroma client. Config: path=%s", db_dir)
            client = chromadb.PersistentClient(path=db_dir)
        except Exception as e:
            logger.error("Storage: failed to create PersistentClient: %s", str(e))
            raise
        # Get or create collection using v1.0.x API
        try:
            col = client.get_or_create_collection(
                name="conversational_docs",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity for better matching
            )
            logger.info("Storage: obtained collection 'conversational_docs'")
        except Exception as e:
            logger.error("Failed to get or create collection: %s", str(e))
            raise

        # Obtain embeddings provider (Gemini preferred)
        embeddings = None
        try:
            # Import the local adapter to avoid picking up any same-named package
            from src.gemini import GeminiEmbeddings
            embeddings = GeminiEmbeddings()
            logger.info("Using GeminiEmbeddings for embeddings in ingest pipeline")
        except Exception as e_gem:
            logger.debug(f"GeminiEmbeddings not available during ingest: {e_gem}")
            # Only attempt OpenAI if we have a key set to avoid validation errors
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings()
                    logger.info("Falling back to OpenAIEmbeddings for ingest")
                except Exception as e_oa:
                    logger.error(f"OpenAIEmbeddings instantiation failed: {e_oa}")
            else:
                logger.info("No OPENAI_API_KEY present; skipping OpenAIEmbeddings during ingest")

        if embeddings is None:
            logger.error("No embeddings provider available (install google-generativeai or set OPENAI_API_KEY)")
            return 0

        docs = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"doc-{i}" for i in range(len(chunks))]

        # Use the embeddings provider to embed documents. Many adapters provide
        # `embed_documents`; adapt if different method names are exposed.
        if hasattr(embeddings, 'embed_documents'):
            doc_embeddings = embeddings.embed_documents(docs)
        elif hasattr(embeddings, 'embed_texts'):
            doc_embeddings = embeddings.embed_texts(docs)
        else:
            raise RuntimeError("Embeddings provider does not expose embed_documents/embed_texts")

        # Add documents to collection using v1.0.x API
        try:
            # Add the documents with their embeddings
            col.add(
                documents=docs,
                embeddings=doc_embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Embedded and stored {len(docs)} chunk(s) in ChromaDB.")
            return len(docs)
        except Exception as e:
            logger.error(f"Failed to store documents in ChromaDB: {e}")
            raise
    except Exception as e:
        logger.error(f"Error during embedding/storage: {e}")
        return 0

def ingest_pipeline():
    """Full modular ingestion pipeline: load, (optionally websites), split, embed, store."""
    try:
        logger.info("Pipeline: starting document ingestion. Purpose: process documents for search. Expected outcome: indexed and searchable content.")
        
        # 1. Load PDFs
        logger.info("Pipeline: loading PDFs. Purpose: get content for processing.")
        pdf_paths = load_pdfs(DATA_DIR)
        docs = []
        for p in pdf_paths:
            logger.info("Pipeline: extracting text from %s. Purpose: convert PDF to processable text.", os.path.basename(p))
            docs.append({"text": extract_text_from_pdf(p), "metadata": {"source": os.path.basename(p)}})

        # 2. Optionally add websites (edit this list)
        logger.info("Pipeline: checking for website content. Purpose: include web sources if configured.")
        website_urls = []  # Add URLs here if needed
        website_docs = load_websites(website_urls)
        docs.extend(website_docs)

        if not docs:
            logger.warning("Pipeline: no documents found. Next step: add PDFs to data/ directory.")
            return 0

        # 3. Split
        logger.info("Pipeline: splitting documents. Purpose: create manageable chunks for embedding.")
        chunks = split_documents(docs)

        # 4. Embed + store
        logger.info("Pipeline: embedding and storing chunks. Purpose: create searchable vector database.")
        count = embed_and_store(chunks, DB_DIR)
        if count > 0:
            logger.info("Pipeline: completed successfully. Result: %d chunks indexed and searchable.", count)
        else:
            logger.warning("Pipeline: completed but no chunks stored. Next step: check embedding provider status.")
        return count
    except Exception as e:
        logger.error("Pipeline: failed with error. Error: %s\nStack trace:\n%s", str(e), traceback.format_exc())
        return 0


def create_sample_pdf(path: str) -> None:
    """Create a small valid PDF at `path`.

    Tries to use reportlab if available, otherwise writes a bundled minimal PDF bytes blob.
    """
    try:
        # Prefer reportlab if it's installed
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(path)
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, "Sample PDF generated for ingestion tests.")
        c.drawString(72, 700, "This file is safe to delete.")
        c.save()
        return
    except Exception:
        pass

    # Fallback: write a minimal PDF. Many PDF readers accept this simple structure.
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /ProcSet [/PDF /Text] /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n20 180 Td\n(Hello PDF) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000120 00000 n \n0000000240 00000 n \n0000000320 00000 n \ntrailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n400\n%%EOF\n"
    try:
        with open(path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to write sample PDF to {path}: {e}")


def create_sample_pdf_if_needed() -> str:
    """Create a sample PDF in DATA_DIR if no PDFs exist. Returns path to created file or empty string."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if pdfs:
        return ""
    sample_path = os.path.join(DATA_DIR, "sample_generated.pdf")
    create_sample_pdf(sample_path)
    return sample_path


def list_pdfs() -> List[str]:
    """Return list of PDF file paths in data/"""
    if not os.path.exists(DATA_DIR):
        return []
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pypdf"""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {path}: {e}")
        return f"Error processing {os.path.basename(path)}"




# For backward compatibility with app.py and CLI
def ingest_pdfs() -> int:
    """Legacy entrypoint for ingestion (calls modular pipeline)."""
    return ingest_pipeline()


def query_docs(query: str, top_k: int = 5) -> List[Dict]:
    """Query Chroma and return top_k results as a list of dicts {id, metadata, distance}

    Note: This uses cosine similarity via chromadb. Ensure the DB has been populated.
    """
    if chromadb is None:
        raise RuntimeError("chromadb not available — install required packages")

    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=DB_DIR))
    col = client.get_or_create_collection("conversational_docs")
    # If using embeddings client directly, embed the query then query the collection
    # initialize embeddings (if available)
    embeddings = None
    try:
        if OpenAIEmbeddings is not None:
            embeddings = OpenAIEmbeddings()
    except Exception:
        embeddings = None

    if embeddings is not None:
        try:
            # prefer embed_query, otherwise fall back
            if hasattr(embeddings, 'embed_query'):
                q_emb = embeddings.embed_query(query)
            else:
                q_emb = embeddings.embed_documents([query])[0]
            results = col.query(query_embeddings=[q_emb], n_results=top_k)
        except Exception:
            # fallback to text-based query if embedding call fails
            results = col.query(query_texts=[query], n_results=top_k)
    else:
        results = col.query(query_texts=[query], n_results=top_k)
    # results structure depends on chromadb version; normalize a safe format
    hits = []
    for ids, docs, metadatas, distances in zip(results.get('ids', []), results.get('documents', []), results.get('metadatas', []), results.get('distances', [])):
        for i in range(len(ids)):
            hits.append({
                'id': ids[i],
                'document': docs[i] if i < len(docs) else None,
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'distance': distances[i] if i < len(distances) else None,
            })
    return hits


if __name__ == "__main__":
    logger.info("Starting modular ingest pipeline...")
    try:
        count = ingest_pipeline()
        if count > 0:
            logger.info(f"Ingestion complete. Indexed {count} chunk(s). Success.")
            print("Success")
            sys.exit(0)
        else:
            logger.warning("No documents indexed. Check data/ directory and logs.")
            print("Need Attention")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}\n{traceback.format_exc()}")
        print("Need Attention")
        sys.exit(1)
