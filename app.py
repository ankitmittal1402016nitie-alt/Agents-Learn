"""FastAPI front-end for multi-user conversational document Q&A.

Endpoints:
- POST /query  -> Ask a question over the indexed documents (returns top matches + session_id)
- POST /ingest -> triggers ingestion of files via ingest.ingest_pdfs (optional)
- POST /clear_history -> clears a session's chat history
- GET  /      -> health check
- GET  /provider_status -> get current provider status (live API or debug stubs)

This app keeps lightweight in-memory chat histories per session_id for multi-user behavior.
For production, replace the in-memory store with a persistent cache (Redis, DB) and add auth.
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Do not require OpenAI key when using Gemini; optional environment keys:
# - GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS for Google Gemini
import uuid
import threading
import traceback
from typing import Optional, List, Dict, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import UploadFile, File, Request
from fastapi.responses import HTMLResponse, Response

# Optional imports: try to import provider-specific helpers but don't fail at import time.
# We will handle missing providers at runtime and produce helpful log messages.
Chroma = None
GeminiEmbeddings = None
GeminiChat = None
ChatMessageHistory = None
ExternalSearch = None
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    Chroma = None
try:
    # src.gemini is our local adapter; prefer importing it so we use the project
    # copy rather than any same-named package installed in the environment.
    from src.gemini import GeminiEmbeddings, GeminiChat
except Exception:
    GeminiEmbeddings = None
    GeminiChat = None
try:
    from src.external_search import SerpAPIExternalSearch, StubExternalSearch
except Exception:
    SerpAPIExternalSearch = None
    StubExternalSearch = None
try:
    # ChatMessageHistory comes from LangChain memory; if langchain isn't available
    # provide a tiny shim so the app can still manage simple histories.
    from langchain.memory import ChatMessageHistory
except Exception:
    # Fallback shim: minimal API used by this app
    class ChatMessage:
        def __init__(self, _type, content):
            self.type = _type
            self.content = content

    class ChatMessageHistory:
        def __init__(self):
            self.messages = []
        def add_user_message(self, text: str):
            self.messages.append(ChatMessage('user', text))
        def add_ai_message(self, text: str):
            self.messages.append(ChatMessage('ai', text))

from src.logger import logger
from src.provider_status import app_status, ProviderMode, ProviderStatus

# Configuration
DB_DIR = os.path.join(os.path.dirname(__file__), "db")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4-mini")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))

# Global components (initialized at startup)
_retriever = None
_llm = None
_qa = None
_embed_fn = None
_external_search = None  # External search provider for web knowledge
_external_search = None
_embed_fn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embeddings, vectorstore, LLM and QA chain once on startup.
    This will run when the FastAPI app starts and yield control back to the server.
    """
    global _retriever, _llm, _qa, _embed_fn, _external_search
    try:
        logger.info("Startup: beginning app initialization. Purpose: set up providers and services. Expected outcome: ready-to-use embeddings, vectorstore, LLM, and external search.")
        
        # Initialize external search provider
        try:
            from src.external_search import get_search_provider
            _external_search = get_search_provider()
            logger.info("External search provider initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize external search: %s. Web search augmentation will be disabled.", str(e))
            _external_search = None
        
        # Provider selection logic (preferred order): Gemini -> OpenAI -> disabled
        embeddings = None
        try:
            # Try Gemini first (needs `google-generativeai` and credentials)
            logger.info("Embeddings: attempting to initialize Gemini embeddings. Purpose: provide vector embeddings for document indexing and retrieval.")
            embeddings = GeminiEmbeddings()
            app_status.embeddings_status = ProviderStatus(
                mode=ProviderMode.LIVE_API if not os.environ.get("DEBUG") else ProviderMode.DEBUG_STUB,
                provider_name="Gemini"
            )
            logger.info("Embeddings: GeminiEmbeddings initialized successfully. Next step: initialize vectorstore.")
        except Exception as e_gem:
            app_status.embeddings_status = ProviderStatus(
                mode=ProviderMode.DISABLED,
                provider_name="Gemini",
                error_details=str(e_gem)
            )
            logger.debug(f"GeminiEmbeddings not available: {e_gem}")
            # Fall back to OpenAI embeddings only if an API key is present
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings()
                    app_status.embeddings_status = ProviderStatus(
                        mode=ProviderMode.LIVE_API,
                        provider_name="OpenAI"
                    )
                    logger.info("Falling back to OpenAIEmbeddings (OPENAI_API_KEY detected)")
                except Exception as e_oa:
                    app_status.embeddings_status = ProviderStatus(
                        mode=ProviderMode.DISABLED,
                        provider_name="OpenAI",
                        error_details=str(e_oa)
                    )
                    logger.debug(f"OpenAIEmbeddings instantiation failed: {e_oa}")
            else:
                logger.info("No OPENAI_API_KEY present; skipping OpenAIEmbeddings instantiation")

        # If we couldn't obtain an embeddings provider, skip vectorstore init
        if embeddings is None:
            logger.warning("No embeddings provider available at startup. Vectorstore will not be initialized.")
            yield
            return

        # If the DB directory appears empty/non-existent, attempt a local ingest to populate it.
        if not os.path.exists(DB_DIR):
            logger.info("Database: directory %s not found. Purpose: initial data ingestion. Next step: attempt to ingest PDFs.", DB_DIR)
            try:
                from ingest import ingest_pdfs
                count = ingest_pdfs()
                if count > 0:
                    logger.info("Database: ingestion successful. Result: %d chunks indexed and ready for search.", count)
                else:
                    logger.warning("Database: no documents ingested. Next step: add PDFs to data/ directory before querying.")
                    yield
                    return
            except Exception as e:
                logger.error(f"Failed to run ingest: {e}\n{traceback.format_exc()}")
                yield
                return

        # chromadb / langchain-chroma: some versions accept an embedding function, others expect
        # a full embeddings object. Provide a wrapper object that exposes the methods
        # `embed_documents` and `embed_query` (and is also callable) so Chroma works
        # consistently regardless of the underlying provider implementation.
        class EmbeddingWrapper:
            def __init__(self, provider):
                self._provider = provider

            def __call__(self, texts: list):
                # Accepts list[str] -> list[vector]
                if hasattr(self._provider, 'embed_documents'):
                    return self._provider.embed_documents(texts)
                elif hasattr(self._provider, 'embed_texts'):
                    return self._provider.embed_texts(texts)
                else:
                    raise RuntimeError("Embeddings provider does not expose embed_documents/embed_texts")

            def embed_documents(self, texts: list):
                return self(texts)

            def embed_query(self, text: str):
                # Prefer a dedicated embed_query if available
                if hasattr(self._provider, 'embed_query'):
                    return self._provider.embed_query(text)
                # Fall back to embedding the single item and return first vector
                res = self([text])
                return res[0] if res else []

        _embed_fn = EmbeddingWrapper(embeddings)

        # Create Chroma vectorstore from LangChain community adapter (if available)
        try:
            logger.info("Vectorstore: initializing Chroma. Purpose: connect to vector database. Config: directory=%s", DB_DIR)
            # Ensure we open the same collection name that ingestion used so both
            # codepaths operate on the same stored vectors.
            collection_name = "conversational_docs"
            vs = Chroma(
                persist_directory=DB_DIR,
                embedding_function=_embed_fn,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"},  # Match the space used in storage
            )
            # Check collection size
            try:
                col_res = vs.get() if hasattr(vs, "get") else {}
                col_size = len(col_res.get("ids", [])) if isinstance(col_res, dict) else "unknown"
                logger.info("Vectorstore: connected to database. Status: found %s stored vectors. Purpose: verify data availability.", col_size)
                # If LangChain adapter reports zero but ingest reported stored vectors,
                # inspect the raw Chroma DB directly to help debug mismatches.
                if col_size == 0:
                    try:
                        import chromadb
                        from chromadb.config import Settings
                        pc = chromadb.PersistentClient(path=DB_DIR)
                        try:
                            raw_col = pc.get_collection(name=collection_name)
                            raw_ids = raw_col.count() if hasattr(raw_col, 'count') else None
                            logger.info("Raw ChromaDB: collection '%s' count=%s (inspected directly via PersistentClient).", collection_name, raw_ids)
                        except Exception as _e:
                            logger.debug("Raw ChromaDB inspection failed: %s", str(_e))
                    except Exception:
                        # best-effort only; don't fail startup because of this debug check
                        logger.debug("Raw ChromaDB client not available for deeper inspection.")
            except Exception as e:
                logger.warning("Vectorstore: could not check collection size. Error: %s. Next step: verify database state.", str(e))
            
            _retriever = vs.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.5  # Add similarity threshold
                }
            )
            logger.info("Vectorstore: created retriever. Config: k=4 docs per query, similarity search. Next step: ready for search.")
        except Exception as e_chroma:
            logger.error("Vectorstore: initialization failed. Error: %s\nStack trace:\n%s", str(e_chroma), traceback.format_exc())
            yield
            return

        # Use Gemini chat LLM wrapper when available; fall back to a simple wrapper that raises
        try:
            # Get models from environment with fallbacks
            primary_model = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
            fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
            
            logger.info("LLM: initializing GeminiChat with primary model. Config: model=%s, temperature=%s", 
                       primary_model, temperature)
            try:
                _llm = GeminiChat(model=primary_model, temperature=temperature)
            except Exception as e:
                logger.warning(f"Failed to initialize primary model {primary_model}, trying fallback {fallback_model}: {str(e)}")
                try:
                    _llm = GeminiChat(model=fallback_model, temperature=temperature)
                except Exception as e2:
                    logger.error(f"Both primary and fallback models failed. Primary error: {str(e)}, Fallback error: {str(e2)}")
                    raise GeminiError(f"Failed to initialize both primary model ({primary_model}) and fallback model ({fallback_model})")
            
            app_status.llm_status = ProviderStatus(
                mode=ProviderMode.LIVE_API if not os.environ.get("DEBUG") else ProviderMode.DEBUG_STUB,
                provider_name="Gemini"
            )
            logger.info("LLM: GeminiChat initialized successfully. Next step: create QA wrapper.")
        except Exception as e_llm:
            app_status.llm_status = ProviderStatus(
                mode=ProviderMode.DISABLED,
                provider_name="Gemini",
                error_details=str(e_llm)
            )
            logger.debug("GeminiChat not available; LLM calls will use a simple raise until configured.")
            class _LLMStub:
                def __call__(self, *args, **kwargs):
                    raise RuntimeError("No LLM provider configured (Gemini or OpenAI).")
            _llm = _LLMStub()

        # Keep a minimal callable wrapper for answering queries. This is intentionally simple
        # so it remains compatible with several LLM wrappers; it accepts (question, chat_history).
        def _qa_call(question: str, chat_history: list, retrieved_docs: list = None):
            """Prepare a combined context from chat history, retrieved documents, and external search, then call the LLM.

            Inputs:
            - question: the user question (already contains system/style hints)
            - chat_history: list of (role, text) tuples
            - retrieved_docs: list of LangChain Documents returned by the retriever
            """
            global _external_search
            
            # Build conversation history text
            history_text = "\n".join([f"{t}: {c}" for t, c in chat_history]) if chat_history else ""

            # Build retrieved context block: include a short snippet and source metadata for each doc
            docs_text = []
            if retrieved_docs:
                for i, d in enumerate(retrieved_docs[:5], start=1):
                    meta = getattr(d, "metadata", {})
                    src = meta.get("source", "")
                    page = meta.get("page", "")
                    snippet = (d.page_content[:800] + ("..." if len(d.page_content) > 800 else "")) if getattr(d, "page_content", None) else ""
                    docs_text.append(f"[DOC {i}] source={src} page={page}\n{snippet}")
            retrieved_block = "\n\n".join(docs_text) if docs_text else ""

            # Get external search results if available
            external_text = []
            if _external_search:
                try:
                    # Extract the core question without style hints and system instructions
                    core_q = question.split("Question: ")[-1] if "Question: " in question else question
                    results = _external_search.search(core_q, num_results=3)
                    for i, result in enumerate(results, start=1):
                        external_text.append(
                            f"[WEB {i}] title={result['title']} source={result['link']}\n{result['snippet']}"
                        )
                except Exception as e:
                    logger.warning("External search failed: %s. Continuing with document context only.", str(e))

            external_block = "\n\n".join(external_text) if external_text else ""

            # Compose final context with explicit instructions for the LLM
            context_parts = []
            
            if history_text:
                context_parts.append("Conversation history:\n" + history_text)
            
            if retrieved_block:
                context_parts.append(
                    "Retrieved document context (PRIMARY SOURCE - prefer this information):\n" + 
                    retrieved_block
                )
            else:
                context_parts.append("No relevant documents found in the primary source.")
                
            if external_block:
                context_parts.append(
                    "Additional web search results (use to supplement document knowledge):\n" +
                    external_block
                )

            # Add explicit instructions for the model
            context_parts.append(
                "Instructions for using sources:\n"
                "1. Primarily use information from the retrieved documents (marked as [DOC X]).\n"
                "2. If the documents don't fully address the question, supplement with web search results (marked as [WEB X]).\n"
                "3. Clearly indicate when you're using information from documents vs web sources.\n"
                "4. If contradictions exist between sources, prefer document information and note the discrepancy."
            )

            context = "\n\n".join(context_parts)

            # Call the LLM with the enriched context
            try:
                return _llm(prompt=question, context=context)
            except TypeError:
                # Fallback: some LLM wrappers expect a single string prompt - prepend the context
                combined = context + "\n\n" + question
                return _llm(combined)

        _qa = _qa_call

        logger.info("Startup initialization complete")
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}\n{traceback.format_exc()}")
    # Let the application run
    yield
    # Cleanup (if needed)
    logger.info("Shutting down")

# App
app = FastAPI(
    title="Agentic AI: Conversational Docs (FastAPI)",
    lifespan=lifespan,
)

# In-memory session histories: session_id -> ChatMessageHistory
_histories: Dict[str, ChatMessageHistory] = {}
_hist_lock = threading.Lock()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    style: Optional[str] = "summary"  # or 'detailed'
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict]





def _get_or_create_history(session_id: Optional[str]) -> Tuple[str, ChatMessageHistory]:
    """Return (session_id, ChatMessageHistory). Creates a new session if session_id is None or not found."""
    with _hist_lock:
        if session_id and session_id in _histories:
            logger.info("History: reusing existing chat history for session %s. Purpose: maintain conversation context.", session_id)
            return session_id, _histories[session_id]
        # create new session
        new_id = session_id or str(uuid.uuid4())
        hist = ChatMessageHistory()
        _histories[new_id] = hist
        logger.info("History: created new session %s. Purpose: track new conversation. Expected outcome: empty history ready for chat.", new_id)
        return new_id, hist


@app.get("/")
async def root():
    return {"status": "ok", "message": "Agentic AI: Conversational Docs (FastAPI) - running"}


@app.get("/provider_status")
async def provider_status():
    """Return current status of providers (embeddings and LLM).
    
    Shows whether each provider is using live API, debug stubs, or is disabled,
    along with any error details from initialization.
    """
    return app_status.to_dict()


@app.get('/favicon.ico')
async def favicon():
    # Return no content for favicon to silence 404s in dev
    return Response(status_code=204)


@app.post("/ingest")
async def ingest_endpoint():
    """Trigger ingestion (if `ingest.ingest_pdfs` is available in the repo)."""
    logger.info("Ingest: starting manual ingestion. Purpose: index all PDFs in data directory for search.")
    try:
        from ingest import ingest_pdfs
    except Exception:
        logger.error("Ingest: module not available. Next step: ensure ingest.py exists and is importable.")
        raise HTTPException(status_code=501, detail="Ingest module not available")

    try:
        logger.info("Ingest: running PDF ingestion. Expected outcome: chunks created and indexed in vectorstore.")
        result = ingest_pdfs()
        logger.info("Ingest: completed successfully. Result: %d chunks indexed.", result)
        return {"status": "ok", "ingested": result}
    except Exception as e:
        logger.error("Ingest: failed to process PDFs. Error: %s\nStack trace:\n%s", 
                    str(e), traceback.format_exc())
        try:
            handle_provider_error(e)
        except HTTPException as he:
            raise he
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF to the data directory and trigger ingestion.

    Expects multipart/form-data with a `file` field. Returns ingestion result.
    """
    try:
        logger.info("Upload: processing new file '%s'. Purpose: save PDF and prepare for indexing.", file.filename)
        
        # ensure data dir exists
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)

        dest_path = os.path.join(data_dir, file.filename)
        
        # read bytes and write
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        logger.info("Upload: file saved successfully to %s. Next step: check embeddings provider for ingestion.", dest_path)

        # Optional ingestion requires provider
        if app_status.embeddings_status.mode == ProviderMode.DISABLED:
            logger.warning("Upload: indexing skipped - no embeddings provider. Result: file saved but not searchable.")
            return {"status": "warn", "filename": file.filename, "indexed_chunks": 0,
                   "message": "Upload saved but indexing skipped - no embeddings provider available"}

        # Before ingestion, check DB state
        try:
            vs = Chroma(persist_directory=DB_DIR, embedding_function=_embed_fn)
            initial_size = len(vs.get()["ids"]) if hasattr(vs, "get") else 0
            logger.info("Upload: checked database state. Current size: %d vectors. Purpose: baseline for ingestion.", initial_size)
        except Exception as e:
            logger.warning("Upload: could not check database size (%s). Purpose: proceed with ingestion anyway.", str(e))
            initial_size = 0

        # Optionally trigger ingestion (could be controlled by client)
        try:
            logger.info("Upload: triggering ingestion. Purpose: index new content for search. Expected outcome: chunks created and indexed.")
            from ingest import ingest_pdfs
            count = ingest_pdfs()
            
            # Verify ingestion by checking final DB state
            try:
                vs = Chroma(persist_directory=DB_DIR, embedding_function=_embed_fn)
                final_size = len(vs.get()["ids"]) if hasattr(vs, "get") else 0
                vectors_added = final_size - initial_size
                logger.info("Upload: ingestion complete. Database: %d total vectors (added %d). Result: content is searchable.", 
                          final_size, vectors_added)
            except Exception as e:
                logger.warning("Upload: completed but could not verify database state (%s). Reported chunks: %d", str(e), count)
        except Exception:
            logger.warning("Upload: ingestion skipped - ingest module not available. Next step: manual ingestion may be needed.")
            count = 0

        return {"status": "ok", "filename": file.filename, "indexed_chunks": count}
    except Exception as e:
        logger.error("Upload: failed to process file '%s'. Error: %s\nStack trace:\n%s", 
                    file.filename, str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/library")
async def list_library():
        """Return list of PDFs in the data/ folder."""
        logger.info("Library: listing available PDFs. Purpose: show uploadable/searchable documents to user.")
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
                logger.info("Library: data directory not found. Result: returning empty list.")
                return {"files": []}
        pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        logger.info("Library: found %d PDF(s). Result: returning file list to client.", len(pdfs))
        return {"files": pdfs}


@app.get("/ui", response_class=HTMLResponse)
async def ui():
        """Serve a simple HTML page for uploading PDFs and chatting with the agent."""
        html = r"""
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>PDF Chat Assistant</title>
                <style>
                    body {
                        font-family: system-ui, -apple-system, sans-serif;
                        max-width: 1000px;
                        margin: 0 auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }
                    .container {
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                    }
                    h1, h2 { 
                        color: #2c3e50;
                        margin-top: 0;
                    }
                    button {
                        background: #3498db;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        cursor: pointer;
                        transition: background 0.3s;
                    }
                    button:hover { background: #2980b9; }
                    button:disabled {
                        background: #bdc3c7;
                        cursor: not-allowed;
                    }
                    select, textarea {
                        width: 100%;
                        padding: 8px;
                        margin: 8px 0;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }
                    .loading {
                        display: inline-block;
                        margin-left: 8px;
                        color: #7f8c8d;
                    }
                    #answer {
                        margin-top: 20px;
                        white-space: pre-wrap;
                    }
                    .answer-container {
                        border-left: 4px solid #3498db;
                        padding-left: 16px;
                        margin-top: 16px;
                    }
                    .sources {
                        margin-top: 16px;
                        font-size: 0.9em;
                        color: #7f8c8d;
                    }
                    .controls {
                        display: flex;
                        gap: 8px;
                        align-items: center;
                    }
                    .error {
                        color: #e74c3c;
                        padding: 8px;
                        border-radius: 4px;
                        background: #fadbd8;
                    }
                    .success {
                        color: #27ae60;
                        padding: 8px;
                        border-radius: 4px;
                        background: #daf7e8;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>PDF Chat Assistant</h1>
                    <div id="uploadSection">
                        <h2>Upload PDF</h2>
                        <div class="controls">
                            <input id="fileInput" type="file" accept="application/pdf" />
                            <button onclick="upload()" id="uploadBtn">Upload & Index</button>
                        </div>
                        <div id="uploadResult"></div>
                    </div>
                </div>

                <div class="container">
                    <h2>Document Library</h2>
                    <div class="controls">
                        <select id="librarySelect" onchange="enableChat()"></select>
                        <button onclick="refreshLibrary()">Refresh</button>
                    </div>
                </div>

                <div class="container">
                    <h2>Chat</h2>
                    <div class="controls">
                        <select id="responseStyle">
                            <option value="summary">Concise Summary</option>
                            <option value="detailed">Detailed Response</option>
                        </select>
                    </div>
                    <textarea 
                        id="queryBox" 
                        rows="3" 
                        placeholder="Select a PDF from the library above, then ask your question here..." 
                        disabled
                    ></textarea>
                    <div class="controls">
                        <button onclick="ask()" id="askButton" disabled>Ask Question</button>
                        <span id="status" class="loading" style="display:none">Processing...</span>
                    </div>
                    <div id="answer"></div>
                </div>

                <script>
                let currentSession = null;

                function setLoading(isLoading) {
                    document.getElementById('status').style.display = isLoading ? 'inline-block' : 'none';
                    document.getElementById('askButton').disabled = isLoading;
                    document.getElementById('uploadBtn').disabled = isLoading;
                }

                function enableChat() {
                    const selected = document.getElementById('librarySelect').value;
                    const chatEnabled = selected && selected.length > 0;
                    document.getElementById('queryBox').disabled = !chatEnabled;
                    document.getElementById('askButton').disabled = !chatEnabled;
                    if (chatEnabled) {
                        document.getElementById('queryBox').placeholder = "Ask a question about the selected PDF...";
                    }
                }

                async function upload() {
                    const inp = document.getElementById('fileInput');
                    if (!inp.files.length) { 
                        alert('Please choose a PDF file first'); 
                        return; 
                    }

                    setLoading(true);
                    const resultDiv = document.getElementById('uploadResult');
                    try {
                        const f = inp.files[0];
                        const fd = new FormData(); 
                        fd.append('file', f);
                        const res = await fetch('/upload', { method: 'POST', body: fd });
                        const j = await res.json();
                        
                        if (j.status === 'ok') {
                            resultDiv.className = 'success';
                            resultDiv.textContent = `Successfully uploaded ${f.name} and indexed ${j.indexed_chunks} chunks.`;
                        } else {
                            resultDiv.className = 'error';
                            resultDiv.textContent = j.message || 'Upload failed';
                        }
                        await refreshLibrary();
                    } catch (e) {
                        resultDiv.className = 'error';
                        resultDiv.textContent = 'Upload failed: ' + e.message;
                    }
                    setLoading(false);
                }

                async function refreshLibrary() {
                    try {
                        const res = await fetch('/library');
                        const j = await res.json();
                        const sel = document.getElementById('librarySelect');
                        sel.innerHTML = '<option value="">-- Select a PDF --</option>';
                        for (const f of j.files) {
                            const opt = document.createElement('option');
                            opt.value = f;
                            opt.innerText = f;
                            sel.appendChild(opt);
                        }
                        enableChat();
                    } catch (e) {
                        console.error('Failed to refresh library:', e);
                    }
                }

                function formatAnswer(response) {
                    let html = '<div class="answer-container">';
                    html += `<div>${response.answer}</div>`;
                    if (response.sources && response.sources.length) {
                        html += '<div class="sources"><strong>Sources:</strong><br>';
                        for (const src of response.sources) {
                            html += `• ${src.source}${src.page ? ` (page ${src.page})` : ''}<br>`;
                            if (src.text_snippet) {
                                html += `<small>${src.text_snippet}</small><br>`;
                            }
                        }
                        html += '</div>';
                    }
                    html += '</div>';
                    return html;
                }

                async function ask() {
                    const q = document.getElementById('queryBox').value;
                    if (!q) { 
                        alert('Please enter a question'); 
                        return; 
                    }

                    setLoading(true);
                    const answerDiv = document.getElementById('answer');
                    try {
                        const body = { 
                            query: q,
                            top_k: 3,
                            style: document.getElementById('responseStyle').value,
                            session_id: currentSession
                        };
                        
                        const res = await fetch('/query', { 
                            method: 'POST', 
                            headers: { 'Content-Type': 'application/json' }, 
                            body: JSON.stringify(body) 
                        });

                        if (!res.ok) {
                            const error = await res.json();
                            if (res.status === 402) {
                                answerDiv.innerHTML = '<div class="error">' +
                                    '<strong>⚠️ API Quota Exceeded</strong><br>' +
                                    'The AI service quota has been exhausted. Please try again later.</div>';
                            } else if (res.status === 429) {
                                answerDiv.innerHTML = '<div class="error">' +
                                    '<strong>⚠️ Rate Limit</strong><br>' +
                                    'Too many requests. Please wait a moment and try again.</div>';
                            } else {
                                answerDiv.innerHTML = '<div class="error">' +
                                    `<strong>Error:</strong> ${error.detail || 'Unknown error occurred'}</div>`;
                            }
                            return;
                        }

                        const response = await res.json();
                        currentSession = response.session_id;
                        answerDiv.innerHTML = formatAnswer(response);
                    } catch (e) {
                        answerDiv.innerHTML = '<div class="error">' +
                            '<strong>Error:</strong> Failed to communicate with the server. Please try again.</div>';
                    }
                    setLoading(false);
                }

                // Handle Enter key in textarea
                document.getElementById('queryBox').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        ask();
                    }
                });

                // Initial load
                refreshLibrary();
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=html)


def handle_provider_error(e: Exception) -> None:
    """Check if the error is related to the upstream LLM provider (quota/auth) and
    raise an appropriate HTTP exception with a helpful message.
    """
    error_str = str(e).lower()
    error_class = e.__class__.__name__ if hasattr(e, '__class__') else ""

    if error_class == 'RateLimitError' or 'rate limit' in error_str:
        raise HTTPException(
            status_code=429,
            detail="LLM provider rate limit exceeded. Please try again in a few minutes."
        )
    elif error_class == 'AuthenticationError' or 'authentication' in error_str:
        raise HTTPException(
            status_code=401,
            detail="LLM provider authentication failed. Please check your credentials."
        )
    elif 'insufficient_quota' in error_str or 'exceeded your current quota' in error_str or 'quota' in error_str:
        raise HTTPException(
            status_code=402,
            detail="LLM provider quota has been exhausted. Please check your billing/usage status."
        )
    elif 'billing' in error_str:
        raise HTTPException(
            status_code=402,
            detail="LLM provider billing issue detected. Please check your billing status."
        )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Query the document store and return the answer with cited sources.

    Request body:
    - query: user text
    - top_k: number of returned results from retriever (currently not wired to retriever in this wrapper)
    - style: 'summary' or 'detailed'
    - session_id: optional session id (if omitted a new one is created and returned)
    """
    global _qa
    if _qa is None:
        raise HTTPException(status_code=503, detail="Search backend not initialized. Run ingest and restart the service.")

    session_id, history = _get_or_create_history(req.session_id)

    try:
        logger.info("Query: starting query execution for session %s. Purpose: retrieve relevant docs and generate answer.", session_id)
        style = (req.style or "summary").lower()
        style_hint = f"User prefers a {style} answer."
        system_instructions = (
            "You are a helpful, friendly assistant. "
            "Cite specific snippets from the retrieved context when useful. "
            + ("Keep answers concise (3-5 sentences)." if style == "summary" else "Provide step-by-step explanations and include brief source quotes.")
        )

        full_q = f"{style_hint}\n\n{system_instructions}\n\nQuestion: {req.query}"

        # Apply per-request top_k to the retriever (when supported)
        try:
            if _retriever is not None and hasattr(_retriever, 'search_kwargs'):
                logger.info("Retriever: adjusting search parameters. Purpose: use client-requested k=%d.", req.top_k)
                _retriever.search_kwargs = {"k": req.top_k}
        except Exception as e:
            # Non-fatal: if the retriever doesn't support per-request k, continue
            logger.debug("Retriever: could not set top_k=%d (%s). Using default k.", req.top_k, str(e))

        # Prepare chat_history for chain
        chat_history = [(m.type, m.content) for m in history.messages]

        logger.info("Retriever: fetching relevant docs. Purpose: find context for query. Expected outcome: up to %d docs from vectorstore.", req.top_k)
        # Get documents from retriever for sources
        retrieved_docs = _retriever.get_relevant_documents(full_q)
        logger.info("Retriever: found %d relevant docs. Next step: generate answer using LLM.", len(retrieved_docs))
        
        # Get answer from chain (our _qa is a callable wrapper)
        logger.info("LLM: calling _qa with context, history and retrieved docs. Purpose: generate answer using retrieved docs. Expected outcome: natural language response.")
        answer = _qa(full_q, chat_history, retrieved_docs)
        logger.info("LLM: generated answer successfully. Next step: prepare sources and update history.")

        # Prepare sources from retrieved documents
        sources = []
        for doc in retrieved_docs:
            meta = getattr(doc, "metadata", {})
            sources.append({
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "text_snippet": (doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
            })

        # update history
        history.add_user_message(req.query)
        history.add_ai_message(answer)

        logger.info("Query: completed successfully for session %s. Result: answer generated with %d source(s).", session_id, len(sources))

        return QueryResponse(session_id=session_id, answer=answer, sources=sources)

    except Exception as e:
        logger.error("Query: failed for session %s. Error in %s: %s\nStack trace:\n%s", 
                    session_id, 
                    e.__class__.__name__,
                    str(e),
                    traceback.format_exc())
        try:
            handle_provider_error(e)
        except HTTPException as he:
            raise he
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_history")
class ClearHistoryRequest(BaseModel):
    session_id: str


@app.post("/clear_history")
async def clear_history(req: ClearHistoryRequest):
    """Clear the chat history for a given session_id (body: {session_id})."""
    session_id = req.session_id
    with _hist_lock:
        if session_id in _histories:
            del _histories[session_id]
            logger.info(f"Cleared history for session {session_id}")
            return {"status": "ok", "cleared": session_id}
        else:
            raise HTTPException(status_code=404, detail="session_id not found")

