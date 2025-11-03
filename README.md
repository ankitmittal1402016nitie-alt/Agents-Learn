# Agentic AI: Conversational Docs

A small starter project to ingest PDFs, index them with embeddings, and query them conversationally.

Project layout
```
your-agent/
├─ app.py
├─ ingest.py
├─ requirements.txt
├─ .env                 # holds OPENAI_API_KEY (optional)
├─ .gitignore
├─ data/                # put PDFs here
└─ db/                  # created by Chroma automatically
```

Quick start
1. Create and activate a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure provider credentials in `.env`:

Preferred: Google Gemini (recommended)

```
# Either set an API key
GOOGLE_API_KEY=ya29.YOUR_KEY_HERE
# Or set a service account JSON path
# GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/service-account.json
```

Optional fallback: OpenAI (only used when Google Gemini is not configured)

```
# OPENAI_API_KEY=sk-...
```

3. Place PDFs in `data/` and run ingestion (example):

```powershell
# Run the FastAPI app with uvicorn (preferred):
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# If you only want to run the ingestion pipeline (index PDFs):
python ingest.py

# After the server is running, you can POST to /ingest or use the /ui for uploads and chat.
```

Notes
- `ingest.py` contains placeholder PDF extraction logic. Replace with `pypdf` or `pdfminer.six` for production quality extraction.
- This scaffold uses `chromadb` for storage; `db/` will be created automatically.
- Secure your `.env` and never commit it to Git.

Gemini API & debug fallback
- Configure Google Gemini credentials in `.env` as described above. Either set `GOOGLE_API_KEY` (simple) or point `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON for server-to-server auth.
- If you get provider errors at runtime (HTTP 429 Quota, 401 Auth, or 402 Billing/Quota exhausted), the app's test harness (`test_gemini.py`) and the adapter (`src/gemini.py`) will automatically fall back to deterministic debug stubs so local development and tests continue to work without live API access.
- To force stub mode locally set `DEBUG=1` in your environment or in `.env`.
- If you need production API access, ensure your Google Cloud project has billing enabled and the appropriate API/quota for the Gemini endpoints you plan to use.

Next steps
- Implement robust PDF text extraction in `ingest.extract_text_from_pdf`
- Add authentication to the FastAPI app if exposing publicly
- Add conversational chain logic (e.g., LangChain QA over retrieved documents)
"# Agents-Learn" 
