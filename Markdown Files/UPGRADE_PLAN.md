# 🚀 Agents-Learn — FastAPI Upgrade Plan (RAG + Gemini + Hoverable Citations)

**Objective:**  
Enhance the existing FastAPI-based Agentic AI app so it provides accurate, grounded answers from PDFs and web pages, fuses external data when relevant, and shows inline hoverable citations (ChatGPT-style).  
Keep it fully compatible with **Gemini Free Tier**.

---

## 🧠 0. Target Architecture

**Services (FastAPI routes):**
- `POST /ingest` — upload and index PDFs  
- `POST /ingest_url` — fetch & index one webpage  
- `POST /chat` — streaming RAG chat (uses docs + optional web)  
- `GET /documents` — list indexed docs for debugging  
- `GET /healthz` — simple status  
- `GET /ui` — lightweight HTML/JS chat interface

**Storage Layout:**
```
/data   → raw PDFs (input)
/db     → Chroma vector store for PDFs
/webdb  → Chroma vector store for fetched web pages
/logs   → structured JSON logs
/tmp    → temporary downloads
```

**Models (Gemini Free Compatible):**
- Chat model → `gemini-2.5-flash-lite`
- Embedding model → `text-embedding-004`
- Optional reranker → local CPU reranker (`flashrank` or `bge-reranker-mini`)

---

## 1️⃣ Ingestion — Idempotent, Reliable, Page-Level

### Tasks
- Use **PyPDF** for text; fall back to **pymupdf**; add optional OCR.
- Chunk PDFs: `chunk_size=1000`, `overlap=150`.
- Add metadata:
  ```json
  {
    "source": "data/file.pdf",
    "page": 12,
    "type": "pdf",
    "title": "file.pdf",
    "sha": "sha1(source+page+normalized_text)"
  }
  ```
- Use `sha` as Chroma document ID → **no duplicates on re-ingest**.
- Persist embeddings in `/db` (PDFs) or `/webdb` (webpages).
- Add timestamp and total chunk count in logs.

---

## 2️⃣ Retrieval — Hybrid + MMR + Optional Reranker

### Hybrid retrieval pipeline
1. **BM25Retriever** for keyword match  
2. **ChromaRetriever** for embeddings  
3. Merge both results (Reciprocal Rank Fusion or weighted scores)  
4. Apply **MMR** (`fetch_k=24`, `k=8`, `lambda_mult=0.5`)  
5. Optionally **rerank top 12** with local reranker (CPU-based)

### Query expansion (optional)
- Generate 2–3 reformulations using Gemini (multi-query)
- Union all retrieved results, deduplicate, then rerank

### Benefits
- Sharper, context-aware retrieval
- Handles synonyms and exact matches together
- Improves precision without extra cost

---

## 3️⃣ Web Augmentation — External Data Integration

### Logic
- Call external web tool **only when needed**:
  - No strong document match, or
  - User provides a URL, or
  - Query clearly asks for “latest” / “current info”.

### Implementation
- `POST /ingest_url`
  - Fetch page (`requests`)
  - Clean text with `readability-lxml`
  - Split into ~1200-char chunks
  - Embed + upsert into `/webdb`
- Keep `captured_at` + TTL = **7 days**
- Combine `/db` and `/webdb` results in same retriever pipeline

---

## 4️⃣ Answer Generation — Grounded, Context-Limited, Trustworthy

### Prompting
- Compact system prompt:
  > You are a helpful assistant. Use retrieved sources only.  
  > Cite each claim inline as [1], [2] with hoverable snippet info.  
  > If uncertain, say you don’t know.

- “Summary” vs “Detailed” toggle supported via `style` param.

### Context preparation
- Group chunks by source; keep max 6.
- Cap total context tokens to 12k.
- Drop low-score chunks if over limit.
- Generate brief outline (headings of sources) before answer.

### Attribution
- Match each sentence → supporting chunk.
- Remove or rewrite sentences with no matching context.
- Build citation tags (`C1`, `C2`, etc.) for each chunk.

---

## 5️⃣ `/chat` Route — Streaming, Structured Response

### Response JSON (from API)
```json
{
  "segments": [
    {"text": "The warranty lasts 24 months", "cite": "C1"},
    {"text": ", covering parts and labor.", "cite": "C2"}
  ],
  "citations": [
    {"id": "C1", "label": "1", "source": "handbook.pdf", "page": 12, "snippet": "Warranty valid 24 months...", "type": "pdf"},
    {"id": "C2", "label": "2", "source": "https://example.com/warranty", "snippet": "Covers parts and labor...", "type": "web"}
  ]
}
```

### Implementation
- Stream segments via **Server-Sent Events (SSE)**.
- After each sentence + citation computed, send it downstream.
- Use **asyncio.Semaphore(2)** for concurrency guard.
- Add 429 **backoff with jitter** for Gemini API limits.

---

## 6️⃣ `/ui` Page — Minimal Frontend with Hoverable Citations

### Frontend Tech
- Plain HTML + JS (no framework) OR  
- `htmx + Alpine.js` (tiny, reactive)

### Rendering
```html
<span class="answer-seg">
  The warranty lasts 24 months
  <sup class="cite" data-cite="C1">[1]</sup>
</span>
```

### Hover tooltips
- Use **Tippy.js** or custom CSS tooltip.
- On hover, fetch citation info from JSON in memory.
- Show snippet + source title/page in popup.
- Keep a bottom “Sources” accordion for accessibility.

---

## 7️⃣ Environment Configuration

Add these variables to `.env.example`:

```
GEMINI_MODEL_CHAT=gemini-2.5-flash-lite
GEMINI_MODEL_EMBED=text-embedding-004
RAG_TOP_K=4
RAG_FETCH_K=24
RAG_USE_MMR=true
RAG_USE_RERANK=false
MAX_TOKENS_CONTEXT=12000
WEB_TTL_DAYS=7
PORT=8000
```

---

## 8️⃣ Logging & Health

- Log JSON objects per request to `/logs/app.log`:
  ```json
  {"ts":"...", "route":"/chat", "q":"...", "k":6, "docs":[{"src":"file.pdf","p":12}], "latency_ms":820, "tokens_in":3400, "tokens_out":420, "ok":true}
  ```
- Add `/healthz` returning:
  ```json
  {"status":"ok","docs":123,"chunks":456}
  ```

---

## 9️⃣ Evaluation & Testing

- Maintain a **golden Q&A set (10–15)** to test retrieval quality.
- Compute **Recall@k** to verify hybrid & MMR effectiveness.
- Add unit tests:
  - PDF text extraction
  - Web page ingestion
  - RAG pipeline recall
  - 429 retry handling
  - Citation rendering JSON structure

---

## 🔟 README / Documentation Updates

- Update to reflect **FastAPI-only** workflow:
  ```bash
  uvicorn app:app --reload
  ```
- Provide one-page quick start:
  1. Drop PDFs in `/data`
  2. `python ingest.py`
  3. Run FastAPI → open `/ui`
  4. Ask: “What is warranty coverage?”
- Add cleanup snippet:
  ```bash
  rmdir /s /q db
  python ingest.py
  ```

---

## ✅ Expected Results After Upgrade

- FastAPI backend only — clean, organized routes.
- Answers grounded in **documents + live web**.
- Inline hover citations with page/URL snippets.
- High precision due to **Hybrid + MMR + Rerank**.
- Gemini free-tier safe (rate limits respected).
- Structured logs, minimal UI, better UX.

---

## 🧩 GitHub Tasks Checklist

1. [ ] Switch to FastAPI-only routes & static `/ui`.  
2. [ ] Implement idempotent ingestion with PDF + URL.  
3. [ ] Add hybrid retriever (BM25 + Chroma).  
4. [ ] Enable MMR & local reranker.  
5. [ ] Integrate conditional web ingestion.  
6. [ ] Build context + attribution system.  
7. [ ] Return structured JSON with `segments[]`, `citations[]`.  
8. [ ] Add hoverable citation frontend.  
9. [ ] Add rate-limit backoff & logging.  
10. [ ] Update README and `.env.example`.

---

**Author:** Ankit Mittal  
**Repository:** [Agents-Learn](https://github.com/ankitmittal1402016nitie-alt/Agents-Learn)  
**LLM Backend:** Gemini 2.5 Flash-Lite (Free Tier)  
**Framework:** FastAPI + Chroma + PyPDF + Readability + minimal JS
