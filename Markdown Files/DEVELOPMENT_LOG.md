# Development Log - Conversational Document QA System

## Project Evolution and Changes

### 1. Initial Setup and Model Configuration
- Updated model configurations to use correct Gemini models
- Moved from older PaLM model (text-bison-001) to current Gemini models
- Configured models in environment:
  - Primary: `gemini-2.5-flash`
  - Fallback: `gemini-1.5-flash`
  - Embeddings: `models/gemini-embedding-001`

### 2. Environment Configuration Consolidation
- Consolidated all configuration into a single `.env` file
- Removed `.env.example` in favor of a single source of truth
- Added temperature control and debug settings
- Environment structure:
```properties
# Google Gemini Configuration
GOOGLE_API_KEY=...
GEMINI_PRIMARY_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
DEBUG=False
TEMPERATURE=0.7
```

### 3. Embedding System Enhancement
- Implemented `EmbeddingWrapper` class in `app.py`
- Added required methods: `__call__`, `embed_documents`, `embed_query`
- Ensured proper integration with ChromaDB for document storage
- Fixed vector dimensionality (3072-D vectors) compatibility

### 4. Context and Response Improvements
#### Initial Context Issue
Problem: Model responses lacked document context:
```
"I'd love to tell you about Jupiter, but unfortunately, no context was retrieved for me to pull information from..."
```

#### Solution Implementation
1. Enhanced `_qa_call` to properly include document context
2. Added proper context formatting with source attribution
3. Implemented retrieved document handling with metadata

### 5. Hybrid Search Implementation
Added capability to combine document and web knowledge:
1. Created external search integration with Google Custom Search
2. Implemented priority system:
   - Primary: Document context from vector store
   - Secondary: Web search results
3. Added clear source attribution in responses

### 6. Error Handling and Reliability
- Implemented fallback mechanism between models
- Added comprehensive error handling
- Improved logging for debugging
- Added graceful degradation for missing services

## Key Technical Improvements

### Document Context Enhancement
```python
def _qa_call(question: str, chat_history: list, retrieved_docs: list = None):
    """Enhanced context handling with both document and web sources"""
    # Document context formatting
    docs_text = []
    for i, d in enumerate(retrieved_docs[:5], start=1):
        meta = getattr(d, "metadata", {})
        snippet = d.page_content[:800] + "..."
        docs_text.append(f"[DOC {i}] source={meta.get('source')} page={meta.get('page')}\n{snippet}")
```

### Response Structure
The system now provides structured responses:
1. Primary document information with source attribution
2. Supplementary web information when relevant
3. Clear handling of any contradictions between sources

## Testing and Verification Framework

### 1. Unit Testing
```python
# test_qa_system.py

import pytest
from unittest.mock import MagicMock
from src.gemini import GeminiChat, GeminiEmbeddings

class TestQASystem:
    @pytest.fixture
    def mock_llm(self):
        return MagicMock(spec=GeminiChat)
    
    @pytest.fixture
    def mock_embeddings(self):
        return MagicMock(spec=GeminiEmbeddings)
    
    def test_document_retrieval(self, mock_embeddings):
        # Test vector similarity search
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        results = self.retriever.get_relevant_documents("test query")
        assert len(results) > 0
        
    def test_context_formatting(self, mock_llm):
        # Test context assembly
        docs = [{"content": "test", "metadata": {"source": "test.pdf"}}]
        result = self._qa_call("test", [], docs)
        assert "[DOC 1]" in result
        
    def test_hybrid_search(self, mock_llm, mock_embeddings):
        # Test combined document and web search
        doc_results = self.retriever.get_relevant_documents("test")
        web_results = self.external_search.search("test")
        assert len(doc_results) > 0
        assert len(web_results) > 0
```

### 2. Integration Testing
```python
# test_integration.py

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_query_endpoint():
    response = client.post(
        "/query",
        json={
            "query": "What is Jupiter?",
            "top_k": 3,
            "style": "detailed"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) <= 3

def test_hybrid_search():
    response = client.post(
        "/hybrid-search",
        json={"query": "Jupiter atmosphere"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_results" in data
    assert "web_results" in data
```

### 3. Load Testing
```python
# test_load.py

import asyncio
from locust import HttpUser, task, between

class QASystemUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def query_documents(self):
        self.client.post(
            "/query",
            json={
                "query": "What is Jupiter?",
                "top_k": 3
            }
        )

    @task
    def hybrid_search(self):
        self.client.post(
            "/hybrid-search",
            json={"query": "Jupiter atmosphere"}
        )
```

### 4. Manual Testing Scripts
```python
# manual_test.py

import requests
import json

def test_query_flow():
    # Test document search
    doc_response = requests.post(
        "http://localhost:8000/query",
        json={
            "query": "What is Jupiter?",
            "top_k": 3
        }
    )
    print("Document Search Results:", json.dumps(doc_response.json(), indent=2))
    
    # Test hybrid search
    hybrid_response = requests.post(
        "http://localhost:8000/hybrid-search",
        json={"query": "Jupiter recent discoveries"}
    )
    print("Hybrid Search Results:", json.dumps(hybrid_response.json(), indent=2))

if __name__ == "__main__":
    test_query_flow()
```

### 5. Performance Monitoring
```python
# monitoring.py

import time
from functools import wraps
from prometheus_client import Counter, Histogram

# Metrics
QUERY_DURATION = Histogram(
    'query_duration_seconds',
    'Time spent processing queries',
    ['endpoint']
)
QUERY_ERRORS = Counter(
    'query_errors_total',
    'Total query errors',
    ['error_type']
)

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            QUERY_DURATION.labels(endpoint=func.__name__).observe(
                time.time() - start
            )
            return result
        except Exception as e:
            QUERY_ERRORS.labels(error_type=type(e).__name__).inc()
            raise
    return wrapper
```

## Current Features
1. Document Search:
   - Vector-based similarity search
   - Metadata preservation
   - Source attribution

2. External Knowledge:
   - Web search integration
   - Source prioritization
   - Clear attribution

3. Response Generation:
   - Context-aware responses
   - Source citations
   - Handling of contradictions

4. Error Handling:
   - Model fallbacks
   - Graceful degradation
   - Comprehensive logging

## Detailed Setup Instructions

### 1. Environment Configuration
Create `.env` file with all required settings:
```properties
# Gemini API Configuration
GOOGLE_API_KEY=your_key
GEMINI_PRIMARY_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001

# External Search Configuration
GOOGLE_SEARCH_API_KEY=your_search_key
GOOGLE_SEARCH_CX=your_search_engine_id

# Optional Settings
DEBUG=false
TEMPERATURE=0.7
MAX_TOKENS=1024
RETRIEVER_K=3
SIMILARITY_THRESHOLD=0.5
```

### 2. Dependencies Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov locust
```

### 3. Database Setup
```powershell
# Create required directories
mkdir data db

# Initialize ChromaDB
python -c "import chromadb; chromadb.PersistentClient(path='db')"
```

### 4. API Key Setup
1. Google Cloud Console:
   ```powershell
   # Set environment variables
   $env:GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   $env:GOOGLE_API_KEY="your_api_key"
   ```

2. Custom Search Setup:
   - Visit [Google Custom Search](https://programmablesearchengine.google.com/)
   - Create new search engine
   - Copy Search Engine ID (cx)
   - Add to `.env`: `GOOGLE_SEARCH_CX=your_cx`

### 3. Running the Server
```powershell
uvicorn app:app --reload
```

## Usage Examples

### Basic Query
```http
POST /query
{
    "query": "Tell me about Jupiter",
    "top_k": 3,
    "style": "detailed"
}
```

### Expected Response
```json
{
    "session_id": "uuid",
    "answer": "Based on our documents [DOC 1], Jupiter is... Additional recent findings [WEB 1] indicate...",
    "sources": [
        {"source": "astronomy.pdf", "page": "12", "text_snippet": "..."},
        {"source": "web", "link": "https://...", "text_snippet": "..."}
    ]
}
```

## Future Improvements
1. Add source citation numbering in responses
2. Implement caching for web search results
3. Add configuration for adjusting the balance of document vs web information
4. Enhance testing coverage