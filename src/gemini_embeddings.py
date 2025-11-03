"""Core Gemini embeddings functionality.

This module provides a more focused and robust implementation of the
embedding-specific functionality for the Gemini API, with proper error
handling and API version compatibility.
"""
import os
import logging
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any

from .gemini_setup import validate_setup

logger = logging.getLogger(__name__)

class GeminiEmbeddingError(Exception):
    """Raised for Gemini embedding-specific errors."""
    pass

class BaseEmbedder:
    """Base class for embedding providers."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
        
    def embed_query(self, text: str) -> List[float]:
        """Default implementation: embed single text using batch method."""
        return self.embed_documents([text])[0]

class StubEmbedder(BaseEmbedder):
    """Deterministic stub embedder for testing/offline use."""
    def __init__(self, dim: int = 768):
        self.dim = dim
        logger.info(f"Using StubEmbedder (dim={dim}) - will return deterministic test vectors")
        
    def _hash_to_vector(self, text: str) -> List[float]:
        """Generate deterministic unit vector from text hash."""
        # Use text hash as random seed for reproducibility
        hash_bytes = hashlib.sha256(text.encode()).digest()[:8]
        seed = int.from_bytes(hash_bytes, 'big')
        
        # Generate random unit vector (normalized for cosine similarity)
        rng = np.random.RandomState(seed)
        vec = rng.standard_normal(self.dim)
        return (vec / np.linalg.norm(vec)).tolist()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create deterministic vectors for testing."""
        return [self._hash_to_vector(text) for text in texts]

class GeminiEmbedder(BaseEmbedder):
    """Production embedder using Gemini API."""
    def __init__(self, model: Optional[str] = None):
        # Validate imports and credentials first
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError as e:
            raise GeminiEmbeddingError("Failed to import google-generativeai") from e
            
        # Configure client
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise GeminiEmbeddingError("GOOGLE_API_KEY environment variable not set")
        self.genai.configure(api_key=api_key)
        
        # Set up model
        self.model = model or os.getenv('GEMINI_EMBEDDING_MODEL', 'models/gemini-embedding-001')
        logger.info(f"Initialized GeminiEmbedder with model {self.model}")
        
        # Validate model availability
        try:
            models = self.genai.list_models()
            if not any(m.name == self.model for m in models):
                raise GeminiEmbeddingError(f"Model {self.model} not found in available models")
        except Exception as e:
            raise GeminiEmbeddingError(f"Failed to validate model availability: {e}")
    
    def _call_embeddings_api(self, texts: List[str]) -> List[List[float]]:
        """Make the actual API call with proper error handling."""
        try:
            # Use embeddings-specific endpoint
            response = self.genai.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document" # or "retrieval_query" for queries
            )
            # Extract embeddings from response
            if hasattr(response, 'embeddings'):
                return [emb.values for emb in response.embeddings]
            else:
                raise GeminiEmbeddingError("Unexpected response format")
        except Exception as e:
            raise GeminiEmbeddingError(f"Embeddings API call failed: {e}") from e
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with batching and error handling."""
        if not texts:
            return []
            
        try:
            # Could add batching here if needed
            return self._call_embeddings_api(texts)
        except Exception as e:
            raise GeminiEmbeddingError(f"Document embedding failed: {e}") from e
            
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query with query-specific options."""
        try:
            # Use query-specific endpoint options
            response = self.genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"  # Optimize for queries
            )
            if hasattr(response, 'embeddings') and response.embeddings:
                return response.embeddings[0].values
            raise GeminiEmbeddingError("Unexpected response format")
        except Exception as e:
            raise GeminiEmbeddingError(f"Query embedding failed: {e}") from e