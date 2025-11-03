"""
A simplified and robust adapter for the Google Gemini API.

This module provides clean, straightforward wrappers for Gemini's embedding
and chat functionalities, based on the latest stable SDK practices. It also
includes deterministic stubs for offline testing when the DEBUG environment
variable is set.
"""
import os
import logging
import hashlib
import random
from typing import List, Optional, Dict

# --- Configuration ---
# Configure logger to include filename and line number
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Exception ---
class GeminiError(Exception):
    """Custom exception for Gemini-related errors."""
    pass

# --- Stub Implementations for DEBUG mode ---

class StubEmbeddings:
    """A deterministic stub for embeddings, used for offline testing."""
    def __init__(self, dim: int = 768):
        self.dim = dim
        logger.info("Using StubEmbeddings for offline testing.")

    def _get_deterministic_vector(self, text: str) -> List[float]:
        """Generates a consistent vector from a string without requiring numpy.

        Uses Python's random.Random seeded from a stable hash so outputs are
        deterministic across runs. Values are in [0,1).
        """
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        return [rng.random() for _ in range(self.dim)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_deterministic_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_deterministic_vector(text)

class StubChat:
    """A deterministic stub for chat, used for offline testing."""
    def __init__(self):
        logger.info("StubChat: initializing test provider. Purpose: provide deterministic responses for testing.")
        self._responses = [
            "This is a deterministic test response.",
            "As a stub model, I provide predictable answers.",
            "Your query has been noted by the stub chat model.",
        ]
        logger.info("StubChat: initialization successful. Result: ready with %d canned responses.", len(self._responses))

    def generate(self, prompt: str, **kwargs) -> str:
        """Generates a consistent response from a prompt."""
        seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(self._responses)
        return self._responses[seed]

# --- Live Gemini API Implementations ---

class GeminiEmbeddings:
    """Wrapper for Google Gemini Embeddings."""
    def __init__(self, model_name: str = None):
        if os.getenv("DEBUG", "false").lower() in ("true", "1"):
            logger.info("GeminiEmbeddings: DEBUG mode detected. Purpose: use deterministic stubs for testing.")
            self._provider = StubEmbeddings()
            return

        try:
            # Initialize model name from parameter or environment
            self.model_name = model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
            logger.info(f"GeminiEmbeddings: initializing provider with model {self.model_name}")
            
            import google.generativeai as genai
            self._genai = genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise GeminiError("GOOGLE_API_KEY not found in environment.")
            self._genai.configure(api_key=api_key)
            self._provider = self  # Point to self for live mode
            logger.info("GeminiEmbeddings: initialization successful. Result: live provider ready with model %s.", self.model_name)
        except ImportError:
            raise GeminiError("The 'google-generativeai' package is not installed. Please install it with 'pip install google-generativeai'.")
        except Exception as e:
            raise GeminiError(f"Failed to configure Gemini: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._provider is not self:
            return self._provider.embed_documents(texts)
        try:
            logger.info(
                "embed_documents: requesting embeddings for %d documents. Purpose: create vectors for indexing/retrieval.",
                len(texts),
            )
            
            vectors = []
            for text in texts:
                # Format content as per Gemini API requirements
                content = {"text": text}
                result = self._genai.embed_content(
                    model=self.model_name,
                    content=content,
                    task_type="retrieval_document"
                )
                
                # Extract embeddings from the response
                if hasattr(result, 'embedding'):
                    vectors.append(result.embedding)
                elif isinstance(result, dict) and 'embedding' in result:
                    vectors.append(result['embedding'])
                elif hasattr(result, 'values'):
                    vectors.append(result.values)
                elif isinstance(result, dict) and 'values' in result:
                    vectors.append(result['values'])
                else:
                    raise GeminiError(f"Unexpected embedding format: {type(result)}")
                    
            logger.info("embed_documents: extracted %d vectors; first vector length=%d", 
                       len(vectors), len(vectors[0]) if vectors and vectors[0] else 0)
            return vectors
            
        except Exception as e:
            raise GeminiError(f"Failed to embed documents: {e}")

    def embed_query(self, text: str) -> List[float]:
        if self._provider is not self:
            return self._provider.embed_query(text)
        try:
            # Format content as per Gemini API requirements
            content = {"text": text}
            result = self._genai.embed_content(
                model=self.model_name,
                content=content,
                task_type="retrieval_query"
            )
            
            logger.info("embed_query: received response. Extracting embedding vector.")
            
            # Extract embeddings from the response
            if hasattr(result, 'embedding'):
                values = result.embedding
            elif isinstance(result, dict) and 'embedding' in result:
                values = result['embedding']
            elif hasattr(result, 'values'):
                values = result.values
            elif isinstance(result, dict) and 'values' in result:
                values = result['values']
            else:
                raise GeminiError(f"Unexpected embedding format: {type(result)}")
                
            logger.info("embed_query: extraction successful (vector length=%d)", len(values))
            return values
            
        except Exception as e:
            raise GeminiError(f"Failed to embed query: {e}")


class GeminiChat:
    """Wrapper for Google Gemini Chat."""
    
    def __init__(self, model: str = None, temperature: float = None, **kwargs):
        # allow DEBUG stub
        if os.getenv("DEBUG", "false").lower() in ("true", "1"):
            self._provider = StubChat()
            return

        # Get model from parameters or environment
        self.model_name = model or os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))
        
        try:
            import google.generativeai as genai
            self._genai = genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise GeminiError("GOOGLE_API_KEY not found in environment.")
            self._genai.configure(api_key=api_key)
            
            # Handle model name format - Gemini uses clean names without prefix
            if str(self.model_name).startswith("models/"):
                self.model_name = self.model_name.replace("models/", "", 1)
                logger.info(f"Removing 'models/' prefix from model name. Using: {self.model_name}")
            
            # Create the GenerativeModel instance
            try:
                self.model = self._genai.GenerativeModel(self.model_name)
            except Exception as e:
                raise GeminiError(f"Failed to initialize model {self.model_name}: {str(e)}")
                
            self._provider = self  # Point to self for live mode
            logger.info(f"GeminiChat initialized; model={self.model_name}")
        except ImportError:
            raise GeminiError("The 'google-generativeai' package is not installed. Please install it with 'pip install google-generativeai'.")
        except Exception as e:
            raise GeminiError(f"Failed to configure Gemini: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        if self._provider is not self:
            return self._provider.generate(prompt, **kwargs)
        try:
            # Some SDK versions do not accept arbitrary kwargs like `context`.
            # If a `context` kwarg is provided, merge it into the prompt string
            # instead of passing it through to generate_content.
            context = None
            if 'context' in kwargs:
                context = kwargs.pop('context')
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            response = self.model.generate_content(full_prompt, **kwargs)
            # The response object may expose `.text` or `.content` depending on SDK.
            if hasattr(response, 'text'):
                return response.text
            if hasattr(response, 'content'):
                return getattr(response, 'content')
            # Fallback: stringify
            return str(response)
        except Exception as e:
            raise GeminiError(f"Failed to generate chat response: {e}")

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)
