"""Core Gemini chat functionality.

This module provides a focused implementation of chat/completion
functionality using the Gemini API, with proper error handling and
version compatibility.
"""
import os
import logging
import hashlib
from typing import List, Optional, Dict, Any

from .gemini_setup import validate_setup

logger = logging.getLogger(__name__)

class GeminiChatError(Exception):
    """Raised for Gemini chat-specific errors."""
    pass

class BaseChatModel:
    """Base class for chat models."""
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        raise NotImplementedError
        
    def __call__(self, prompt: str, context: str = "", **kwargs) -> str:
        return self.generate(prompt, context, **kwargs)

class StubChatModel(BaseChatModel):
    """Deterministic chat model for testing/offline use."""
    def __init__(self):
        logger.info("Using StubChatModel - will return deterministic responses")
        self._responses = [
            "Hello! I am a test response.",
            "This is a deterministic reply based on your input.",
            "I understand what you're asking about.",
            "Let me help you with that question.",
        ]
        
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        """Return deterministic response based on input hash."""
        combined = f"{context}\n{prompt}"
        hash_int = int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)
        return self._responses[hash_int % len(self._responses)]

class GeminiChatModel(BaseChatModel):
    """Production chat model using Gemini API."""
    def __init__(
        self, 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        # Import and configure API
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError as e:
            raise GeminiChatError("Failed to import google-generativeai") from e
            
        # Set up credentials
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise GeminiChatError("GOOGLE_API_KEY environment variable not set")
        self.genai.configure(api_key=api_key)
        
        # Set up model and parameters
        self.model = model or os.getenv('GEMINI_MODEL', 'models/gemini-pro')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Validate model availability
        try:
            models = self.genai.list_models()
            if not any(m.name == self.model for m in models):
                raise GeminiChatError(f"Model {self.model} not found in available models")
        except Exception as e:
            raise GeminiChatError(f"Failed to validate model availability: {e}")
            
        logger.info(f"Initialized GeminiChat with model {self.model}")
        
    def _prepare_messages(self, prompt: str, context: str = "") -> List[Dict[str, Any]]:
        """Format messages for the API."""
        messages = []
        if context:
            messages.append({
                "role": "user",
                "parts": [{"text": context}]
            })
        messages.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        return messages
        
    def generate(
        self, 
        prompt: str, 
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate completion with proper error handling."""
        try:
            messages = self._prepare_messages(prompt, context)
            
            # Set up generation config
            generation_config = {
                "temperature": temperature if temperature is not None else self.temperature,
                "max_output_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                **kwargs
            }
            
            # Start chat and get response
            chat = self.genai.start_chat(model=self.model)
            response = chat.send_message(
                content=messages[-1]["parts"][0]["text"],
                generation_config=generation_config
            )
            
            if not response.text:
                raise GeminiChatError("Empty response from API")
            
            return response.text
            
        except Exception as e:
            raise GeminiChatError(f"Chat generation failed: {e}") from e