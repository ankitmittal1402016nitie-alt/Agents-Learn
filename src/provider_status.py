"""Provider status tracking for the app.

This module defines enums and status objects for tracking the state of providers
(e.g., embeddings and LLM providers) used by the app.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class ProviderMode(str, Enum):
    """Operating mode of a provider."""
    LIVE_API = "live_api"      # Using live API (e.g., OpenAI, Gemini)
    DEBUG_STUB = "debug_stub"  # Using local debug stubs
    DISABLED = "disabled"      # Provider not available


@dataclass
class ProviderStatus:
    """Status information for a provider."""
    mode: ProviderMode
    provider_name: str
    error_details: Optional[str] = None

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class AppStatus:
    """Top-level app status tracking."""
    embeddings_status: ProviderStatus
    llm_status: ProviderStatus

    def __init__(self):
        # Initialize with disabled state
        logger.info("Status: initializing app status. Purpose: track provider availability and modes.")
        self.embeddings_status = ProviderStatus(
            mode=ProviderMode.DISABLED,
            provider_name="None"
        )
        self.llm_status = ProviderStatus(
            mode=ProviderMode.DISABLED,
            provider_name="None"
        )
        logger.info("Status: initialized to disabled state. Next step: providers will update their status during setup.")

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        logger.info("Status: generating status report. Purpose: inform client of provider states.")
        status = {
            "embeddings": self.embeddings_status.to_dict(),
            "llm": self.llm_status.to_dict()
        }
        logger.info("Status: Embeddings=%s/%s, LLM=%s/%s", 
                   self.embeddings_status.provider_name, 
                   self.embeddings_status.mode,
                   self.llm_status.provider_name,
                   self.llm_status.mode)
        return status


# Global app status singleton
app_status = AppStatus()