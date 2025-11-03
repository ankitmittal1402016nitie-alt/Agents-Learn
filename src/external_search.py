"""
Simple external search adapters.

Implements a small interface used by the app to fetch external snippets for queries.
Supports SerpAPI if SERPAPI_API_KEY is set; otherwise a stub that returns no results.

Return format for `search(query, k)` is a list of dicts with keys: title, snippet, link.
"""
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ExternalSearch:
    """Base interface."""
    def search(self, query: str, k: int = 3) -> List[Dict]:
        raise NotImplementedError()


class StubExternalSearch(ExternalSearch):
    def search(self, query: str, k: int = 3) -> List[Dict]:
        logger.info("StubExternalSearch: no external search provider configured; returning empty results.")
        return []


class SerpAPIExternalSearch(ExternalSearch):
    def __init__(self, api_key: str):
        try:
            import requests
        except Exception:
            raise RuntimeError("SerpAPIExternalSearch requires the 'requests' package. Install with 'pip install requests'.")
        self.requests = requests
        self.api_key = api_key
        self.endpoint = "https://serpapi.com/search"

    def search(self, query: str, k: int = 3) -> List[Dict]:
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": k,
        }
        try:
            resp = self.requests.get(self.endpoint, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            # SerpAPI returns 'organic_results' list when using google engine
            for item in data.get("organic_results", [])[:k]:
                title = item.get("title") or ""
                snippet = item.get("snippet") or item.get("rich_snippet", {}).get("top", {}).get("detected_extensions", "") or ""
                link = item.get("link") or item.get("source") or ""
                results.append({"title": title, "snippet": snippet, "link": link})
            return results
        except Exception as e:
            logger.warning("SerpAPIExternalSearch failed: %s", str(e))
            return []
