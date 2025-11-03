"""Minimal test for Gemini integration.

This script verifies basic Gemini embedding and chat functionality to ensure
the integration is working correctly in the current environment.
"""
# import debugpy
# import pdb; pdb.set_trace()
import os
import sys
import logging
from typing import List
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger("test_gemini")

try:
    # Load .env so local GOOGLE_API_KEY is available when running under the .venv
    load_dotenv()
    # Import the adapter from the local project 'src' package to ensure we use
    # the project copy rather than any same-named package installed in the venv.
    from src.gemini import GeminiEmbeddings, GeminiChat, GeminiError
    logger.info("Successfully imported local Gemini adapter.")
except Exception as e:
    logger.error("Failed to import local Gemini adapter: %s", e)
    sys.exit(1)


def run_embeddings_test(texts: List[str]) -> bool:
    logger.info("Starting embeddings test (DEBUG=%s)", os.environ.get("DEBUG"))
    try:
        emb = GeminiEmbeddings()
        logger.info("Successfully created GeminiEmbeddings instance")
        vecs = emb.embed_documents(texts)
        logger.info("Received %d vectors; first vector length=%d", len(vecs), len(vecs[0]) if vecs else 0)

        qv = emb.embed_query("sample query")
        logger.info("Query vector length=%d", len(qv))

        # If in DEBUG mode, verify deterministic behavior
        if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
            vecs2 = emb.embed_documents(texts)
            if vecs != vecs2:
                logger.error("Stub embedder outputs are not deterministic")
                return False
            logger.info("Stub embeddings deterministic check passed")

        return True
    except GeminiError as ge:
        logger.error("GeminiError during embeddings test: %s", ge)
        return False
    except Exception as e:
        logger.error("Unexpected error during embeddings test: %s", e)
        return False


def run_chat_test() -> bool:
    logger.info("Starting chat test (DEBUG=%s)", os.environ.get("DEBUG"))
    try:
        chat = GeminiChat()
        resp1 = chat.generate("Say hello in a creative way.")
        logger.info("Chat response (truncated): %s", resp1[:200])

        # In debug mode verify determinism
        if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
            resp2 = chat.generate("Say hello in a creative way.")
            if resp1 != resp2:
                logger.error("Stub chat responses are not deterministic")
                return False
            logger.info("Stub chat determinism check passed")

        return True
    except GeminiError as ge:
        logger.error("GeminiError during chat test: %s", ge)
        return False
    except Exception as e:
        logger.error("Unexpected error during chat test: %s", e)
        return False


def main():
    # Run tests with a small set of texts
    texts = ["Hello world", "Another document", "Hello world"]

    # Run embeddings test
    ok = run_embeddings_test(texts)
    if not ok:
        logger.error("Embeddings test FAILED") 
        return 1

    # Run chat test
    ok = run_chat_test()
    if not ok:
        logger.error("Chat test FAILED")
        return 1

    logger.info("All tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())