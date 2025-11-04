"""verify_install.py

Run this script inside your activated environment to check that required
packages are importable and to print their versions (or error details).

Usage:
    conda activate agenticai
    python verify_install.py

The script exits with code 0 if all packages imported successfully, otherwise 1.
"""
import sys
import importlib

try:
    # Python 3.8+: importlib.metadata in stdlib
    import importlib.metadata as importlib_metadata
except Exception:
    import importlib_metadata

CHECKS = {
    # Core server/runtime
    "fastapi": {"modules": ["fastapi"], "dists": ["fastapi"]},
    "uvicorn": {"modules": ["uvicorn"], "dists": ["uvicorn"]},

    # Validation + typing
    "pydantic": {"modules": ["pydantic"], "dists": ["pydantic"]},
    "pydantic-core": {"modules": ["pydantic_core"], "dists": ["pydantic-core"]},

    # Embedding store / vector DB
    "chromadb": {"modules": ["chromadb"], "dists": ["chromadb"]},

    # Document processing
    "pypdf": {"modules": ["pypdf"], "dists": ["pypdf"]},
    "python-multipart": {"modules": ["multipart", "python_multipart"], "dists": ["python-multipart"]},
    "readability-lxml": {"modules": ["readability", "readability.readability"], "dists": ["readability-lxml"]},
    "beautifulsoup4": {"modules": ["bs4"], "dists": ["beautifulsoup4"]},
    "reportlab": {"modules": ["reportlab"], "dists": ["reportlab"]},

    # Networking / scraping
    "requests": {"modules": ["requests"], "dists": ["requests"]},

    # LangChain + community integrations
    "langchain": {"modules": ["langchain"], "dists": ["langchain"]},
    "langchain-community": {"modules": ["langchain_community", "langchain.community"], "dists": ["langchain-community"]},
    "langchain-openai": {"modules": ["langchain_openai", "langchain.openai"], "dists": ["langchain-openai"]},

    # LLM providers + helpers
    "openai": {"modules": ["openai"], "dists": ["openai"]},
    "google-generativeai": {"modules": ["google.generativeai"], "dists": ["google-generativeai"]},
    "google-auth": {"modules": ["google.auth"], "dists": ["google-auth"]},

    # Tokenizers / embeddings
    "tiktoken": {"modules": ["tiktoken"], "dists": ["tiktoken"]},

    # Utilities / config
    "python-dotenv": {"modules": ["dotenv"], "dists": ["python-dotenv", "dotenv"]},
    "numpy": {"modules": ["numpy"], "dists": ["numpy"]},

    # Optional UI
    "streamlit": {"modules": ["streamlit"], "dists": ["streamlit"]},

    # Keep a catch-all alias (duplicate of beautifulsoup4) for clarity in output
    "beautifulsoup4 (alias)": {"modules": ["bs4"], "dists": ["beautifulsoup4"]},
}

failures = []

print("Verifying installed packages and their versions:\n")
for name, info in CHECKS.items():
    modules = info.get("modules", [])
    dists = info.get("dists", [])

    imported = None
    version = None
    errors = []

    # Try imports first
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            imported = mod
            # try common version attributes
            version = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
            break
        except Exception as e:
            errors.append(f"import {mod_name!r} failed: {e}")

    # If import succeeded but no version, try metadata
    if imported is not None and not version:
        for dist_name in dists:
            try:
                version = importlib_metadata.version(dist_name)
                break
            except Exception:
                continue

    # If no import, try checking distribution metadata directly
    if imported is None:
        for dist_name in dists:
            try:
                version = importlib_metadata.version(dist_name)
                break
            except Exception as e:
                errors.append(f"metadata {dist_name!r} not found: {e}")

    if version:
        print(f"OK  - {name:20s} version: {version}")
    else:
        print(f"FAIL - {name:20s} - could not determine version or import. Details:")
        for e in errors:
            print(f"       - {e}")
        failures.append(name)

print("\nSummary:")
if failures:
    print(f"Some packages failed verification ({len(failures)}): {', '.join(failures)}")
    sys.exit(1)
else:
    print("All packages verified successfully.")
    sys.exit(0)
