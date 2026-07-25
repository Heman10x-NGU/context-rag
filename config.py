"""Configuration for ContextRAG.

All source roots are explicit. The project deliberately never scans a user's
home directory, agent installations, or editor configuration for data.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def _explicit_paths(value: str | None) -> list[Path]:
    """Parse colon-separated existing source roots from an opt-in variable."""
    if not value:
        return []
    return [Path(raw).expanduser() for raw in value.split(":") if raw.strip()]


# A small synthetic corpus is the safe default. Configure real local sources
# through .env or the shell; those sources and their derived index stay ignored.
VAULT_DIR = Path(
    os.environ.get("RAG_VAULT_DIR", str(PROJECT_DIR / "examples" / "notes"))
).expanduser()
MEMORY_DIRS = _explicit_paths(os.environ.get("RAG_MEMORY_DIRS"))
MEMORY_DIR = MEMORY_DIRS[0] if MEMORY_DIRS else None

TRANSCRIPTS_DIR = PROJECT_DIR / "transcripts"
INDEX_DIR = PROJECT_DIR / "index"

# Ollama nomic-embed-text has Matryoshka representations. We retain 512 dims.
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 512

# Retrieval settings.
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 15
DEFAULT_LIMIT = 5
USE_RERANKING = True

# Ingestion settings.
BATCH_SIZE = 100
