"""Embedding module — Ollama nomic-embed-text, loaded once, reused.
Uses Matryoshka truncation to EMBED_DIM (512)."""
import os
from functools import lru_cache
import numpy as np
import ollama
from config import EMBED_DIM

ENABLE_HYDE = os.environ.get("RAG_ENABLE_HYDE", "0") == "1"

_client = None

class OllamaNotRunningError(Exception):
    """Raised when Ollama is not reachable."""
    pass

def get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client()  # Connects to localhost:11434
    return _client

def embed(texts: list[str], max_chars: int = 2000, task_type: str = "") -> list[list[float]]:
    """Embed texts at EMBED_DIM using Ollama + Matryoshka truncation."""
    client = get_client()
    prefix = f"{task_type}: " if task_type else ""
    safe_texts = [prefix + (t[:max_chars] if len(t) > max_chars else t) for t in texts]

    try:
        response = client.embed(model="nomic-embed-text", input=safe_texts)
    except Exception as e:
        if "connect" in str(e).lower() or "connection" in str(e).lower():
            raise OllamaNotRunningError(
                "Ollama is not running. Start it with: ollama serve"
            ) from e
        raise

    embeddings = response.embeddings
    results = []
    for vec in embeddings:
        # Matryoshka: truncate from 768 to 512
        v = list(vec[:EMBED_DIM])
        results.append(v)
    return results

@lru_cache(maxsize=2000)
def embed_query(query: str) -> tuple[float, ...]:
    """Cached query embedding. Returns tuple for hashability."""
    return tuple(embed([query], task_type="")[0])

def _generate_hypothetical(query: str) -> str:
    """Generate a hypothetical answer document for HyDE."""
    client = get_client()
    prompt = f"Write a short technical paragraph (3-5 sentences) that would answer this question: {query}"
    response = client.generate(model="qwen2.5:0.5b", prompt=prompt)
    return response.get("response", "")

def embed_query_with_hyde(query: str) -> tuple[float, ...]:
    """HyDE-fused query embedding: average of raw query and hypothetical document embeddings."""
    raw_vec = np.array(embed([query], task_type="search_query")[0])
    hypothetical = _generate_hypothetical(query)
    if hypothetical.strip():
        hyde_vec = np.array(embed([hypothetical], task_type="search_document")[0])
        # Fusion: average raw query and HyDE vectors
        fused = (raw_vec + hyde_vec) / 2.0
        norm = np.linalg.norm(fused)
        if norm > 1e-10:
            fused = fused / norm
        return tuple(fused.tolist())
    return tuple(raw_vec.tolist())
