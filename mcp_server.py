"""MCP server for ContextRAG's local hybrid retrieval index."""
import sys
import os
import hashlib
import json
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from search import SearchEngine, format_results
from embed import embed, embed_query, embed_query_with_hyde, ENABLE_HYDE, OllamaNotRunningError
from config import VAULT_DIR, INDEX_DIR

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mcp_server")

mcp = FastMCP(
    "context-rag",
    instructions=(
        "Search explicitly configured local notes, transcripts, and agent-memory "
        "files using dense + BM25 hybrid retrieval with reciprocal-rank fusion."
    ),
)

_engine = None

def _vault_file_hashes() -> dict[str, str]:
    """Compute per-file content hashes for incremental indexing."""
    EXCLUDE = {".obsidian", "Templates", "Archive"}
    hashes = {}
    for p in sorted(VAULT_DIR.rglob("*.md")):
        rel = p.relative_to(VAULT_DIR)
        if any(part in EXCLUDE for part in rel.parts):
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        hashes[str(rel)] = hashlib.md5(content.encode()).hexdigest()
    return hashes

def _check_reindex() -> bool:
    """Reindex vault if files changed since last build. Returns True if rebuilt."""
    hash_path = INDEX_DIR / ".vault_file_hashes.json"
    current = _vault_file_hashes()
    stored = {}

    if hash_path.exists():
        with open(hash_path, "r") as f:
            stored = json.load(f)
        if stored == current:
            return False

    # Find changed files
    added = [k for k in current if k not in stored]
    removed = [k for k in stored if k not in current]
    changed = [k for k in current if k in stored and current[k] != stored[k]]

    if not added and not removed and not changed:
        return False

    log.warning("Vault changed — +%d -%d ~%d files, rebuilding index",
                len(added), len(removed), len(changed))
    try:
        from run_all import build_vault_index
        build_vault_index()
    except Exception as e:
        log.error("Vault rebuild failed: %s", e)
        return False

    with open(hash_path, "w") as f:
        json.dump(current, f)
    return True

def _background_reindex():
    """Run reindex in background thread. Reload engine if rebuilt."""
    global _engine
    try:
        if _check_reindex():
            _engine = SearchEngine()
            log.warning("Vault reindex complete — engine reloaded")
    except Exception as e:
        log.error("Background reindex failed: %s", e)

def _get_query_vec(query: str) -> list[float]:
    """Get query embedding, optionally with HyDE fusion."""
    if ENABLE_HYDE:
        return list(embed_query_with_hyde(query))
    return list(embed_query(query))

def get_engine():
    global _engine
    if _engine is None:
        _engine = SearchEngine()
        # Check for vault changes in background (non-blocking)
        import threading
        threading.Thread(target=_background_reindex, daemon=True).start()
    return _engine


@mcp.tool()
def search_vault_notes(query: str, note_type: str = "all", limit: int = 5) -> str:
    """Search an explicitly configured Markdown collection.

    Args:
        query: What to search for (semantic + keyword search)
        note_type: Filter by type — "idea", "research", "project", "contact", "business", "creative", or "all"
        limit: Number of results to return (default 5)

    Returns:
        Ranked search results with text excerpts, scores, and file citations.
    """
    engine = get_engine()
    try:
        query_vec = _get_query_vec(query)
    except OllamaNotRunningError:
        return "Error: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"Error: Embedding failed — {e}"
    results = engine.search_vault(query_vec, query, limit=limit, note_type=note_type)
    if not results:
        return f"No vault notes found matching: {query}"
    return format_results(results)


@mcp.tool()
def search_youtube_transcripts(query: str, limit: int = 5) -> str:
    """Search YouTube AI engineering transcripts for talks, tutorials, and insights.

    Args:
        query: What to search for (e.g., "MCP apps", "agent swarms", "context windows")
        limit: Number of results to return (default 5)

    Returns:
        Ranked search results with text excerpts, video titles, timestamps, and direct YouTube links.
    """
    engine = get_engine()
    try:
        query_vec = _get_query_vec(query)
    except OllamaNotRunningError:
        return "Error: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"Error: Embedding failed — {e}"
    results = engine.search_transcripts(query_vec, query, limit=limit)
    if not results:
        return f"No transcript results found matching: {query}"
    return format_results(results)


@mcp.tool()
def search_memory(query: str, mem_type: str = "all", limit: int = 5) -> str:
    """Search explicitly configured local agent-memory files.

    Args:
        query: What to search for (semantic + keyword search)
        mem_type: Filter by type — "feedback", "reference", "project", "user", or "all"
        limit: Number of results to return (default 5)

    Returns:
        Ranked search results with text excerpts, scores, and file citations.
    """
    engine = get_engine()
    try:
        query_vec = _get_query_vec(query)
    except OllamaNotRunningError:
        return "Error: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"Error: Embedding failed — {e}"
    results = engine.search_memory(query_vec, query, limit=limit, mem_type=mem_type)
    if not results:
        return f"No memory results found matching: {query}"
    return format_results(results)


@mcp.tool()
def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Search all explicitly configured local source collections at once.

    Args:
        query: What to search for
        limit: Number of results to return (default 5)

    Returns:
        Combined results from vault and transcripts, sorted by relevance.
    """
    engine = get_engine()
    try:
        query_vec = _get_query_vec(query)
    except OllamaNotRunningError:
        return "Error: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"Error: Embedding failed — {e}"
    results = engine.search_all(query_vec, query, limit=limit)
    if not results:
        return f"No results found matching: {query}"
    return format_results(results)


@mcp.tool()
def get_video_summary(video_id: str) -> str:
    """Get detailed information about a specific indexed YouTube video.

    Args:
        video_id: The YouTube video ID (e.g., "LMbeDEQO6QM")

    Returns:
        Video metadata including title, URL, language, and chunk count.
    """
    engine = get_engine()
    video = engine.get_video_summary(video_id)
    if not video:
        return f"Video {video_id} not found in the index."
    return (
        f"**{video.get('title', video_id)}**\n"
        f"- URL: {video.get('url', 'N/A')}\n"
        f"- Channel: {video.get('channel', 'AI Engineer')}\n"
        f"- Language: {video.get('language', 'N/A')}\n"
        f"- Segments: {video.get('segment_count', 0)}\n"
        f"- Chunks indexed: {video.get('chunk_count', 0)}"
    )


@mcp.tool()
def list_recent_videos(limit: int = 20) -> str:
    """List recently indexed YouTube videos with titles and topics.

    Args:
        limit: Number of videos to list (default 20)

    Returns:
        List of indexed videos with video ID, title, and chunk count.
    """
    engine = get_engine()
    videos = engine.list_videos(limit=limit)
    if not videos:
        return "No videos indexed yet."
    out = [f"## Indexed Videos ({len(videos)} shown)\n"]
    for i, v in enumerate(videos, 1):
        out.append(f"{i}. **{v.get('title', v.get('video_id', '?'))}**")
        out.append(f"   - ID: `{v.get('video_id', '?')}`")
        out.append(f"   - Chunks: {v.get('chunk_count', 0)} | Segments: {v.get('segment_count', 0)}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run(transport="stdio")
