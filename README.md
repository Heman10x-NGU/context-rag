# ContextRAG

**Local-first hybrid retrieval for agent context.**

ContextRAG indexes Markdown notes, optional transcript JSON, and explicitly
selected agent-memory folders on your machine. It combines dense vector search
with BM25, fuses ranks with reciprocal-rank fusion, optionally reranks with a
small cross-encoder, and exposes the result through MCP.

It is deliberately **not** described as agentic RAG: it does not plan,
autonomously choose tools, or generate an answer. It is the retrieval layer an
agent or application can call for grounded context.

## Why it exists

Local AI applications need useful context without uploading a personal vault,
agent memories, or raw indexes to a hosted vector database. ContextRAG keeps
the source corpus and derived vectors local, returns cited passages, and gives
MCP clients one retrieval interface across several local collections.

## What it includes

- Dense cosine retrieval using local Ollama embeddings (`nomic-embed-text`)
- BM25 lexical retrieval for exact technical terms and identifiers
- Reciprocal-rank fusion across dense and lexical rankings
- Optional FlashRank cross-encoder reranking
- Optional HyDE-style hypothetical-document query expansion
- Incremental vault reindexing based on content hashes
- MCP tools for collection-specific and cross-collection search
- Explicit data roots; no home-directory, editor, or agent-memory discovery

## Architecture

```text
local files -> embeddings + BM25 -> rank fusion -> optional reranker -> cited MCP result
```

Read the fuller design in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start

Requirements: Python 3.11+ and [Ollama](https://ollama.com/).

```bash
git clone https://github.com/Heman10x-NGU/context-rag.git
cd context-rag
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull nomic-embed-text

# Build the included, synthetic example corpus.
RAG_VAULT_DIR="$PWD/examples/notes" .venv/bin/python run_all.py --skip-transcripts --skip-memory
```

To index your own material, copy `.env.example` values into your shell. Source
paths are always opt-in:

```bash
export RAG_VAULT_DIR="/absolute/path/to/markdown-notes"
export RAG_MEMORY_DIRS="/absolute/path/to/agent-memory:/another/allowed/root"
.venv/bin/python run_all.py --skip-transcripts
```

Generated `index/`, `transcripts/`, `.mcp.json`, environment files, and local
source material are ignored by Git. Do not commit data you do not have the
right to publish.

## MCP server

After building an index, add a server entry to your MCP client configuration:

```json
{
  "mcpServers": {
    "context-rag": {
      "command": "/absolute/path/to/context-rag/.venv/bin/python",
      "args": ["/absolute/path/to/context-rag/mcp_server.py"]
    }
  }
}
```

The server provides:

- `search_vault_notes`
- `search_youtube_transcripts`
- `search_memory`
- `search_knowledge_base`
- `get_video_summary`
- `list_recent_videos`

## Evaluation and tests

The repository includes a small synthetic example and offline unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

The evaluation runner measures keyword recall, top-1 keyword presence, and
latency for the corpus you choose:

```bash
.venv/bin/python eval/eval_search.py --limit 5
```

Those metrics are corpus-specific. The repository intentionally does not ship
the prior private corpus, derived index, or its old evaluation results.

## Limitations

- Retrieval quality depends on source quality, chunking, and the embedding
  model; it is not an answer-quality guarantee.
- FlashRank is loaded lazily and may download its compact model on first use.
- The project is designed for a local single-user corpus, not multi-tenant
  hosted search.
- Transcript ingestion expects local JSON input. This repository does not ship
  third-party transcript corpora.

## License

[MIT](LICENSE)
