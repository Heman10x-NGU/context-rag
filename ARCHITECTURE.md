# Architecture

ContextRAG is a local retrieval layer. It does not generate final answers or
run an autonomous retrieval agent. An MCP client or application supplies the
generation and orchestration layer.

```text
Explicit local source roots
        |
        v
Markdown / transcript ingestion
        |
        +--> Ollama embeddings (nomic-embed-text, 512 dimensions)
        +--> BM25 lexical index
        |
        v
Dense cosine ranking + BM25 ranking
        |
        v
Reciprocal-rank fusion
        |
        v
Optional FlashRank cross-encoder reranking
        |
        v
MCP tools or Python API return cited passages
```

## Source boundaries

- `RAG_VAULT_DIR` selects one Markdown collection.
- `RAG_MEMORY_DIRS` explicitly selects zero or more memory roots.
- `transcripts/` is opt-in local input for transcript JSON files.
- `index/` contains derived vectors and metadata and is always ignored by Git.

The project never searches a home directory or agent installation for files.

## Retrieval strategy

For each configured collection, ContextRAG calculates dense cosine similarity
and BM25 lexical scores. Reciprocal-rank fusion merges their rank order. When
enabled, FlashRank reranks the top candidates with a compact cross-encoder.
Results carry source paths and, for transcripts, timestamps/URLs.

Cross-collection scores are normalized before the final top-k selection. This
is a practical local retrieval design, not a claim of a universal benchmark.
