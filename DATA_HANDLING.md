# Data handling and public-release boundary

ContextRAG is source code, not a public corpus.

## Never committed

- Raw Markdown notes, agent-memory files, or transcript files
- Embeddings, vector indexes, result metadata, and content hashes
- Local MCP configuration, environment files, API keys, or editor state
- Backup directories and virtual environments

These files are ignored because derived metadata and embeddings can still
reveal source structure or content. Keeping an index out of Git is therefore
not merely a repository-size decision.

## Safe public material

The repository contains only a small synthetic Markdown example and matching
evaluation queries. They demonstrate the input format and test the code path;
they are not a representative retrieval benchmark.

## Operating rule

Before contributing a new fixture, verify that it is synthetic, public-domain,
or licensed for redistribution. Keep any personal or client data entirely
outside this repository.
