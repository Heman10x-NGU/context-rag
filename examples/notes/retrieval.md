---
category: architecture
tags: [retrieval, hybrid-search]
---

# Hybrid retrieval

Dense retrieval finds semantically related passages. BM25 preserves exact
technical terms and identifiers. Reciprocal-rank fusion combines their rank
positions before a cross-encoder reranks the best candidates.
