"""Offline tests for public ContextRAG behavior.

These tests use a temporary synthetic index; no personal corpus, model server,
or network connection is required.
"""
from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
import search
from run_all import _chunk_markdown, _parse_frontmatter


class ContextRAGCoreTests(unittest.TestCase):
    def test_explicit_paths_do_not_discover_home_directories(self):
        self.assertEqual(config._explicit_paths(None), [])
        self.assertEqual(config._explicit_paths(""), [])
        paths = config._explicit_paths("/tmp/one:/tmp/two")
        self.assertEqual(paths, [Path("/tmp/one"), Path("/tmp/two")])

    def test_frontmatter_and_header_chunking(self):
        meta, body = _parse_frontmatter("---\ncategory: retrieval\n---\n# One\nfirst\n# Two\nsecond")
        self.assertEqual(meta["category"], "retrieval")
        self.assertEqual(len(_chunk_markdown(body)), 2)

    def test_hybrid_search_returns_matching_synthetic_document(self):
        previous_index = search.INDEX_DIR
        previous_reranking = search.USE_RERANKING
        try:
            with tempfile.TemporaryDirectory() as tmp:
                search.INDEX_DIR = Path(tmp)
                search.USE_RERANKING = False
                vectors = np.zeros((2, 512), dtype=np.float32)
                vectors[0, 0] = 1.0
                vectors[1, 1] = 1.0
                np.save(search.INDEX_DIR / "vault_vectors.npy", vectors)
                metadata = [
                    {"source": "vault", "file_path": "latency.md", "text": "Token latency matters."},
                    {"source": "vault", "file_path": "storage.md", "text": "Storage durability matters."},
                ]
                (search.INDEX_DIR / "vault_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
                engine = search.SearchEngine()
                query = [1.0] + [0.0] * 511
                results = engine.search_vault(query, "token latency", limit=1)
                self.assertEqual(results[0]["file_path"], "latency.md")
        finally:
            search.INDEX_DIR = previous_index
            search.USE_RERANKING = previous_reranking

    def test_result_format_keeps_local_citation(self):
        rendered = search.format_results([
            {"source": "vault", "file_path": "notes/retrieval.md", "header": "Hybrid", "score": 0.9, "text": "Hybrid retrieval"}
        ])
        self.assertIn("[vault] notes/retrieval.md > Hybrid", rendered)


if __name__ == "__main__":
    unittest.main()
