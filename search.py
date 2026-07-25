"""V2 Search Engine — numpy cosine + BM25 with RRF fusion + FlashRank reranking."""
import re
import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from config import INDEX_DIR, RRF_K, DEFAULT_LIMIT, USE_RERANKING, EMBED_DIM

# Stopwords (minimal set for technical content)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "and", "but", "or", "nor",
    "not", "so", "very", "just", "than", "too", "also", "about", "up",
})

def _tokenize(text: str) -> list[str]:
    """Lowercase + split. Simple baseline for BM25."""
    return text.lower().split()


class SearchEngine:
    def __init__(self):
        self._reranker = None
        self._load()

    def _load(self):
        # Vault
        vp = INDEX_DIR / "vault_vectors.npy"
        if vp.exists():
            self.vault_vecs = np.load(str(vp))
            self.vault_vecs = self.vault_vecs / np.linalg.norm(self.vault_vecs, axis=1, keepdims=True)
            with open(INDEX_DIR / "vault_meta.json", "r", encoding="utf-8") as f:
                self.vault_meta = json.load(f)
            self.vault_bm25 = BM25Okapi([_tokenize(m["text"]) for m in self.vault_meta])
        else:
            self.vault_vecs = np.empty((0, EMBED_DIM))
            self.vault_meta = []
            self.vault_bm25 = None

        # Transcripts
        tp = INDEX_DIR / "transcript_vectors.npy"
        if tp.exists():
            self.trans_vecs = np.load(str(tp))
            self.trans_vecs = self.trans_vecs / np.linalg.norm(self.trans_vecs, axis=1, keepdims=True)
            with open(INDEX_DIR / "transcript_meta.json", "r", encoding="utf-8") as f:
                self.trans_meta = json.load(f)
            self.trans_bm25 = BM25Okapi([_tokenize(m["text"]) for m in self.trans_meta])
        else:
            self.trans_vecs = np.empty((0, EMBED_DIM))
            self.trans_meta = []
            self.trans_bm25 = None

        # Video index
        vip = INDEX_DIR / "video_index.json"
        if vip.exists():
            with open(vip, "r", encoding="utf-8") as f:
                self.video_index = json.load(f)
        else:
            self.video_index = []

        # Memory
        mp = INDEX_DIR / "memory_vectors.npy"
        if mp.exists():
            self.mem_vecs = np.load(str(mp))
            self.mem_vecs = self.mem_vecs / np.linalg.norm(self.mem_vecs, axis=1, keepdims=True)
            with open(INDEX_DIR / "memory_meta.json", "r", encoding="utf-8") as f:
                self.mem_meta = json.load(f)
            self.mem_bm25 = BM25Okapi([_tokenize(m["text"]) for m in self.mem_meta])
        else:
            self.mem_vecs = np.empty((0, EMBED_DIM))
            self.mem_meta = []
            self.mem_bm25 = None

        print(f"Loaded {len(self.vault_meta)} vault chunks, {len(self.trans_meta)} transcript chunks, {len(self.mem_meta)} memory chunks, {len(self.video_index)} videos")

    def _get_reranker(self):
        """Lazy-load FlashRank reranker on first use."""
        if self._reranker is None:
            from flashrank import Ranker, RerankRequest
            self._reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            self._rerank_request_cls = RerankRequest
        return self._reranker, self._rerank_request_cls

    def _rerank(self, query_text, results):
        """Rerank results using FlashRank cross-encoder. Returns results sorted by reranker score."""
        if len(results) <= 1:
            return results
        ranker, RerankRequest = self._get_reranker()
        passages = [{"id": i, "text": r.get("text", "")} for i, r in enumerate(results)]
        rerank_req = RerankRequest(query=query_text, passages=passages)
        ranked = ranker.rerank(rerank_req)
        reranked = []
        for item in ranked:
            idx = item["id"]
            results[idx]["rerank_score"] = item["score"]
            results[idx]["score"] = item["score"]
            reranked.append(results[idx])
        return reranked

    def _rrf_search(self, query_vec, vectors, meta, bm25, query_text, limit, filters=None):
        """Hybrid search: dense cosine + BM25, fused with RRF."""
        if len(meta) == 0 or limit <= 0:
            return []

        # Dense scores
        q = np.array(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return []
        q = q / norm
        dense_scores = vectors @ q

        # BM25 scores
        bm25_scores = bm25.get_scores(_tokenize(query_text))

        # RRF fusion
        dense_ranks = np.argsort(-dense_scores)
        bm25_ranks = np.argsort(-bm25_scores)

        rrf_scores = {}
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        rerank_cap = max(limit, 20) if USE_RERANKING else limit
        for idx in sorted_indices:
            m = meta[idx]
            # Apply filters
            if filters:
                skip = False
                for k, v in filters.items():
                    if v and m.get(k, "") != v:
                        skip = True
                        break
                if skip:
                    continue
            results.append({"score": rrf_scores[idx], "raw_rrf_score": rrf_scores[idx], **m})
            if len(results) >= rerank_cap:
                break

        # Rerank top candidates with FlashRank cross-encoder
        if USE_RERANKING and len(results) > 1:
            results = self._rerank(query_text, results)

        return results[:limit]

    def search_vault(self, query_vec, query_text, limit=DEFAULT_LIMIT, note_type=None):
        filters = {}
        if note_type and note_type != "all":
            filters["category"] = note_type
        return self._rrf_search(query_vec, self.vault_vecs, self.vault_meta, self.vault_bm25, query_text, limit, filters or None)

    def search_transcripts(self, query_vec, query_text, limit=DEFAULT_LIMIT):
        return self._rrf_search(query_vec, self.trans_vecs, self.trans_meta, self.trans_bm25, query_text, limit)

    def search_memory(self, query_vec, query_text, limit=DEFAULT_LIMIT, mem_type=None):
        filters = {}
        if mem_type and mem_type != "all":
            filters["type"] = mem_type
        return self._rrf_search(query_vec, self.mem_vecs, self.mem_meta, self.mem_bm25, query_text, limit, filters or None)

    def search_all(self, query_vec, query_text, limit=DEFAULT_LIMIT):
        vault = self.search_vault(query_vec, query_text, limit)
        trans = self.search_transcripts(query_vec, query_text, limit)
        mem = self.search_memory(query_vec, query_text, limit)

        # Normalize scores within each collection to [0, 1].
        # RRF scores from different-sized collections aren't directly comparable.
        # After FlashRank, scores are already in [0,1] range — use those directly.
        def _normalize(results):
            if not results:
                return results
            scores = [r["score"] for r in results]
            lo, hi = min(scores), max(scores)
            if hi - lo < 1e-10:
                for r in results:
                    r["normalized_score"] = r["score"]
                return results
            for r in results:
                r["normalized_score"] = (r["score"] - lo) / (hi - lo)
            return results

        vault_n = _normalize(vault)
        trans_n = _normalize(trans)
        mem_n = _normalize(mem)

        combined = sorted(
            vault_n + trans_n + mem_n,
            key=lambda x: x.get("normalized_score", x["score"]),
            reverse=True,
        )

        # Low confidence marker: if top result has low normalized score
        if combined and combined[0].get("normalized_score", 0) < 0.3:
            for r in combined:
                r["low_confidence"] = True

        return combined[:limit]

    def get_video_summary(self, video_id):
        for v in self.video_index:
            if v.get("video_id") == video_id:
                return v
        return None

    def list_videos(self, limit=20):
        return self.video_index[:limit]


def format_results(results):
    if not results:
        return "No results found."
    out = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        if r.get("source") == "vault":
            cite = f"[vault] {r.get('file_path','?')}"
            if r.get("header"):
                cite += f" > {r['header']}"
        elif r.get("source") == "memory":
            cite = f"[memory] {r.get('file_path','?')}"
            if r.get("title"):
                cite = f"[memory] {r.get('title','?')} ({r.get('file_path','')})"
            if r.get("header"):
                cite += f" > {r['header']}"
        elif r.get("video_id"):
            cite = f"[yt] {r.get('title','?')}"
            ts = r.get("timestamp_start", 0)
            if ts:
                cite += f" ({int(ts//60)}:{int(ts%60):02d})"
            if r.get("url"):
                cite += f"\n   {r['url']}"
        else:
            cite = "?"
        text = r.get("text", "")[:1200]
        if len(r.get("text", "")) > 1200:
            text += "..."
        out.append(f"[{i}] ({score:.4f}) {cite}\n{text}\n")
    return "\n".join(out)
