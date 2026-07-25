"""
Eval harness for RAG knowledge base.
Measures recall, top-k quality, latency, and prints result examples.

Usage:
    python eval/eval_search.py                # Run all queries
    python eval/eval_search.py --category rag # Run only RAG queries
    python eval/eval_search.py --limit 5      # Top-5 results per query
    python eval/eval_search.py --json         # Output as JSON
"""
import sys
import os
import time
import json
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search import SearchEngine, format_results
from embed import embed, embed_query


def load_queries(category=None):
    query_file = Path(__file__).parent / "queries.yaml"
    with open(query_file, "r") as f:
        data = yaml.safe_load(f)
    queries = data["queries"]
    if category:
        queries = [q for q in queries if q.get("category") == category]
    return queries


def evaluate_query(engine, query_entry, limit=5):
    """Evaluate a single query. Returns metrics dict."""
    query_text = query_entry["query"]
    expected_keywords = [k.lower() for k in query_entry.get("expected_keywords", [])]

    # Embed (cached)
    t0 = time.time()
    query_vec = list(embed_query(query_text))
    embed_time = time.time() - t0

    # Search all collections
    t1 = time.time()
    results = engine.search_all(query_vec, query_text, limit=limit)
    search_time = time.time() - t1

    total_time = embed_time + search_time

    # Keyword recall: fraction of expected keywords found in top-k result texts
    all_text = " ".join(r.get("text", "").lower() for r in results)
    keywords_found = [kw for kw in expected_keywords if kw in all_text]
    keyword_recall = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0

    # Top-1 has keyword match?
    top1_text = results[0].get("text", "").lower() if results else ""
    top1_hit = any(kw in top1_text for kw in expected_keywords) if expected_keywords else True

    # Sources represented
    sources = set(r.get("source", "?") for r in results)

    # Top result score
    top_score = results[0].get("score", 0) if results else 0

    return {
        "query": query_text,
        "category": query_entry.get("category", "?"),
        "keyword_recall": round(keyword_recall, 3),
        "top1_hit": top1_hit,
        "top_score": round(top_score, 4),
        "num_results": len(results),
        "sources": sorted(sources),
        "embed_ms": round(embed_time * 1000),
        "search_ms": round(search_time * 1000),
        "total_ms": round(total_time * 1000),
        "results": [
            {
                "rank": i + 1,
                "score": round(r.get("score", 0), 4),
                "source": r.get("source", "?"),
                "title": r.get("title", "?")[:80],
                "excerpt": r.get("text", "")[:200],
            }
            for i, r in enumerate(results)
        ],
    }


def print_result(eval_result, verbose=False):
    """Pretty-print a single eval result."""
    q = eval_result
    recall_bar = "=" * int(q["keyword_recall"] * 20) + "-" * (20 - int(q["keyword_recall"] * 20))
    top1 = "OK" if q["top1_hit"] else "MISS"

    print(f"\n{'─' * 70}")
    print(f"  [{q['category']}] {q['query']}")
    print(f"  Recall: [{recall_bar}] {q['keyword_recall']:.0%}  Top1: {top1}  Score: {q['top_score']:.4f}")
    print(f"  Latency: {q['total_ms']}ms (embed: {q['embed_ms']}ms, search: {q['search_ms']}ms)")
    print(f"  Sources: {', '.join(q['sources'])}")

    if verbose:
        for r in q["results"]:
            print(f"    [{r['rank']}] ({r['score']:.4f}) [{r['source']}] {r['title']}")
            print(f"        {r['excerpt'][:150]}...")


def run_eval(args):
    queries = load_queries(category=args.category)
    if not queries:
        print("No queries found.")
        return

    print(f"Loading search engine...")
    engine = SearchEngine()

    print(f"Running {len(queries)} queries (limit={args.limit})...")
    results = []
    for i, q in enumerate(queries):
        print(f"  [{i + 1}/{len(queries)}] {q['query'][:60]}...", end=" ", flush=True)
        r = evaluate_query(engine, q, limit=args.limit)
        results.append(r)
        recall_str = f"{r['keyword_recall']:.0%}"
        print(f"recall={recall_str} top1={'OK' if r['top1_hit'] else 'MISS'} {r['total_ms']}ms")

    # Aggregate
    avg_recall = sum(r["keyword_recall"] for r in results) / len(results)
    top1_rate = sum(1 for r in results if r["top1_hit"]) / len(results)
    avg_latency = sum(r["total_ms"] for r in results) / len(results)
    avg_embed = sum(r["embed_ms"] for r in results) / len(results)
    avg_search = sum(r["search_ms"] for r in results) / len(results)

    print(f"\n{'=' * 70}")
    print(f"  EVAL SUMMARY ({len(results)} queries)")
    print(f"{'=' * 70}")
    print(f"  Avg Keyword Recall:  {avg_recall:.1%}")
    print(f"  Top-1 Hit Rate:      {top1_rate:.1%}")
    print(f"  Avg Total Latency:   {avg_latency:.0f}ms")
    print(f"    Avg Embed:         {avg_embed:.0f}ms")
    print(f"    Avg Search:        {avg_search:.0f}ms")
    print(f"{'=' * 70}")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    if len(categories) > 1:
        print(f"\n  Per-Category Breakdown:")
        print(f"  {'Category':<12} {'Recall':>8} {'Top1':>8} {'Latency':>8}")
        print(f"  {'─' * 40}")
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            cat_recall = sum(r["keyword_recall"] for r in cat_results) / len(cat_results)
            cat_top1 = sum(1 for r in cat_results if r["top1_hit"]) / len(cat_results)
            cat_lat = sum(r["total_ms"] for r in cat_results) / len(cat_results)
            print(f"  {cat:<12} {cat_recall:>7.1%} {cat_top1:>7.1%} {cat_lat:>6.0f}ms")

    if args.json:
        out_path = Path(__file__).parent / "results.json"
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                return super().default(obj)
        with open(out_path, "w") as f:
            json.dump({
                "summary": {
                    "num_queries": len(results),
                    "avg_keyword_recall": round(avg_recall, 3),
                    "top1_hit_rate": round(top1_rate, 3),
                    "avg_total_ms": round(avg_latency),
                    "avg_embed_ms": round(avg_embed),
                    "avg_search_ms": round(avg_search),
                },
                "results": results,
            }, f, indent=2, cls=NumpyEncoder)
        print(f"\n  Results saved to {out_path}")

    if args.verbose:
        print(f"\n{'=' * 70}")
        print(f"  DETAILED RESULTS")
        print(f"{'=' * 70}")
        for r in results:
            print_result(r, verbose=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Knowledge Base Eval")
    parser.add_argument("--category", type=str, help="Filter by query category")
    parser.add_argument("--limit", type=int, default=5, help="Top-k results per query")
    parser.add_argument("--json", action="store_true", help="Save results to JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    args = parser.parse_args()
    run_eval(args)
