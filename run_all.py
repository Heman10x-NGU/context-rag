"""Build a local ContextRAG index from explicitly configured sources.

Usage: python run_all.py [--skip-vault] [--skip-transcripts] [--skip-memory]
"""
import sys
import os
import re
import json
import gc
import argparse
import yaml
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from config import VAULT_DIR, TRANSCRIPTS_DIR, INDEX_DIR, MEMORY_DIRS, EMBED_DIM, BATCH_SIZE
from embed import embed


# ── Shared transcript chunking ─────────────────────────────────────

def _merge_segments(segments, target_chars=1200):
    """Merge transcript segments into ~300-token chunks at sentence boundaries."""
    if not segments:
        return []
    merged = []
    texts, start, end, chars = [], None, None, 0
    for seg in segments:
        t = seg.get("text", "").strip()
        if not t:
            continue
        if start is None:
            start = seg["start"]
        texts.append(t)
        end = seg["start"] + seg.get("duration", 0)
        chars += len(t)
        if chars >= target_chars and (t.endswith(".") or t.endswith("?") or t.endswith("!")):
            merged.append({"text": " ".join(texts), "start": start, "end": end})
            texts, start, chars = [], None, 0
    if texts:
        merged.append({"text": " ".join(texts), "start": start, "end": end})
    return merged


# ── Phase 2: Build vault index ─────────────────────────────────────

def build_vault_index():
    """Walk vault, chunk, embed, save to numpy + JSON."""
    import yaml as _yaml

    EXCLUDE = {".obsidian", "Templates", "Archive"}

    print(f"\n{'='*60}")
    print("Phase 2: BUILD VAULT INDEX")
    print(f"{'='*60}")

    if not VAULT_DIR.exists():
        print(f"Vault not found: {VAULT_DIR}")
        return

    # Collect files
    files = []
    for p in VAULT_DIR.rglob("*.md"):
        rel = p.relative_to(VAULT_DIR)
        if any(part in EXCLUDE for part in rel.parts):
            continue
        files.append(p)
    files.sort()
    print(f"Found {len(files)} vault files")

    # Parse + chunk
    all_texts = []
    all_meta = []

    for filepath in files:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(content)
        if not body.strip() or len(body.strip()) < 30:
            continue

        rel_path = str(filepath.relative_to(VAULT_DIR))
        parts = filepath.relative_to(VAULT_DIR).parts
        folder = parts[0] if len(parts) > 1 else "Root"
        title = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", filepath.stem).replace("-", " ").replace("_", " ").strip().title()

        chunks = _chunk_markdown(body)
        for chunk in chunks:
            if len(chunk["text"].strip()) < 20:
                continue
            all_texts.append(chunk["text"])
            all_meta.append({
                "source": "vault",
                "file_path": rel_path,
                "title": title,
                "folder": folder,
                "header": chunk.get("header", ""),
                "tags": meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
                "category": meta.get("category", ""),
                "date_captured": str(meta.get("date_captured", meta.get("date", ""))),
                "status": meta.get("status", ""),
                "text": chunk["text"],
            })

    print(f"Chunked into {len(all_texts)} pieces")

    if not all_texts:
        print("No chunks to index.")
        return

    # Embed in batches
    print("Embedding...")
    vectors = _embed_batches(all_texts)

    # Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(INDEX_DIR / "vault_vectors.npy"), vectors.astype(np.float32))
    with open(INDEX_DIR / "vault_meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False)

    print(f"Saved {len(all_texts)} vault chunks → {INDEX_DIR}")


# ── Phase 3: Build transcript index ────────────────────────────────

def build_transcript_index():
    """Load transcript JSONs, chunk, embed, save to numpy + JSON."""
    print(f"\n{'='*60}")
    print("Phase 3: BUILD TRANSCRIPT INDEX")
    print(f"{'='*60}")

    json_files = sorted(f for f in TRANSCRIPTS_DIR.glob("*.json") if not f.name.startswith("_"))
    print(f"Found {len(json_files)} transcript files")

    if not json_files:
        print("No transcripts found.")
        return

    all_texts = []
    all_meta = []
    video_index = []
    total_videos = 0

    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        vid = data.get("video_id", filepath.stem)
        title = data.get("title", vid)
        url = data.get("url", f"https://www.youtube.com/watch?v={vid}")
        channel = data.get("channel", "AI Engineer")
        chunks = data.get("chunks", [])

        # If no pre-computed chunks, build from raw segments
        if not chunks:
            segments = data.get("segments", data.get("raw_segments", []))
            if segments:
                chunks = _merge_segments(segments)

        if not chunks:
            continue

        # Video summary entry
        video_index.append({
            "video_id": vid,
            "title": title,
            "url": url,
            "channel": channel,
            "language": data.get("language", ""),
            "segment_count": data.get("segment_count", len(data.get("segments", []))),
            "chunk_count": len(chunks),
        })

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text or len(text) < 20:
                continue
            # Safety: truncate mega-chunks to ~1500 chars at sentence boundary
            if len(text) > 2000:
                text = text[:1500]
                last_period = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
                if last_period > 300:
                    text = text[:last_period + 1]
            start = chunk.get("start", 0)
            all_texts.append(text)
            all_meta.append({
                "source": "youtube",
                "video_id": vid,
                "title": title,
                "url": f"{url}&t={int(start)}" if start else url,
                "channel": channel,
                "timestamp_start": start,
                "timestamp_end": chunk.get("end", start),
                "text": text,
            })

        total_videos += 1

    print(f"Chunked {total_videos} videos into {len(all_texts)} pieces")

    if not all_texts:
        print("No chunks to index.")
        return

    # Embed in batches
    print("Embedding...")
    vectors = _embed_batches(all_texts)

    # Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(INDEX_DIR / "transcript_vectors.npy"), vectors.astype(np.float32))
    with open(INDEX_DIR / "transcript_meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False)
    with open(INDEX_DIR / "video_index.json", "w", encoding="utf-8") as f:
        json.dump(video_index, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_texts)} transcript chunks from {total_videos} videos → {INDEX_DIR}")


# ── Phase 4: Build memory index ────────────────────────────────────

def build_memory_index():
    """Walk all memory directories, chunk, embed, save to numpy + JSON."""
    print(f"\n{'='*60}")
    print("Phase 4: BUILD MEMORY INDEX")
    print(f"{'='*60}")

    if not MEMORY_DIRS:
        print("No memory directories found.")
        return

    # Collect .md files from ALL memory directories
    files = []
    for mem_dir in MEMORY_DIRS:
        if not mem_dir.exists():
            continue
        for p in mem_dir.rglob("*.md"):
            if p.name == "MEMORY.md":
                continue
            files.append(p)
    files.sort()
    print(f"Found {len(files)} memory files across {len(MEMORY_DIRS)} directories")

    if not files:
        print("No memory files to index.")
        return

    # Parse + chunk
    all_texts = []
    all_meta = []

    for filepath in files:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(content)
        if not body.strip() or len(body.strip()) < 20:
            continue

        # Find which memory directory this file belongs to
        parent_mem_dir = None
        for mem_dir in MEMORY_DIRS:
            try:
                filepath.relative_to(mem_dir)
                parent_mem_dir = mem_dir
                break
            except ValueError:
                continue
        if parent_mem_dir is None:
            continue
        rel_path = str(filepath.relative_to(parent_mem_dir))
        parts = filepath.relative_to(parent_mem_dir).parts
        folder = parts[0] if len(parts) > 1 else "root"
        # Use frontmatter name if available, else derive from filename
        title = meta.get("name", filepath.stem.replace("-", " ").strip().title())
        mem_type = meta.get("type", folder)
        description = meta.get("description", "")

        chunks = _chunk_markdown(body)
        for chunk in chunks:
            if len(chunk["text"].strip()) < 20:
                continue
            all_texts.append(chunk["text"])
            all_meta.append({
                "source": "memory",
                "file_path": rel_path,
                "title": title,
                "folder": folder,
                "header": chunk.get("header", ""),
                "type": mem_type,
                "description": description,
                "text": chunk["text"],
            })

    print(f"Chunked into {len(all_texts)} pieces")

    if not all_texts:
        print("No chunks to index.")
        return

    # Embed in batches
    print("Embedding...")
    vectors = _embed_batches(all_texts)

    # Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(INDEX_DIR / "memory_vectors.npy"), vectors.astype(np.float32))
    with open(INDEX_DIR / "memory_meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False)

    print(f"Saved {len(all_texts)} memory chunks → {INDEX_DIR}")


# ── Helpers ─────────────────────────────────────────────────────────

def _parse_frontmatter(content):
    if not content.startswith("---"):
        return {}, content
    m = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", content, re.DOTALL)
    if m is None:
        return {}, content
    try:
        meta = yaml.safe_load(m.group(1))
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    return meta, content[m.end():].strip()


def _chunk_markdown(text, max_chars=1200):
    """Split markdown by headers, then by paragraphs if too long."""
    chunks = []
    current_header = ""
    current_text = ""

    for line in text.split("\n"):
        header_match = re.match(r"^(#{1,4})\s+(.+)", line)
        if header_match:
            if current_text.strip():
                chunks.append({"text": current_text.strip(), "header": current_header})
            current_header = header_match.group(2).strip()
            current_text = line + "\n"
        else:
            current_text += line + "\n"
            if len(current_text) > max_chars:
                chunks.append({"text": current_text.strip(), "header": current_header})
                current_text = ""

    if current_text.strip():
        chunks.append({"text": current_text.strip(), "header": current_header})
    return chunks


def _embed_batches(texts):
    """Embed texts in small batches with gc.collect(). Uses search_document prefix."""
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs = embed(batch)
        all_vecs.extend(vecs)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
        gc.collect()
    return np.array(all_vecs, dtype=np.float32)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ContextRAG index builder")
    parser.add_argument("--skip-vault", action="store_true", help="Skip vault indexing")
    parser.add_argument("--skip-transcripts", action="store_true", help="Skip transcript indexing")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory indexing")
    parser.add_argument("--memory-only", action="store_true", help="Only build memory index")
    args = parser.parse_args()

    print("=" * 60)
    print("ContextRAG — Local Hybrid Retrieval")
    print(f"Model: nomic-embed-text @ {EMBED_DIM}d")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)

    if args.memory_only:
        build_memory_index()
        gc.collect()
    else:
        if not args.skip_vault:
            build_vault_index()
            gc.collect()

        if not args.skip_transcripts:
            build_transcript_index()
            gc.collect()

        if not args.skip_memory:
            build_memory_index()
            gc.collect()

    print(f"\n{'='*60}")
    print("ALL PHASES COMPLETE")
    print(f"Index at: {INDEX_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    from embed import OllamaNotRunningError

    try:
        main()
    except OllamaNotRunningError as error:
        raise SystemExit(f"ContextRAG could not build an index: {error}") from error
