"""Context Compressor — MCP server for compressing and expanding agent context.

Tools:
  compress_pages     — Compress wiki pages / carryover files into latent chunk summaries
  compress_text      — Compress inline text without requiring a file on disk
  expand_chunk       — Restore original content for a compressed chunk
  get_chunk_metadata — Get metadata for a compressed chunk (ratio, entities, confidence)
  search_chunks      — Semantic search across compressed chunks by query
  list_chunks        — List all compressed chunks, optionally filtered by source path
  compression_stats  — Global statistics (total pages, chunks, avg ratio, tokens saved)
  purge_stale        — Remove chunks whose source file has changed or been deleted
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from itertools import zip_longest

from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .compressor import (
    Compressor,
    CompressionResult,
    SectionSummary,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "00000000")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")

# Directory for persisting compressed chunks on disk
CHUNK_STORE = Path(
    os.getenv(
        "CONTEXT_COMPRESSOR_STORE",
        os.path.expanduser("~/.hermes/context-compressor"),
    )
)
CHUNK_STORE.mkdir(parents=True, exist_ok=True)

# Manifest file tracks all chunks
MANIFEST_PATH = CHUNK_STORE / "manifest.json"

# ---------------------------------------------------------------------------
# In-memory chunk index (loaded from disk on startup)
# ---------------------------------------------------------------------------
_chunk_index: dict[str, dict] = {}


class SearchIndexState:
    """In-memory TF-IDF search index state."""

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.chunk_ids: list[str] = []


# TF-IDF search index (rebuilt on startup and after compress)
_search_index = SearchIndexState()


def _load_manifest() -> dict[str, dict]:
    """Load chunk manifest from disk."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(index: dict[str, dict]) -> None:
    """Persist chunk manifest to disk."""
    MANIFEST_PATH.write_text(json.dumps(index, indent=2, default=str))


def _content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(source_path: str, ratio: float, timestamp: str) -> str:
    """Generate a stable chunk ID from source path + ratio + timestamp."""
    raw = f"{source_path}:{ratio}:{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _find_existing_chunk(
    source_path: str, content_hash_val: str, ratio: float
) -> str | None:
    """Check if a chunk already exists for this source+hash+ratio.

    Returns the chunk_id if found, None otherwise.
    """
    for cid, meta in _chunk_index.items():
        if (
            meta.get("source_path") == source_path
            and meta.get("content_hash") == content_hash_val
            and meta.get("ratio") == ratio
        ):
            return cid
    return None


def _check_staleness(meta: dict) -> bool:
    """Check if a chunk's source file has changed since compression.

    Returns True if the source file is missing or its content hash differs.
    """
    source_path = meta.get("source_path", "")
    if not source_path:
        return False
    p = Path(source_path)
    if not p.exists():
        return True
    try:
        current_hash = _content_hash(p.read_text(encoding="utf-8", errors="replace"))
        return current_hash != meta.get("content_hash", "")
    except OSError:
        return True


def _rebuild_search_index() -> None:
    """Rebuild the TF-IDF search index from stored chunks."""
    if not _chunk_index:
        _search_index.vectorizer = None
        _search_index.matrix = None
        _search_index.chunk_ids = []
        return

    # Collect compressed content from chunk files
    corpus: list[str] = []
    chunk_ids: list[str] = []

    for cid in _chunk_index:
        chunk_path = CHUNK_STORE / f"{cid}.json"
        if not chunk_path.exists():
            continue
        try:
            data = json.loads(chunk_path.read_text())
            compressed = data.get("compressed_content", "")
            if compressed:
                corpus.append(compressed)
                chunk_ids.append(cid)
        except (json.JSONDecodeError, OSError):
            continue

    if not corpus:
        _search_index.vectorizer = None
        _search_index.matrix = None
        _search_index.chunk_ids = []
        return

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)

    _search_index.vectorizer = vectorizer
    _search_index.matrix = matrix
    _search_index.chunk_ids = chunk_ids


def _section_summaries_to_dicts(
    sections: list[SectionSummary],
) -> list[dict]:
    """Convert SectionSummary objects to JSON-serializable dicts."""
    return [
        {
            "title": s.title,
            "level": s.level,
            "original_tokens": s.original_tokens,
            "compressed_tokens": s.compressed_tokens,
        }
        for s in sections
    ]


def _interleave_results(
    results: list[dict],
) -> list[dict]:
    """Interleave compressed chunk previews from multiple files.

    Creates an alternating sequence so the consuming model gets
    mixed compressed/fresh conditioning (LCLM-style).
    """
    if len(results) <= 1:
        return results

    # Round-robin interleave by cycling through source files

    # Group by source
    by_source: dict[str, list[dict]] = {}
    for r in results:
        src = r.get("source_path", "unknown")
        by_source.setdefault(src, []).append(r)

    sources = list(by_source.values())
    interleaved = []
    for group in zip_longest(*sources):
        for item in group:
            if item is not None:
                interleaved.append(item)

    return interleaved


# ---------------------------------------------------------------------------
# Server (FastMCP)
# ---------------------------------------------------------------------------
mcp = FastMCP("context-compressor")


@mcp.tool()
async def compress_pages(
    paths: list[str],
    ratio: float = 4.0,
    interleave: bool = False,
    preserve_entities: bool = True,
) -> str:
    """Compress one or more wiki pages / carryover files into latent chunk summaries.

    Returns chunk IDs and metadata. Supports configurable compression ratio (1-16x).
    Uses structure-aware Markdown parsing with TF-IDF sentence scoring and entity
    preservation. Deduplicates: if the file content hasn't changed since last
    compression at the same ratio, returns the existing chunk.

    Args:
        paths: File paths to compress (wiki pages, carryover files, etc.).
        ratio: Target compression ratio (1-16, default: 4). Higher = more compression.
        interleave: If true and multiple paths given, interleave compressed chunks
            so the model receives mixed compressed/fresh conditioning (LCLM-style).
        preserve_entities: Always preserve named entities and key facts (default: true).
    """
    # Clamp ratio
    ratio = max(1.0, min(16.0, ratio))

    compressor = Compressor(ratio=ratio, preserve_entities=preserve_entities)
    results = []

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            results.append({
                "path": path_str,
                "error": f"File not found: {path_str}",
            })
            continue

        content = path.read_text(encoding="utf-8", errors="replace")
        chash = _content_hash(content)
        resolved = str(path.resolve())

        # --- Deduplication: check for existing identical chunk ---
        existing_cid = _find_existing_chunk(resolved, chash, ratio)
        if existing_cid:
            meta = _chunk_index[existing_cid]
            results.append({
                "chunk_id": existing_cid,
                "source_path": path_str,
                "ratio": ratio,
                "actual_ratio": meta.get("actual_ratio", ratio),
                "original_tokens": meta.get("original_tokens", 0),
                "compressed_tokens": meta.get("compressed_tokens", 0),
                "key_entities": meta.get("key_entities", []),
                "confidence": meta.get("confidence", 0.0),
                "deduplicated": True,
            })
            continue

        # --- Compress ---
        compression_result: CompressionResult = compressor.compress(content)

        # Generate chunk ID
        timestamp = datetime.now(timezone.utc).isoformat()
        cid = _chunk_id(path_str, ratio, timestamp)

        # Build metadata
        metadata = {
            "chunk_id": cid,
            "source_path": resolved,
            "ratio": ratio,
            "original_tokens": compression_result.original_tokens,
            "compressed_tokens": compression_result.compressed_tokens,
            "actual_ratio": compression_result.actual_ratio,
            "key_entities": compression_result.key_entities,
            "confidence": compression_result.confidence,
            "created_at": timestamp,
            "interleaved": interleave,
            "content_hash": chash,
            "is_stale": False,
            "sections": _section_summaries_to_dicts(compression_result.sections),
        }

        # Persist compressed chunk to disk
        chunk_path = CHUNK_STORE / f"{cid}.json"
        chunk_path.write_text(json.dumps({
            "metadata": metadata,
            "compressed_content": compression_result.compressed_text,
            "original_content": content,  # stored for expand; Phase 2 will drop this
        }, indent=2))

        # Update manifest
        _chunk_index[cid] = metadata
        _save_manifest(_chunk_index)

        results.append({
            "chunk_id": cid,
            "source_path": path_str,
            "ratio": ratio,
            "actual_ratio": compression_result.actual_ratio,
            "original_tokens": compression_result.original_tokens,
            "compressed_tokens": compression_result.compressed_tokens,
            "key_entities": compression_result.key_entities,
            "confidence": compression_result.confidence,
            "sections": [s.title for s in compression_result.sections],
            "compressed_preview": (
                compression_result.compressed_text[:300] + "..."
                if len(compression_result.compressed_text) > 300
                else compression_result.compressed_text
            ),
        })

    # Rebuild search index with new chunks
    _rebuild_search_index()

    # Apply interleaving if requested
    success_results = [r for r in results if "error" not in r]
    if interleave:
        success_results = _interleave_results(success_results)

    output = {
        "chunks_created": len(success_results),
        "errors": [r for r in results if "error" in r],
        "chunks": success_results,
        "interleaved": interleave,
    }

    return json.dumps(output, indent=2)


@mcp.tool()
async def compress_text(
    text: str,
    ratio: float = 4.0,
    preserve_entities: bool = True,
    persist: bool = False,
    label: str = "inline",
) -> str:
    """Compress inline text without requiring a file on disk.

    Useful for compressing conversation history, tool output, pasted content,
    or any text the agent already has in context.

    Args:
        text: The text content to compress.
        ratio: Target compression ratio (1-16, default: 4).
        preserve_entities: Always preserve named entities and key facts.
        persist: If true, save the compressed chunk to disk for later retrieval.
        label: A label for this chunk (used as source_path if persisted).
    """
    ratio = max(1.0, min(16.0, ratio))
    compressor = Compressor(ratio=ratio, preserve_entities=preserve_entities)
    result: CompressionResult = compressor.compress(text)

    output = {
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "actual_ratio": result.actual_ratio,
        "key_entities": result.key_entities,
        "confidence": result.confidence,
        "sections": [s.title for s in result.sections],
        "compressed_text": result.compressed_text,
    }

    if persist:
        timestamp = datetime.now(timezone.utc).isoformat()
        chash = _content_hash(text)
        cid = _chunk_id(label, ratio, timestamp)

        metadata = {
            "chunk_id": cid,
            "source_path": f"inline:{label}",
            "ratio": ratio,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "actual_ratio": result.actual_ratio,
            "key_entities": result.key_entities,
            "confidence": result.confidence,
            "created_at": timestamp,
            "interleaved": False,
            "content_hash": chash,
            "is_stale": False,
            "sections": _section_summaries_to_dicts(result.sections),
        }

        chunk_path = CHUNK_STORE / f"{cid}.json"
        chunk_path.write_text(json.dumps({
            "metadata": metadata,
            "compressed_content": result.compressed_text,
            "original_content": text,
        }, indent=2))

        _chunk_index[cid] = metadata
        _save_manifest(_chunk_index)
        _rebuild_search_index()

        output["chunk_id"] = cid
        output["persisted"] = True

    return json.dumps(output, indent=2)


@mcp.tool()
async def expand_chunk(chunk_id: str) -> str:
    """Restore the original full-text content for a compressed chunk.

    This is the 'selective expansion' operation — the agent skims compressed
    chunks and only expands the ones it needs detail from.

    Args:
        chunk_id: The chunk ID returned by compress_pages or compress_text.
    """
    chunk_path = CHUNK_STORE / f"{chunk_id}.json"

    if not chunk_path.exists():
        if chunk_id in _chunk_index:
            return json.dumps({
                "error": (
                    f"Chunk file missing for {chunk_id}. "
                    "Manifest entry exists but file was deleted."
                ),
                "metadata": _chunk_index[chunk_id],
            }, indent=2)
        return json.dumps({"error": f"Chunk not found: {chunk_id}"}, indent=2)

    chunk_data = json.loads(chunk_path.read_text())
    original_content = chunk_data["original_content"]
    metadata = chunk_data["metadata"]

    return json.dumps({
        "chunk_id": chunk_id,
        "source_path": metadata["source_path"],
        "original_tokens": metadata["original_tokens"],
        "content": original_content,
    }, indent=2)


@mcp.tool()
async def get_chunk_metadata(chunk_id: str) -> str:
    """Get metadata for a compressed chunk without expanding it.

    Returns: original path, compression ratio, key entities, confidence score,
    token counts, creation timestamp, section outline, staleness status.

    Args:
        chunk_id: The chunk ID returned by compress_pages or compress_text.
    """
    if chunk_id in _chunk_index:
        meta = dict(_chunk_index[chunk_id])
        # Update staleness
        meta["is_stale"] = _check_staleness(meta)
        return json.dumps(meta, indent=2, default=str)

    return json.dumps({"error": f"Chunk not found: {chunk_id}"}, indent=2)


@mcp.tool()
async def search_chunks(query: str, top_k: int = 5) -> str:
    """Semantic search across compressed chunks by query.

    Uses TF-IDF similarity to find chunks most relevant to the query.
    Returns ranked results with relevance scores, chunk IDs, source paths,
    and compressed previews.

    Args:
        query: Search query text.
        top_k: Maximum number of results to return (default: 5).
    """
    if _search_index.vectorizer is None or _search_index.matrix is None:
        return json.dumps({
            "error": "Search index not available. Compress some content first.",
            "results": [],
        }, indent=2)

    query_vec = _search_index.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _search_index.matrix).flatten()

    # Get top-k indices
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < 0.01:
            continue  # skip negligible matches

        cid = _search_index.chunk_ids[idx]
        meta = _chunk_index.get(cid, {})

        # Load compressed preview
        preview = ""
        chunk_path = CHUNK_STORE / f"{cid}.json"
        if chunk_path.exists():
            try:
                data = json.loads(chunk_path.read_text())
                compressed = data.get("compressed_content", "")
                preview = compressed[:300] + "..." if len(compressed) > 300 else compressed
            except (json.JSONDecodeError, OSError):
                pass

        results.append({
            "chunk_id": cid,
            "relevance_score": round(score, 4),
            "source_path": meta.get("source_path", ""),
            "key_entities": meta.get("key_entities", []),
            "original_tokens": meta.get("original_tokens", 0),
            "compressed_tokens": meta.get("compressed_tokens", 0),
            "confidence": meta.get("confidence", 0.0),
            "compressed_preview": preview,
        })

    return json.dumps({
        "query": query,
        "returned": len(results),
        "results": results,
    }, indent=2)


@mcp.tool()
async def list_chunks(
    source_prefix: str | None = None,
    limit: int = 50,
) -> str:
    """List all compressed chunks in the store.

    Optionally filter by source path prefix.

    Args:
        source_prefix: Filter chunks whose original path starts with this prefix.
        limit: Maximum number of results (default: 50).
    """
    chunks = list(_chunk_index.values())

    if source_prefix:
        chunks = [
            c for c in chunks if c["source_path"].startswith(source_prefix)
        ]

    # Sort by creation time, newest first
    chunks.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    chunks = chunks[:limit]

    return json.dumps({
        "total": len(_chunk_index),
        "returned": len(chunks),
        "chunks": chunks,
    }, indent=2, default=str)


@mcp.tool()
async def compression_stats() -> str:
    """Global compression statistics.

    Returns total pages compressed, total chunks, average compression ratio,
    estimated tokens saved, and count of stale chunks.
    """
    total_chunks = len(_chunk_index)
    if total_chunks == 0:
        return json.dumps({
            "total_chunks": 0,
            "message": "No chunks compressed yet.",
        }, indent=2)

    total_original = sum(
        c.get("original_tokens", 0) for c in _chunk_index.values()
    )
    total_compressed = sum(
        c.get("compressed_tokens", 0) for c in _chunk_index.values()
    )
    avg_ratio = (
        sum(c.get("actual_ratio", 0) for c in _chunk_index.values())
        / total_chunks
    )
    avg_confidence = (
        sum(c.get("confidence", 0) for c in _chunk_index.values())
        / total_chunks
    )
    unique_sources = len(
        set(c["source_path"] for c in _chunk_index.values())
    )

    # Count stale chunks
    stale_count = sum(1 for c in _chunk_index.values() if _check_staleness(c))

    return json.dumps({
        "total_chunks": total_chunks,
        "unique_sources": unique_sources,
        "total_original_tokens": total_original,
        "total_compressed_tokens": total_compressed,
        "tokens_saved": total_original - total_compressed,
        "avg_compression_ratio": round(avg_ratio, 2),
        "avg_confidence": round(avg_confidence, 3),
        "stale_chunks": stale_count,
        "store_path": str(CHUNK_STORE),
    }, indent=2)


@mcp.tool()
async def purge_stale(dry_run: bool = True) -> str:
    """Remove chunks whose source file has changed or been deleted.

    Args:
        dry_run: If true (default), report which chunks would be purged
            without actually deleting them. Set to false to delete.
    """
    stale: list[dict] = []
    for cid, meta in list(_chunk_index.items()):
        if _check_staleness(meta):
            stale.append({
                "chunk_id": cid,
                "source_path": meta.get("source_path", ""),
                "created_at": meta.get("created_at", ""),
                "reason": (
                    "source deleted"
                    if not Path(meta.get("source_path", "")).exists()
                    else "content changed"
                ),
            })

    if not dry_run:
        for entry in stale:
            cid = entry["chunk_id"]
            # Remove chunk file
            chunk_path = CHUNK_STORE / f"{cid}.json"
            if chunk_path.exists():
                chunk_path.unlink()
            # Remove from index
            _chunk_index.pop(cid, None)

        _save_manifest(_chunk_index)
        _rebuild_search_index()

    return json.dumps({
        "dry_run": dry_run,
        "stale_count": len(stale),
        "stale_chunks": stale,
        "message": (
            f"Would purge {len(stale)} stale chunk(s)."
            if dry_run
            else f"Purged {len(stale)} stale chunk(s)."
        ),
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Start the context-compressor MCP server."""
    # Load manifest on startup
    _chunk_index.update(_load_manifest())

    # Build search index from existing chunks
    _rebuild_search_index()

    mcp.run()


if __name__ == "__main__":
    main()
