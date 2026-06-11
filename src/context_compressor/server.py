"""Context Compressor — MCP server for compressing and expanding agent context.

Tools:
  compress_pages    — Compress wiki pages / carryover files into latent chunk summaries
  expand_chunk      — Restore original content for a compressed chunk
  get_chunk_metadata — Get metadata for a compressed chunk (ratio, entities, confidence)
  compression_stats  — Global statistics (total pages, chunks, avg ratio, tokens saved)
  list_chunks       — List all compressed chunks, optionally filtered by source path
"""

from __future__ import annotations

import json
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool

from .compressor import Compressor, CompressionResult
from .types import ChunkMetadata

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "00000000")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")

# Directory for persisting compressed chunks on disk
CHUNK_STORE = Path(os.getenv("CONTEXT_COMPRESSOR_STORE", os.path.expanduser("~/.hermes/context-compressor")))
CHUNK_STORE.mkdir(parents=True, exist_ok=True)

# Manifest file tracks all chunks
MANIFEST_PATH = CHUNK_STORE / "manifest.json"

# ---------------------------------------------------------------------------
# In-memory chunk index (loaded from disk on startup)
# ---------------------------------------------------------------------------
_chunk_index: dict[str, dict] = {}


def _load_manifest() -> dict[str, dict]:
    """Load chunk manifest from disk."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(index: dict[str, dict]) -> None:
    """Persist chunk manifest to disk."""
    MANIFEST_PATH.write_text(json.dumps(index, indent=2, default=str))


def _chunk_id(source_path: str, ratio: float, timestamp: str) -> str:
    """Generate a stable chunk ID from source path + ratio + timestamp."""
    raw = f"{source_path}:{ratio}:{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
server = Server("context-compressor")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="compress_pages",
            description=(
                "Compress one or more wiki pages / carryover files into latent chunk summaries. "
                "Returns chunk IDs and metadata. Supports configurable compression ratio (1-16x). "
                "Phase 1 uses extractive compression (TF-IDF sentence scoring + entity preservation). "
                "Phase 2 will use a learned LCLM encoder."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to compress (wiki pages, carryover files, etc.)",
                    },
                    "ratio": {
                        "type": "number",
                        "description": "Target compression ratio (1-16, default: 4). Higher = more compression.",
                    },
                    "interleave": {
                        "type": "boolean",
                        "description": (
                            "If true and multiple paths given, interleave compressed chunks "
                            "so the model learns mixed compressed/fresh conditioning (LCLM-style). "
                            "Default: false."
                        ),
                    },
                    "preserve_entities": {
                        "type": "boolean",
                        "description": "Always preserve named entities and key facts (default: true).",
                    },
                },
                "required": ["paths"],
            },
        ),
        Tool(
            name="expand_chunk",
            description=(
                "Restore the original full-text content for a compressed chunk. "
                "This is the 'selective expansion' operation — the agent skims compressed chunks "
                "and only expands the ones it needs detail from."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID returned by compress_pages.",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="get_chunk_metadata",
            description=(
                "Get metadata for a compressed chunk without expanding it. "
                "Returns: original path, compression ratio, key entities, confidence score, "
                "token counts, creation timestamp."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID returned by compress_pages.",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="list_chunks",
            description=(
                "List all compressed chunks in the store. "
                "Optionally filter by source path prefix."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_prefix": {
                        "type": "string",
                        "description": "Filter chunks whose original path starts with this prefix (optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 50).",
                    },
                },
            },
        ),
        Tool(
            name="compression_stats",
            description=(
                "Global compression statistics: total pages compressed, total chunks, "
                "average compression ratio, estimated tokens saved."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "compress_pages":
        return await _compress_pages(arguments)
    elif name == "expand_chunk":
        return await _expand_chunk(arguments)
    elif name == "get_chunk_metadata":
        return await _get_chunk_metadata(arguments)
    elif name == "list_chunks":
        return await _list_chunks(arguments)
    elif name == "compression_stats":
        return await _compression_stats(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _compress_pages(args: dict) -> list[TextContent]:
    """Compress one or more files into latent chunk summaries."""
    global _chunk_index

    paths: list[str] = args["paths"]
    ratio: float = args.get("ratio", 4.0)
    interleave: bool = args.get("interleave", False)
    preserve_entities: bool = args.get("preserve_entities", True)

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
        compression_result: CompressionResult = compressor.compress(content)

        # Generate chunk ID
        timestamp = datetime.now(timezone.utc).isoformat()
        cid = _chunk_id(path_str, ratio, timestamp)

        # Build metadata
        metadata = {
            "chunk_id": cid,
            "source_path": str(path.resolve()),
            "ratio": ratio,
            "original_tokens": compression_result.original_tokens,
            "compressed_tokens": compression_result.compressed_tokens,
            "actual_ratio": compression_result.actual_ratio,
            "key_entities": compression_result.key_entities,
            "confidence": compression_result.confidence,
            "created_at": timestamp,
            "interleaved": interleave,
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
            "compressed_preview": compression_result.compressed_text[:200] + "..."
                if len(compression_result.compressed_text) > 200
                else compression_result.compressed_text,
        })

    output = {
        "chunks_created": len([r for r in results if "error" not in r]),
        "errors": [r for r in results if "error" in r],
        "chunks": [r for r in results if "error" not in r],
        "interleaved": interleave,
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _expand_chunk(args: dict) -> list[TextContent]:
    """Restore original content for a compressed chunk."""
    cid: str = args["chunk_id"]
    chunk_path = CHUNK_STORE / f"{cid}.json"

    if not chunk_path.exists():
        # Check if it's in the manifest but file was deleted
        if cid in _chunk_index:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Chunk file missing for {cid}. Manifest entry exists but file was deleted.",
                    "metadata": _chunk_index[cid],
                }, indent=2)
            )]
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Chunk not found: {cid}"}, indent=2)
        )]

    chunk_data = json.loads(chunk_path.read_text())
    original_content = chunk_data["original_content"]
    metadata = chunk_data["metadata"]

    return [TextContent(
        type="text",
        text=json.dumps({
            "chunk_id": cid,
            "source_path": metadata["source_path"],
            "original_tokens": metadata["original_tokens"],
            "content": original_content,
        }, indent=2)
    )]


async def _get_chunk_metadata(args: dict) -> list[TextContent]:
    """Get metadata for a compressed chunk without expanding."""
    cid: str = args["chunk_id"]

    if cid in _chunk_index:
        return [TextContent(
            type="text",
            text=json.dumps(_chunk_index[cid], indent=2, default=str)
        )]

    return [TextContent(
        type="text",
        text=json.dumps({"error": f"Chunk not found: {cid}"}, indent=2)
    )]


async def _list_chunks(args: dict) -> list[TextContent]:
    """List compressed chunks, optionally filtered by source prefix."""
    source_prefix: str | None = args.get("source_prefix")
    limit: int = args.get("limit", 50)

    chunks = list(_chunk_index.values())

    if source_prefix:
        chunks = [c for c in chunks if c["source_path"].startswith(source_prefix)]

    # Sort by creation time, newest first
    chunks.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    chunks = chunks[:limit]

    return [TextContent(
        type="text",
        text=json.dumps({
            "total": len(_chunk_index),
            "returned": len(chunks),
            "chunks": chunks,
        }, indent=2, default=str)
    )]


async def _compression_stats(args: dict) -> list[TextContent]:
    """Global compression statistics."""
    total_chunks = len(_chunk_index)
    if total_chunks == 0:
        return [TextContent(
            type="text",
            text=json.dumps({
                "total_chunks": 0,
                "message": "No chunks compressed yet.",
            }, indent=2)
        )]

    total_original = sum(c.get("original_tokens", 0) for c in _chunk_index.values())
    total_compressed = sum(c.get("compressed_tokens", 0) for c in _chunk_index.values())
    avg_ratio = sum(c.get("actual_ratio", 0) for c in _chunk_index.values()) / total_chunks
    avg_confidence = sum(c.get("confidence", 0) for c in _chunk_index.values()) / total_chunks
    unique_sources = len(set(c["source_path"] for c in _chunk_index.values()))

    return [TextContent(
        type="text",
        text=json.dumps({
            "total_chunks": total_chunks,
            "unique_sources": unique_sources,
            "total_original_tokens": total_original,
            "total_compressed_tokens": total_compressed,
            "tokens_saved": total_original - total_compressed,
            "avg_compression_ratio": round(avg_ratio, 2),
            "avg_confidence": round(avg_confidence, 3),
            "store_path": str(CHUNK_STORE),
        }, indent=2)
    )]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import anyio

    # Load manifest on startup
    global _chunk_index
    _chunk_index = _load_manifest()

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream,
                InitializationOptions(
                    server_name="context-compressor",
                    server_version="0.1.0",
                    capabilities=ServerCapabilities(),
                ),
            )

    anyio.run(run)


if __name__ == "__main__":
    main()
