# Context Compressor v0.2.0 — Walkthrough

## Summary

Rewrote the context-compressor MCP server from a Phase 1 prototype into a production-ready tool with 7 major improvements across architecture, compression quality, and developer ergonomics.

---

## Changes Made

### 1. Compressor Engine Rewrite ([compressor.py](context-compressor/src/context_compressor/compressor.py))

**Before:** Naive sentence splitting via a single regex. Code blocks, lists, and headings were treated as flat text. Entities could span across heading boundaries.

**After:**
- `parse_blocks()` — Markdown-aware parser recognizing 5 block types: `HEADING`, `PARAGRAPH`, `CODE`, `LIST`, `METADATA` (YAML frontmatter)
- Hierarchical compression: sections compressed independently, outline always preserved
- Code blocks and metadata always preserved whole (score = ∞)
- Abbreviation-aware sentence splitting (handles `e.g.`, `Dr.`, etc.)
- Configurable `important_terms` instead of a hardcoded set
- Entity regex strips heading markers before matching to prevent false cross-boundary entities

### 2. Server Architecture ([server.py](context-compressor/src/context_compressor/server.py))

**Before:** Low-level `Server` class with manual `list_tools()`/`call_tool()` dispatch — 413 lines for 5 tools.

**After:** `FastMCP` with `@mcp.tool()` decorators — typed parameters, no dispatch boilerplate. Now 8 tools in ~470 lines.

### 3. New Tools

| Tool | Purpose |
|------|---------|
| `compress_text` | Compress inline text without a file on disk |
| `search_chunks` | TF-IDF semantic search across all chunks |
| `purge_stale` | Detect and remove stale/orphaned chunks |

### 4. Deduplication & Staleness

- SHA-256 content hash stored with each chunk
- `compress_pages` checks for existing chunk with same `source_path + content_hash + ratio` — returns it without recompression
- `_check_staleness()` compares stored hash against current file content
- `purge_stale` supports dry-run preview before deletion

### 5. Real Interleaving

`interleave=True` now round-robin alternates chunks from different source files, creating mixed compressed/fresh conditioning (LCLM-style).

### 6. Types & Schema ([types.py](context-compressor/src/context_compressor/types.py))

Added `SectionInfo`, `content_hash`, `is_stale`, and `stale_chunks` fields.

### 7. Test Suite

| File | Tests | Coverage |
|------|-------|----------|
| [test_compressor.py](context-compressor/tests/test_compressor.py) | 35 | Token estimation, block parsing, entity extraction, compression at ratios 1/4/8/16, headings/code/frontmatter preservation, section summaries |
| [test_server.py](context-compressor/tests/test_server.py) | 32 | All 8 tools, deduplication, interleaving, staleness, search index, purge |

---

## What Was Tested

```
$ uv run pytest tests/ -v
============================== 67 passed in 1.18s ==============================
```

All 67 tests pass. The test suite covers:
- Block parsing for every Markdown element type
- Entity extraction edge cases (cross-heading, abbreviations, custom terms)
- Compression at ratios 1×, 4×, 8×, 16× with structural preservation
- Full tool lifecycle: compress → list → metadata → expand → search → purge
- Deduplication: same file compressed twice returns cached chunk
- Staleness: modified/deleted source files detected and purgeable

---

## Post-v0.2.0: LCLM Encoder Removal

An `LCLMEncoder` class was added (then removed) as a Phase 2 experiment. Key lessons:

1. **Dead code**: It was never wired to any MCP tool — `server.py` only imported `Compressor`
2. **Wrong model**: Loaded generic `Qwen3-Embedding-0.6B`, not the trained LCLM checkpoint
3. **Ad-hoc pooling**: Used grouped mean-pooling instead of the paper's learned compression
4. **No decoder**: Without the LCLM 4B decoder, latents can't reconstruct text — `expand_chunk` only worked because original text was stored verbatim
5. **Hardcoded confidence**: `confidence=0.9` — the exact anti-pattern for a metacognition project
6. **Stdout corruption risk**: `print()` calls in lazy-load would corrupt the stdio MCP protocol
7. **Storage math**: Latents inflate storage ~25× vs original text (100 latents × 1024-dim × 2 bytes = ~200KB for an 8KB document)
8. **Architecture mismatch**: LCLM latents are decoder-specific. Hermes routes to arbitrary OpenRouter models that eat tokens, not latent embeddings. Text-out compression is the only kind that transfers.

**Decision**: Remove `LCLMEncoder` entirely. The extractive compressor is the production tool. If LCLM is ever needed, it requires the full 0.6B encoder + 4B decoder running locally — a separate project, not a context-compression layer for Hermes.

## Post-v0.2.0: Embedding Search Upgrade

Replaced TF-IDF search with Qwen3-Embedding-0.6B for real semantic similarity. This salvages the one genuinely useful thing from the LCLM encoder experiment — the embedding model — and uses it for what it's actually good at: search.

**What changed:**
- `SearchIndexState` (TF-IDF) → `EmbeddingSearchIndex` (Qwen3-Embedding-0.6B)
- `_rebuild_search_index()` now batch-embeds all compressed chunks
- `search_chunks` uses cosine similarity on real embeddings
- Model is lazy-loaded on first search (not at startup)
- Falls back gracefully if model unavailable

**Why this works:** The embedding model produces unit-normalized vectors. Cosine similarity = dot product. No decoder needed — just embed the query, compare against stored chunk embeddings, return top-k. Storage model unchanged (embeddings are ~2KB per chunk, stored in memory only).
