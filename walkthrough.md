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
