# Context Compressor MCP Server

An MCP (Model Context Protocol) server for compressing and expanding agent context using structure-aware extractive compression.

This server enables LLM agents to compress large wiki pages, long documents, or carryover files into condensed summaries, conserving context window tokens. The agent can skim the metadata of these compressed chunks, **search** them semantically, and selectively expand only the ones it needs detail from.

**What this is:** An extractive compressor. It selects the most informative sentences from the original text using TF-IDF scoring, preserves document structure, and stores the compressed text alongside the original for on-demand expansion.

**What this is not:** A learned LCLM encoder. LCLM (arXiv 2606.09659) requires a jointly-trained 0.6B encoder + 4B decoder pair. The latents are decoder-specific — they only mean something to the LCLM decoder they were trained with. Since Hermes routes to arbitrary OpenRouter models (which eat tokens, not latent embeddings), LCLM latents would be useless without running the LCLM decoder locally. For an agent gateway architecture, text-out compression is the only kind that transfers.

---

## ✨ Key Features

- **Structure-Aware Markdown Parsing** — Recognizes headings, code blocks, bullet lists, YAML frontmatter, and paragraphs. Never splits mid-code-block or mid-list.
- **Hierarchical Compression** — Sections are compressed independently; the document outline (all headings) is always preserved in the output.
- **TF-IDF Sentence Scoring** — Ranks content blocks by information density with bonuses for entities, structural markers, and position.
- **Entity Preservation** — Named entities and configurable domain-specific terms (e.g., `ELBO`, `PAC-Bayes`, `MCMC`) are prioritized during compression.
- **Embedding Search** — Qwen3-Embedding-0.6B for real semantic similarity (not TF-IDF). Lazy-loaded on first search.
- **Content-Hash Deduplication** — Re-compressing an unchanged file returns the existing chunk instantly.
- **Staleness Detection** — Detects when source files have changed since compression and supports bulk purging.
- **Inline Text Compression** — Compress text directly without needing a file on disk (conversation history, tool output, etc.).
- **Configurable Ratio** — Compression ratio from 1× (minimal) to 16× (aggressive).
- **Honest Confidence** — Compression quality score computed from ratio accuracy × entity coverage, not hardcoded.

---

All 8 context-compressor tools tested ✓

| Tool               | Description                                                                          |
| ------------------ | ------------------------------------------------------------------------------------ |
| compress_pages     | Compresses wiki pages with deduplication, entity preservation, hierarchical sections |
| compress_text      | Compresses inline text without file I/O                                              |
| expand_chunk       | Restores original full-text from chunk ID                                            |
| get_chunk_metadata | Returns ratio, entities, confidence, sections, staleness                             |
| search_chunks      | TF-IDF semantic search across compressed chunks                                      |
| list_chunks        | Lists chunks with source prefix filtering                                            |
| compression_stats  | Global stats: tokens saved, avg ratio, stale count                                   |
| purge_stale        | Dry-run and actual purge of stale chunks                                             |

## 📂 Codebase Overview

| File                                                  | Purpose                                                           |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| [server.py](src/context_compressor/server.py)         | FastMCP server with 8 tool registrations                          |
| [compressor.py](src/context_compressor/compressor.py) | Structure-aware extractive compression engine                     |
| [types.py](src/context_compressor/types.py)           | Pydantic models for metadata and request/response schemas         |
| [pyproject.toml](pyproject.toml)                      | Build configuration and dependencies                              |
| [mcp-config.example.json](mcp-config.example.json)    | Example MCP client configuration                          |
| [tests/](tests/)                                      | 67 tests covering compressor and server tools             |

---

## 🛠️ Installation & Setup

Requires [uv](https://github.com/astral-sh/uv) and Python ≥ 3.12.

```bash
# Install all dependencies
uv sync

# Run the test suite
uv run pytest tests/ -v
```

### Environment Variables

| Variable                   | Default                        | Description                               |
| -------------------------- | ------------------------------ | ----------------------------------------- |
| `CONTEXT_COMPRESSOR_STORE` | `~/.hermes/context-compressor` | Directory for persisted compressed chunks |
| `NEO4J_URI`                | `bolt://localhost:7687`        | Neo4j connection URI (Phase 2)            |
| `NEO4J_USERNAME`           | `neo4j`                        | Neo4j username                            |
| `NEO4J_PASSWORD`           | `00000000`                     | Neo4j password                            |
| `NEO4J_DATABASE`           | `synapse`                      | Neo4j database name                       |

---

## 🚀 MCP Client Integration

Copy [mcp-config.example.json](mcp-config.example.json) into your MCP client configuration.

**Claude Desktop** (Linux): `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "context-compressor": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/context-compressor",
        "run",
        "context-compressor"
      ],
      "env": {
        "CONTEXT_COMPRESSOR_STORE": "/home/wherever-is-convenient/context-compressor-store"
      }
    }
  }
}
```

---

## 🧰 Tools Reference

### `compress_pages`

Compress one or more files into latent chunk summaries. Deduplicates automatically.

| Parameter           | Type       | Default    | Description                                        |
| ------------------- | ---------- | ---------- | -------------------------------------------------- |
| `paths`             | `string[]` | _required_ | File paths to compress                             |
| `ratio`             | `number`   | `4.0`      | Target compression ratio (1-16)                    |
| `interleave`        | `boolean`  | `false`    | Interleave chunks from multiple files (LCLM-style) |
| `preserve_entities` | `boolean`  | `true`     | Always retain entity-bearing sentences             |

### `compress_text`

Compress inline text without requiring a file on disk.

| Parameter           | Type      | Default    | Description                            |
| ------------------- | --------- | ---------- | -------------------------------------- |
| `text`              | `string`  | _required_ | Text content to compress               |
| `ratio`             | `number`  | `4.0`      | Target compression ratio (1-16)        |
| `preserve_entities` | `boolean` | `true`     | Always retain entity-bearing sentences |
| `persist`           | `boolean` | `false`    | Save to disk for later retrieval       |
| `label`             | `string`  | `"inline"` | Label for the chunk if persisted       |

### `expand_chunk`

Restore the original full-text content for a compressed chunk.

| Parameter  | Type     | Description                                                   |
| ---------- | -------- | ------------------------------------------------------------- |
| `chunk_id` | `string` | The 16-char chunk ID from `compress_pages` or `compress_text` |

### `get_chunk_metadata`

Get metadata (ratio, entities, confidence, sections, staleness) without expanding.

| Parameter  | Type     | Description  |
| ---------- | -------- | ------------ |
| `chunk_id` | `string` | The chunk ID |

### `search_chunks`

Semantic search across compressed chunks using TF-IDF similarity.

| Parameter | Type      | Default    | Description           |
| --------- | --------- | ---------- | --------------------- |
| `query`   | `string`  | _required_ | Search query text     |
| `top_k`   | `integer` | `5`        | Max results to return |

### `list_chunks`

List all compressed chunks, optionally filtered.

| Parameter       | Type      | Default | Description                  |
| --------------- | --------- | ------- | ---------------------------- |
| `source_prefix` | `string`  | —       | Filter by source path prefix |
| `limit`         | `integer` | `50`    | Max results                  |

### `compression_stats`

Global metrics: total chunks, tokens saved, average ratio, stale count.

### `purge_stale`

Remove chunks whose source file has changed or been deleted.

| Parameter | Type      | Default | Description              |
| --------- | --------- | ------- | ------------------------ |
| `dry_run` | `boolean` | `true`  | Preview without deleting |
