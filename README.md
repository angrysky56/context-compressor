# Context Compressor MCP Server

An MCP (Model Context Protocol) server for compressing and expanding agent context using LCLM-inspired latent chunk management.

This server enables LLM agents to compress large wiki pages, long documents, or carryover files into condensed summaries, conserving context window tokens. The agent can skim the metadata of these compressed chunks and selectively expand them to retrieve full details when needed.

---

## 🌟 Key Features

*   **Phase 1 (Current): Extractive Compression**
    *   Sentence scoring using TF-IDF.
    *   Preservation of key named entities and domain-specific terms (e.g., `ELBO`, `PAC-Bayes`, `MCMC`, `Neo4j`, etc.).
    *   Configurable compression ratio target ($1\text{x}$ to $16\text{x}$).
    *   LCLM-style interleaving of multiple compressed chunks.
*   **Phase 2 (Planned): Generative/LCLM Compression**
    *   Learned LCLM encoder (0.6B model from arXiv:2606.09659) for true $p(x|z)$ reconstruction.
*   **Selective Expansion:** Low-token chunk tracking and on-demand expansion of content details.
*   **Disk-Based Storage:** Compressed chunks and metadata are saved to disk under `~/.hermes/context-compressor` (configurable).

---

## 📂 Codebase Overview

*   [src/context_compressor/server.py](file:///home/ty/Repositories/ai_workspace/context-compressor/src/context_compressor/server.py): The entry point for the MCP stdio server containing tool registrations and handlers.
*   [src/context_compressor/compressor.py](file:///home/ty/Repositories/ai_workspace/context-compressor/src/context_compressor/compressor.py): Core extractive compression engine using TF-IDF sentence scoring and entity preservation.
*   [src/context_compressor/types.py](file:///home/ty/Repositories/ai_workspace/context-compressor/src/context_compressor/types.py): Pydantic models ([ChunkMetadata](file:///home/ty/Repositories/ai_workspace/context-compressor/src/context_compressor/types.py#L8), [CompressionRequest](file:///home/ty/Repositories/ai_workspace/context-compressor/src/context_compressor/types.py#L22), and [CompressionStats](file:///home/ty/Repositories/ai_workspace/context-compressor/src/context_compressor/types.py#L30)) defining input and output schemas.
*   [pyproject.toml](file:///home/ty/Repositories/ai_workspace/context-compressor/pyproject.toml): Build configuration and project dependencies.
*   [mcp-config.example.json](file:///home/ty/Repositories/ai_workspace/context-compressor/mcp-config.example.json): Configuration example for MCP clients like Claude Desktop.

---

## 🛠️ Installation & Setup

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

### 1. Install Dependencies

In the project directory, run:
```bash
uv sync
```
This sets up a local virtual environment (`.venv/`) and installs all the required dependencies.

### 2. Environment Variables

The server uses the following optional environment variables:
*   `CONTEXT_COMPRESSOR_STORE`: Directory path to store compressed chunks (default: `~/.hermes/context-compressor`).
*   `NEO4J_URI`: Connection URI for Neo4j (default: `bolt://localhost:7687`).
*   `NEO4J_USERNAME` / `NEO4J_USER`: Neo4j username (default: `neo4j`).
*   `NEO4J_PASSWORD`: Neo4j password (default: `00000000`).
*   `NEO4J_DATABASE`: Neo4j database (default: `synapse`).

---

## 🚀 MCP Client Integration

To integrate this server with **Claude Desktop** or another MCP client:

1. Copy the example configuration [mcp-config.example.json](file:///home/ty/Repositories/ai_workspace/context-compressor/mcp-config.example.json) to your MCP client config folder.
   * On Linux, the Claude Desktop configuration is located at `~/.config/Claude/claude_desktop_config.json`.
2. Update the command arguments and environment variables in the file as needed.

### Configuration Example

```json
{
  "mcpServers": {
    "context-compressor": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/context-compressor",
        "run",
        "context-compressor"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your_neo4j_password_here",
        "CONTEXT_COMPRESSOR_STORE": "/home/ty/.hermes/context-compressor"
      }
    }
  }
}
```

---

## 🧰 Tools Reference

### 1. `compress_pages`
Compress one or more text files into latent chunk summaries.
*   **Arguments**:
    *   `paths` (array of strings, *required*): Absolute or relative paths to files to compress.
    *   `ratio` (number, *optional*, default: `4.0`): Target compression ratio (1-16).
    *   `interleave` (boolean, *optional*, default: `false`): If true, interleaves multiple files' chunks to help the model learn mixed conditioning contexts.
    *   `preserve_entities` (boolean, *optional*, default: `true`): Always retain sentences containing key entities and predefined terms.
*   **Returns**: Chunk metadata for each compressed file, along with a truncated preview of the compressed content.

### 2. `expand_chunk`
Retrieve the original, uncompressed text content for a specified chunk.
*   **Arguments**:
    *   `chunk_id` (string, *required*): The 16-character SHA-256 chunk ID returned during compression.
*   **Returns**: Original content, original token estimation, and source path.

### 3. `get_chunk_metadata`
Fetch details for a compressed chunk without restoring the full-text content.
*   **Arguments**:
    *   `chunk_id` (string, *required*): The 16-character chunk ID.
*   **Returns**: Full [ChunkMetadata](file:///home/ty/Repositories/ai_workspace/context-compressor/src/context_compressor/types.py#L8) properties including creation date, exact token sizes, list of preserved key entities, and compression confidence.

### 4. `list_chunks`
List all stored compressed chunks.
*   **Arguments**:
    *   `source_prefix` (string, *optional*): Filter chunks whose original file path starts with this prefix.
    *   `limit` (integer, *optional*, default: `50`): Maximum number of records to return.
*   **Returns**: A list of chunk metadata entries ordered by creation date (newest first).

### 5. `compression_stats`
Retrieve global metrics across all compressed content.
*   **Arguments**: None.
*   **Returns**: Total number of chunks, unique sources, original token count, compressed token count, total tokens saved, average compression ratio, average confidence, and storage path.
