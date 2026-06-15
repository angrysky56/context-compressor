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
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .compressor import (
    CompressionResult,
    Compressor,
    SectionSummary,
)

# ---------------------------------------------------------------------------
# Logging — stderr only (stdout is the JSON-RPC channel for stdio MCP)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("context_compressor")

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

# SQLite DB tracks all chunks
DB_PATH = CHUNK_STORE / "chunks.db"

# Legacy manifest file for migration
MANIFEST_PATH = CHUNK_STORE / "manifest.json"


def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                metadata TEXT,
                original_content TEXT,
                compressed_content TEXT,
                embedding TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER
            )
        """)


def _migrate_json_to_db() -> None:
    if not MANIFEST_PATH.exists():
        return

    logger.info("Migrating existing JSON chunks to SQLite database...")
    try:
        old_index = json.loads(MANIFEST_PATH.read_text())
    except Exception:
        old_index = {}

    with sqlite3.connect(DB_PATH) as conn:
        for cid in old_index:
            chunk_path = CHUNK_STORE / f"{cid}.json"
            if not chunk_path.exists():
                continue

            try:
                data = json.loads(chunk_path.read_text())
                metadata = data.get("metadata", old_index[cid])
                compressed = data.get("compressed_content", "")
                original = data.get("original_content", "")
                embedding = data.get("embedding", [])
                embedding_model = data.get("embedding_model", "")
                embedding_dim = data.get("embedding_dim", 0)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, metadata, original_content, compressed_content, embedding, embedding_model, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        cid,
                        json.dumps(metadata),
                        original,
                        compressed,
                        json.dumps(embedding) if embedding else None,
                        embedding_model,
                        embedding_dim,
                    ),
                )
            except Exception as e:
                logger.error("Failed to migrate chunk %s: %s", cid, e)
            else:
                chunk_path.unlink()

    try:
        MANIFEST_PATH.unlink()
    except OSError:
        pass
    logger.info("Migration to SQLite completed.")


# Embedding backend selection
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local")  # "local" or "openrouter"
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------------------
# In-memory chunk index (loaded from disk on startup)
# ---------------------------------------------------------------------------
_chunk_index: dict[str, dict] = {}


class TfidfSearchIndex:
    """TF-IDF fallback search index.

    Used when the embedding model is unavailable. Provides the same
    `search(query, top_k)` interface as EmbeddingSearchIndex.
    """

    def __init__(self) -> None:
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._tfidf_matrix = None
        self._vocabulary: dict[str, int] = {}
        self._loaded = False

    def build(self, corpus: list[str], chunk_ids: list[str]) -> None:
        """Build TF-IDF matrix from corpus texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._corpus = corpus
        self._chunk_ids = chunk_ids
        if not corpus:
            self._loaded = True
            return

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,
        )
        self._tfidf_matrix = vectorizer.fit_transform(corpus)
        self._vocabulary = vectorizer.vocabulary_
        self._loaded = True

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search for the top-k most similar chunks to the query.

        Returns list of (chunk_id, similarity_score) tuples.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not self._loaded or not self._chunk_ids:
            return []

        # Rebuild a vectorizer with the same vocabulary for query transform
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            vocabulary=self._vocabulary,
        )
        # Fit on empty to set up, then transform query
        vectorizer.fit(self._corpus)
        query_vec = vectorizer.transform([query])

        # Cosine similarity via dot product (TF-IDF vectors are L2-normalized by sklearn)
        similarities = (self._tfidf_matrix @ query_vec.T).toarray().flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.01:
                continue
            results.append((self._chunk_ids[idx], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def clear(self) -> None:
        self._corpus = []
        self._chunk_ids = []
        self._tfidf_matrix = None
        self._vocabulary = {}
        self._loaded = False


class EmbeddingBackend:
    """Abstract embedding backend interface."""

    def embed_texts(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        raise NotImplementedError


class LocalEmbeddingBackend(EmbeddingBackend):
    """Local Qwen3-Embedding backend using transformers.

    Follows the Qwen3-Embedding-0.6B model card:
    - Left padding for batch tokenization
    - Last-token (EOS) pooling: take hidden state of last non-pad token
    - Instruction-prefixed queries for retrieval
    - bfloat16 on CUDA, float32 on CPU
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._model_name = EMBED_MODEL
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        # Left padding as required by Qwen3-Embedding
        self.tokenizer.padding_side = "left"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
        )
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self._loaded = True
        logger.info("Loaded embedding model: %s (dtype=%s)", self._model_name, dtype)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors.

        Args:
            texts: Texts to embed.
            is_query: If True, prepend the instruction prefix required by
                Qwen3-Embedding for retrieval queries.
        """
        import torch

        self._load()

        # Apply instruction prefix for queries per model card
        if is_query:
            prefix = "Instruct: Given a web search query, retrieve relevant passages\nQuery: "
            texts = [f"{prefix}{t}" for t in texts]

        # Tokenize with left padding
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Last-token pooling: with left padding, last non-pad is at position -1
            embeddings = outputs.last_hidden_state[:, -1]

        # Normalize to unit vectors for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()


class OpenRouterEmbeddingBackend(EmbeddingBackend):
    """OpenRouter embedding backend.

    POSTs to OpenRouter's embeddings endpoint. Falls back to TF-IDF
    on network errors.
    """

    def __init__(self) -> None:
        self._model_name = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
        self._api_key = OPENROUTER_API_KEY

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Embed texts via OpenRouter API.

        OpenRouter doesn't use instruction prefixes, so is_query is ignored.
        """
        import json as _json

        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        results = []
        # OpenRouter may have batch limits; send one at a time to be safe
        for text in texts:
            payload = _json.dumps(
                {
                    "model": self._model_name,
                    "input": text,
                }
            ).encode("utf-8")

            req = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
                    embedding = data["data"][0]["embedding"]
                    results.append(embedding)
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                KeyError,
                IndexError,
            ) as e:
                logger.error("OpenRouter embedding failed for text: %s", e)
                raise

        # Normalize
        import numpy as np

        arr = np.array(results)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr = arr / norms
        return arr.tolist()


class EmbeddingSearchIndex:
    """In-memory embedding search index.

    Supports pluggable backends (local Qwen3-Embedding or OpenRouter).
    Falls back to TF-IDF when the embedding backend fails.
    Tracks index status so callers can distinguish "no matches" from "search is broken".
    """

    def __init__(self) -> None:
        self.backend: EmbeddingBackend | None = None
        self.embedding_matrix: list[list[float]] = []
        self.chunk_ids: list[str] = []
        self.tfidf_index = TfidfSearchIndex()
        self._status = "unavailable"
        self._active_model = ""

    @property
    def status(self) -> str:
        return self._status

    def _create_backend(self) -> EmbeddingBackend:
        if EMBED_BACKEND == "openrouter":
            return OpenRouterEmbeddingBackend()
        return LocalEmbeddingBackend()

    def embed_texts(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        if self.backend is None:
            self.backend = self._create_backend()
        return self.backend.embed_texts(texts, is_query=is_query)

    @property
    def _model_name(self) -> str:
        if self.backend is not None:
            return self.backend.model_name
        return EMBED_MODEL

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search for the top-k most similar chunks to the query.

        Returns list of (chunk_id, similarity_score) tuples.
        """
        import numpy as np

        if not self.embedding_matrix or not self.chunk_ids:
            return []

        # Embed the query with instruction prefix
        try:
            query_embedding = self.embed_texts([query], is_query=True)[0]
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            # Fall back to TF-IDF
            return self.tfidf_index.search(query, top_k)

        query_vec = np.array(query_embedding)
        matrix = np.array(self.embedding_matrix)
        similarities = matrix @ query_vec  # dot product = cosine for unit vectors

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.01:
                continue  # skip very low similarity
            results.append((self.chunk_ids[idx], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def clear(self) -> None:
        """Clear the embedding index (but not the backend model)."""
        self.embedding_matrix = []
        self.chunk_ids = []
        self.tfidf_index.clear()

    def unload(self) -> None:
        """Free model memory."""
        self.backend = None
        self._status = "unavailable"
        self._active_model = ""
        self.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Embedding search index (rebuilt on startup and after compress)
_search_index = EmbeddingSearchIndex()


def _load_from_db() -> dict[str, dict]:
    """Load chunk metadata index from SQLite DB."""
    index = {}
    if not DB_PATH.exists():
        return index

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT chunk_id, metadata FROM chunks")
            for row in cursor:
                try:
                    index[row["chunk_id"]] = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    pass
        except sqlite3.OperationalError:
            pass  # DB might not be initialized yet
    return index


def _save_chunk_to_db(cid: str, metadata: dict, original: str, compressed: str) -> None:
    """Persist a chunk to SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, metadata, original_content, compressed_content)
            VALUES (?, ?, ?, ?)
        """,
            (cid, json.dumps(metadata), original, compressed),
        )


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
    """Rebuild the embedding search index from stored chunks.

    Loads existing embeddings from chunk JSONs. Only embeds chunks that
    are missing a vector or whose embedding_model differs from the active model.
    Falls back to TF-IDF if the embedding backend fails.
    """
    _search_index.clear()

    if not _chunk_index:
        _search_index._status = "ready"
        logger.info("No chunks to index")
        return

    # Collect compressed content and existing embeddings from chunk files
    texts_to_embed: list[str] = []
    ids_to_embed: list[str] = []
    existing_embeddings: list[list[float]] = []
    existing_ids: list[str] = []
    all_corpus: list[str] = []
    all_chunk_ids: list[str] = []

    active_model = _search_index._model_name

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for cid in _chunk_index:
            try:
                row = conn.execute(
                    "SELECT compressed_content, embedding, embedding_model FROM chunks WHERE chunk_id = ?",
                    (cid,),
                ).fetchone()
                if not row:
                    continue

                compressed = row["compressed_content"]
                if not compressed:
                    continue
                all_corpus.append(compressed)
                all_chunk_ids.append(cid)

                stored_model = row["embedding_model"]
                stored_embedding_raw = row["embedding"]

                try:
                    stored_embedding = (
                        json.loads(stored_embedding_raw) if stored_embedding_raw else []
                    )
                except Exception:
                    stored_embedding = []

                if stored_embedding and stored_model == active_model:
                    existing_embeddings.append(stored_embedding)
                    existing_ids.append(cid)
                else:
                    texts_to_embed.append(compressed)
                    ids_to_embed.append(cid)
            except Exception as e:
                logger.warning("Failed to read chunk %s: %s", cid, e)
                continue

    if not all_corpus:
        _search_index._status = "ready"
        logger.info("No corpus to index")
        return

    # Always build TF-IDF as a fallback
    _search_index.tfidf_index.build(all_corpus, all_chunk_ids)

    # Embed only the chunks that need it
    if texts_to_embed:
        logger.info(
            "Embedding %d new/changed chunks (reusing %d existing)",
            len(texts_to_embed),
            len(existing_embeddings),
        )
        try:
            # Batch embed at most 16 texts per forward pass
            batch_size = 16
            new_embeddings: list[list[float]] = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i : i + batch_size]
                batch_embeddings = _search_index.embed_texts(batch, is_query=False)
                new_embeddings.extend(batch_embeddings)

            # Merge existing + new, preserving chunk_id order
            emb_map: dict[str, list[float]] = {}
            for cid, emb in zip(existing_ids, existing_embeddings, strict=False):
                emb_map[cid] = emb
            for cid, emb in zip(ids_to_embed, new_embeddings, strict=False):
                emb_map[cid] = emb

            # Store embeddings back into chunks table
            with sqlite3.connect(DB_PATH) as conn:
                for cid in ids_to_embed:
                    if cid in emb_map:
                        try:
                            conn.execute(
                                """
                                UPDATE chunks SET embedding = ?, embedding_model = ?, embedding_dim = ?
                                WHERE chunk_id = ?
                            """,
                                (
                                    json.dumps(emb_map[cid]),
                                    active_model,
                                    len(emb_map[cid]),
                                    cid,
                                ),
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to save embedding for chunk %s: %s", cid, e
                            )

            # Build the matrix in consistent order
            ordered_ids = list(all_chunk_ids)
            ordered_embeddings = [emb_map[cid] for cid in ordered_ids if cid in emb_map]
            _search_index.embedding_matrix = ordered_embeddings
            _search_index.chunk_ids = ordered_ids
            _search_index._status = "ready"
            _search_index._active_model = active_model
            logger.info("Embedding index rebuilt: %d chunks", len(ordered_ids))
        except Exception as e:
            logger.error("Embedding model failed, falling back to TF-IDF: %s", e)
            _search_index.embedding_matrix = []
            _search_index.chunk_ids = []
            _search_index._status = "fallback_tfidf"
    else:
        # All embeddings were reused from disk
        _search_index.embedding_matrix = existing_embeddings
        _search_index.chunk_ids = existing_ids
        _search_index._status = "ready"
        _search_index._active_model = active_model
        logger.info("Embedding index loaded from disk: %d chunks", len(existing_ids))


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
            results.append(
                {
                    "path": path_str,
                    "error": f"File not found: {path_str}",
                }
            )
            continue

        content = path.read_text(encoding="utf-8", errors="replace")
        chash = _content_hash(content)
        resolved = str(path.resolve())

        # --- Deduplication: check for existing identical chunk ---
        existing_cid = _find_existing_chunk(resolved, chash, ratio)
        if existing_cid:
            meta = _chunk_index[existing_cid]
            results.append(
                {
                    "chunk_id": existing_cid,
                    "source_path": path_str,
                    "ratio": ratio,
                    "actual_ratio": meta.get("actual_ratio", ratio),
                    "original_tokens": meta.get("original_tokens", 0),
                    "compressed_tokens": meta.get("compressed_tokens", 0),
                    "key_entities": meta.get("key_entities", []),
                    "confidence": meta.get("confidence", 0.0),
                    "deduplicated": True,
                }
            )
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

        # Persist compressed chunk to DB
        _save_chunk_to_db(cid, metadata, content, compression_result.compressed_text)

        # Update manifest
        _chunk_index[cid] = metadata

        results.append(
            {
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
            }
        )

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

        _save_chunk_to_db(cid, metadata, text, result.compressed_text)
        _chunk_index[cid] = metadata
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
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT original_content, metadata FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()

    if not row:
        if chunk_id in _chunk_index:
            return json.dumps(
                {
                    "error": (
                        f"Chunk DB row missing for {chunk_id}. "
                        "Manifest entry exists but row was deleted."
                    ),
                    "metadata": _chunk_index[chunk_id],
                },
                indent=2,
            )
        return json.dumps({"error": f"Chunk not found: {chunk_id}"}, indent=2)

    original_content = row["original_content"]
    metadata = json.loads(row["metadata"])

    return json.dumps(
        {
            "chunk_id": chunk_id,
            "source_path": metadata["source_path"],
            "original_tokens": metadata["original_tokens"],
            "content": original_content,
        },
        indent=2,
    )


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

    Uses the configured embedding backend (local Qwen3-Embedding-0.6B or OpenRouter)
    for real embedding similarity. Falls back to TF-IDF if the embedding backend
    is unavailable. Returns ranked results with relevance scores, chunk IDs, source paths,
    and compressed previews.

    Args:
        query: Search query text.
        top_k: Maximum number of results to return (default: 5).
    """
    results = _search_index.search(query, top_k)
    index_status = _search_index.status

    if not results:
        return json.dumps(
            {
                "query": query,
                "returned": 0,
                "results": [],
                "index_status": index_status,
                "message": (
                    "No results. Compress some content first."
                    if index_status == "ready"
                    else f"Search index status: {index_status}. Compress some content or check embedding backend."
                ),
            },
            indent=2,
        )

    output_results = []
    for cid, score in results:
        meta = _chunk_index.get(cid, {})

        # Load compressed preview
        preview = ""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT compressed_content FROM chunks WHERE chunk_id = ?", (cid,)
                ).fetchone()
                if row and row[0]:
                    compressed = row[0]
                    preview = (
                        compressed[:300] + "..."
                        if len(compressed) > 300
                        else compressed
                    )
        except Exception:
            pass

        output_results.append(
            {
                "chunk_id": cid,
                "relevance_score": round(score, 4),
                "source_path": meta.get("source_path", ""),
                "key_entities": meta.get("key_entities", []),
                "original_tokens": meta.get("original_tokens", 0),
                "compressed_tokens": meta.get("compressed_tokens", 0),
                "confidence": meta.get("confidence", 0.0),
                "compressed_preview": preview,
            }
        )

    return json.dumps(
        {
            "query": query,
            "returned": len(output_results),
            "index_status": index_status,
            "results": output_results,
        },
        indent=2,
    )


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
        chunks = [c for c in chunks if c["source_path"].startswith(source_prefix)]

    # Sort by creation time, newest first
    chunks.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    chunks = chunks[:limit]

    return json.dumps(
        {
            "total": len(_chunk_index),
            "returned": len(chunks),
            "chunks": chunks,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
async def compression_stats() -> str:
    """Global compression statistics.

    Returns total pages compressed, total chunks, average compression ratio,
    estimated tokens saved, and count of stale chunks.
    """
    total_chunks = len(_chunk_index)
    if total_chunks == 0:
        return json.dumps(
            {
                "total_chunks": 0,
                "message": "No chunks compressed yet.",
            },
            indent=2,
        )

    total_original = sum(c.get("original_tokens", 0) for c in _chunk_index.values())
    total_compressed = sum(c.get("compressed_tokens", 0) for c in _chunk_index.values())
    avg_ratio = (
        sum(c.get("actual_ratio", 0) for c in _chunk_index.values()) / total_chunks
    )
    avg_confidence = (
        sum(c.get("confidence", 0) for c in _chunk_index.values()) / total_chunks
    )
    unique_sources = len(set(c["source_path"] for c in _chunk_index.values()))

    # Count stale chunks
    stale_count = sum(1 for c in _chunk_index.values() if _check_staleness(c))

    return json.dumps(
        {
            "total_chunks": total_chunks,
            "unique_sources": unique_sources,
            "total_original_tokens": total_original,
            "total_compressed_tokens": total_compressed,
            "tokens_saved": total_original - total_compressed,
            "avg_compression_ratio": round(avg_ratio, 2),
            "avg_confidence": round(avg_confidence, 3),
            "stale_chunks": stale_count,
            "store_path": str(CHUNK_STORE),
        },
        indent=2,
    )


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
            stale.append(
                {
                    "chunk_id": cid,
                    "source_path": meta.get("source_path", ""),
                    "created_at": meta.get("created_at", ""),
                    "reason": (
                        "source deleted"
                        if not Path(meta.get("source_path", "")).exists()
                        else "content changed"
                    ),
                }
            )

    if not dry_run:
        with sqlite3.connect(DB_PATH) as conn:
            for entry in stale:
                cid = entry["chunk_id"]
                conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (cid,))
                _chunk_index.pop(cid, None)

        _rebuild_search_index()

    return json.dumps(
        {
            "dry_run": dry_run,
            "stale_count": len(stale),
            "stale_chunks": stale,
            "message": (
                f"Would purge {len(stale)} stale chunk(s)."
                if dry_run
                else f"Purged {len(stale)} stale chunk(s)."
            ),
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Start the context-compressor MCP server."""
    _init_db()
    _migrate_json_to_db()

    # Load manifest on startup from DB
    _chunk_index.update(_load_from_db())

    # Build search index from existing chunks
    _rebuild_search_index()

    mcp.run()


if __name__ == "__main__":
    main()
