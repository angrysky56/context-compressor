"""Tests for the context-compressor MCP server tools."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from context_compressor.server import (
    _chunk_id,
    _content_hash,
    _find_existing_chunk,
    _check_staleness,
    _section_summaries_to_dicts,
    _interleave_results,
    _chunk_index,
    _load_manifest,
    _save_manifest,
    _rebuild_search_index,
    compress_pages,
    compress_text,
    expand_chunk,
    get_chunk_metadata,
    search_chunks,
    list_chunks,
    compression_stats,
    purge_stale,
    CHUNK_STORE,
    MANIFEST_PATH,
)
from context_compressor.compressor import SectionSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path):
    """Create a temp directory with test files."""
    doc = tmp_path / "test_doc.md"
    doc.write_text("""# Test Document

## Section One

The PAC-Bayes framework provides generalization bounds for Bayesian learning.
These bounds are critical for understanding model complexity.
The connection to variational inference has been explored extensively.
Free energy minimization is a key concept in this framework.
The ELBO provides a tractable lower bound on the log-marginal likelihood.

## Section Two

Metropolis-Hastings is a popular MCMC algorithm for sampling.
It generates samples from the posterior distribution efficiently.
The chain is designed to converge to the target distribution.
Convergence diagnostics are critical for reliable inference.

## Conclusion

This document covered key concepts in Bayesian inference.
""")

    doc2 = tmp_path / "test_doc2.md"
    doc2.write_text("""# Second Document

## Overview

This is a completely different document about neural networks.
Deep learning has revolutionized many fields of AI.
The encoder-decoder architecture is widely used.
Attention mechanisms have improved sequence modeling.
Transformer models achieve state-of-the-art results.
""")

    return tmp_path


@pytest.fixture(autouse=True)
def clean_chunk_index():
    """Clear the chunk index before each test."""
    import context_compressor.server as srv
    original_index = dict(srv._chunk_index)
    srv._chunk_index.clear()
    yield
    srv._chunk_index.clear()
    srv._chunk_index.update(original_index)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self) -> None:
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_different_content(self) -> None:
        h1 = _content_hash("hello")
        h2 = _content_hash("world")
        assert h1 != h2

    def test_sha256_length(self) -> None:
        h = _content_hash("test")
        assert len(h) == 64  # SHA-256 hex digest


class TestChunkId:
    def test_deterministic(self) -> None:
        id1 = _chunk_id("path", 4.0, "2024-01-01")
        id2 = _chunk_id("path", 4.0, "2024-01-01")
        assert id1 == id2

    def test_different_inputs(self) -> None:
        id1 = _chunk_id("path1", 4.0, "2024-01-01")
        id2 = _chunk_id("path2", 4.0, "2024-01-01")
        assert id1 != id2

    def test_length(self) -> None:
        cid = _chunk_id("path", 4.0, "2024-01-01")
        assert len(cid) == 16


class TestSectionSummariesToDicts:
    def test_conversion(self) -> None:
        sections = [
            SectionSummary(
                title="Test",
                level=2,
                compressed_body="body",
                original_tokens=100,
                compressed_tokens=25,
            )
        ]
        result = _section_summaries_to_dicts(sections)
        assert len(result) == 1
        assert result[0]["title"] == "Test"
        assert result[0]["level"] == 2
        assert result[0]["original_tokens"] == 100
        assert result[0]["compressed_tokens"] == 25

    def test_empty(self) -> None:
        assert _section_summaries_to_dicts([]) == []


class TestInterleaveResults:
    def test_single_source(self) -> None:
        results = [{"source_path": "a"}, {"source_path": "a"}]
        assert _interleave_results(results) == results

    def test_two_sources(self) -> None:
        results = [
            {"source_path": "a", "id": 1},
            {"source_path": "a", "id": 2},
            {"source_path": "b", "id": 3},
            {"source_path": "b", "id": 4},
        ]
        interleaved = _interleave_results(results)
        # Should alternate between sources
        assert len(interleaved) == 4
        sources = [r["source_path"] for r in interleaved]
        # First two should be different sources
        assert sources[0] != sources[1]

    def test_empty(self) -> None:
        assert _interleave_results([]) == []


# ---------------------------------------------------------------------------
# Tool tests (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCompressPages:
    async def test_compress_single_file(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        result_json = await compress_pages(paths=[str(doc)])
        result = json.loads(result_json)

        assert result["chunks_created"] == 1
        assert len(result["errors"]) == 0
        assert len(result["chunks"]) == 1

        chunk = result["chunks"][0]
        assert "chunk_id" in chunk
        assert chunk["original_tokens"] > 0
        assert chunk["compressed_tokens"] > 0
        assert chunk["actual_ratio"] > 1.0

    async def test_file_not_found(self) -> None:
        result_json = await compress_pages(paths=["/nonexistent/file.md"])
        result = json.loads(result_json)
        assert result["chunks_created"] == 0
        assert len(result["errors"]) == 1

    async def test_deduplication(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        # Compress twice
        r1_json = await compress_pages(paths=[str(doc)])
        r2_json = await compress_pages(paths=[str(doc)])
        r1 = json.loads(r1_json)
        r2 = json.loads(r2_json)

        # Second call should return deduplicated chunk
        assert r2["chunks"][0].get("deduplicated") is True
        assert r1["chunks"][0]["chunk_id"] == r2["chunks"][0]["chunk_id"]

    async def test_custom_ratio(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        result_json = await compress_pages(paths=[str(doc)], ratio=8.0)
        result = json.loads(result_json)
        assert result["chunks_created"] == 1

    async def test_interleave(self, tmp_dir: Path) -> None:
        doc1 = tmp_dir / "test_doc.md"
        doc2 = tmp_dir / "test_doc2.md"
        result_json = await compress_pages(
            paths=[str(doc1), str(doc2)], interleave=True
        )
        result = json.loads(result_json)
        assert result["interleaved"] is True
        assert result["chunks_created"] == 2

    async def test_sections_in_output(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        result_json = await compress_pages(paths=[str(doc)])
        result = json.loads(result_json)
        chunk = result["chunks"][0]
        assert "sections" in chunk
        assert len(chunk["sections"]) > 0


@pytest.mark.asyncio
class TestCompressText:
    async def test_inline_compression(self) -> None:
        text = """# Test

## Section A

The PAC-Bayes framework provides generalization bounds for Bayesian learning.
These bounds are critical for understanding model complexity.
The connection to variational inference has been explored extensively.
Free energy minimization is a key concept.
The ELBO provides a tractable lower bound.

## Section B

More content here about different things.
Metropolis-Hastings is a sampling algorithm.
MCMC methods are widely used in statistics.
"""
        result_json = await compress_text(text=text)
        result = json.loads(result_json)

        assert result["original_tokens"] > 0
        assert result["compressed_tokens"] > 0
        assert "compressed_text" in result
        assert "chunk_id" not in result  # not persisted by default

    async def test_persist_flag(self) -> None:
        text = "A long text " * 50  # ensure enough for compression
        result_json = await compress_text(
            text=text, persist=True, label="test-inline"
        )
        result = json.loads(result_json)
        assert "chunk_id" in result
        assert result["persisted"] is True


@pytest.mark.asyncio
class TestExpandChunk:
    async def test_expand_existing(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        compress_result = json.loads(
            await compress_pages(paths=[str(doc)])
        )
        cid = compress_result["chunks"][0]["chunk_id"]

        expand_result = json.loads(await expand_chunk(chunk_id=cid))
        assert "content" in expand_result
        assert expand_result["chunk_id"] == cid
        assert len(expand_result["content"]) > 0

    async def test_expand_nonexistent(self) -> None:
        result = json.loads(await expand_chunk(chunk_id="nonexistent123"))
        assert "error" in result


@pytest.mark.asyncio
class TestGetChunkMetadata:
    async def test_get_metadata(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        compress_result = json.loads(
            await compress_pages(paths=[str(doc)])
        )
        cid = compress_result["chunks"][0]["chunk_id"]

        meta_result = json.loads(await get_chunk_metadata(chunk_id=cid))
        assert meta_result["chunk_id"] == cid
        assert "content_hash" in meta_result
        assert "sections" in meta_result

    async def test_nonexistent(self) -> None:
        result = json.loads(await get_chunk_metadata(chunk_id="nonexistent"))
        assert "error" in result


@pytest.mark.asyncio
class TestSearchChunks:
    async def test_search_no_index(self) -> None:
        import context_compressor.server as srv
        # Ensure search index is empty for this test
        srv._search_index.vectorizer = None
        srv._search_index.matrix = None
        srv._search_index.chunk_ids = []

        result = json.loads(await search_chunks(query="test"))
        assert result.get("error") or result.get("results") == []

    async def test_search_after_compress(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        await compress_pages(paths=[str(doc)])

        result = json.loads(
            await search_chunks(query="PAC-Bayes variational inference")
        )
        assert "results" in result
        assert result["returned"] >= 1


@pytest.mark.asyncio
class TestListChunks:
    async def test_empty_list(self) -> None:
        result = json.loads(await list_chunks())
        assert result["total"] == 0
        assert result["returned"] == 0

    async def test_list_after_compress(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        await compress_pages(paths=[str(doc)])

        result = json.loads(await list_chunks())
        assert result["total"] >= 1
        assert result["returned"] >= 1

    async def test_source_prefix_filter(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        await compress_pages(paths=[str(doc)])

        # Filter by correct prefix
        result = json.loads(await list_chunks(source_prefix=str(tmp_dir)))
        assert result["returned"] >= 1

        # Filter by wrong prefix
        result = json.loads(
            await list_chunks(source_prefix="/nonexistent/path")
        )
        assert result["returned"] == 0


@pytest.mark.asyncio
class TestCompressionStats:
    async def test_empty_stats(self) -> None:
        result = json.loads(await compression_stats())
        assert result["total_chunks"] == 0

    async def test_stats_after_compress(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        await compress_pages(paths=[str(doc)])

        result = json.loads(await compression_stats())
        assert result["total_chunks"] >= 1
        assert result["tokens_saved"] >= 0
        assert "stale_chunks" in result


@pytest.mark.asyncio
class TestPurgeStale:
    async def test_dry_run(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        await compress_pages(paths=[str(doc)])

        # Modify the source file to make chunk stale
        doc.write_text("Completely different content now.")

        result = json.loads(await purge_stale(dry_run=True))
        assert result["dry_run"] is True
        assert result["stale_count"] >= 1

    async def test_actual_purge(self, tmp_dir: Path) -> None:
        doc = tmp_dir / "test_doc.md"
        await compress_pages(paths=[str(doc)])

        # Delete the source file
        doc.unlink()

        result = json.loads(await purge_stale(dry_run=False))
        assert result["dry_run"] is False
        assert result["stale_count"] >= 1

        # Verify chunk was removed
        stats = json.loads(await compression_stats())
        assert stats["total_chunks"] == 0
