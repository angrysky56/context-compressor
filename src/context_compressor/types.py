"""Pydantic models for context-compressor."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SectionInfo(BaseModel):
    """Summary of a compressed section in the output."""
    title: str = Field(description="Section heading text")
    level: int = Field(ge=1, le=6, description="Heading depth (1-6)")
    original_tokens: int = Field(ge=0, description="Token count of original section body")
    compressed_tokens: int = Field(ge=0, description="Token count of compressed section body")


class ChunkMetadata(BaseModel):
    """Metadata for a compressed chunk."""
    chunk_id: str
    source_path: str
    ratio: float = Field(ge=1.0, le=16.0, description="Target compression ratio")
    original_tokens: int = Field(ge=0, description="Token count of original content")
    compressed_tokens: int = Field(ge=0, description="Token count of compressed content")
    actual_ratio: float = Field(ge=1.0, description="Achieved compression ratio")
    key_entities: list[str] = Field(default_factory=list, description="Named entities preserved")
    confidence: float = Field(ge=0.0, le=1.0, description="Compression quality estimate")
    created_at: str = Field(description="ISO timestamp")
    interleaved: bool = Field(default=False, description="Whether this was interleaved with other chunks")
    content_hash: str = Field(default="", description="SHA-256 hash of original file content")
    is_stale: bool = Field(default=False, description="True if source file has changed since compression")
    sections: list[SectionInfo] = Field(default_factory=list, description="Section summaries")


class CompressionRequest(BaseModel):
    """Request to compress one or more files."""
    paths: list[str] = Field(min_length=1, description="File paths to compress")
    ratio: float = Field(default=4.0, ge=1.0, le=16.0, description="Target compression ratio")
    interleave: bool = Field(default=False, description="Interleave chunks (LCLM-style)")
    preserve_entities: bool = Field(default=True, description="Always preserve named entities")


class CompressionStats(BaseModel):
    """Global compression statistics."""
    total_chunks: int
    unique_sources: int
    total_original_tokens: int
    total_compressed_tokens: int
    tokens_saved: int
    avg_compression_ratio: float
    avg_confidence: float
    store_path: str
    stale_chunks: int = Field(default=0, description="Number of chunks whose source has changed")
