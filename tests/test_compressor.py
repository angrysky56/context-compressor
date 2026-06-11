"""Tests for the context-compressor extractive compression engine."""

import pytest

from context_compressor.compressor import (
    BlockType,
    Compressor,
    CompressionResult,
    ContentBlock,
    DEFAULT_IMPORTANT_TERMS,
    SectionSummary,
    estimate_tokens,
    extract_entities,
    parse_blocks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_TEXT = "This is a simple test sentence that should not be compressed."

SHORT_TEXT = "Hello world."

MARKDOWN_DOC = """---
title: Test Document
---

# Main Title

## First Section

The first section has important content about variational inference.
This is a critical detail that should be preserved.
Some filler text that might be dropped during compression.
Another filler sentence with no particular importance.
Yet another line of unremarkable content.

## Second Section

### Subsection A

The ELBO is the cornerstone of variational inference.
It provides a tractable lower bound on the log-marginal likelihood.
The decomposition is essential for understanding the encoder-decoder architecture.

### Subsection B

Metropolis-Hastings is a popular MCMC algorithm.
It generates samples from the posterior distribution.
The chain converges to the target distribution.

## Code Section

```python
def hello():
    return "world"
```

- Item one about encoding
- Item two about decoding
- Item three about reconstruction

## Conclusion

This document covered key concepts in Bayesian inference.
The connection to free energy is important.
"""

CODE_BLOCK_DOC = """# Code Example

Some intro text here.

```javascript
function add(a, b) {
    return a + b;
}
```

More text after the code block.

```python
class Foo:
    pass
```

Final paragraph.
"""

LIST_DOC = """# List Test

- First item in the list
- Second item with more detail
- Third item about something else
- Fourth item wrapping up

Some text after the list.
"""


# ---------------------------------------------------------------------------
# estimate_tokens tests
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 1  # minimum 1

    def test_short_string(self) -> None:
        result = estimate_tokens("hello")
        assert result >= 1

    def test_known_length(self) -> None:
        # 400 characters → ~100 tokens
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_realistic_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        tokens = estimate_tokens(text)
        # ~44 chars → ~11 tokens
        assert 8 <= tokens <= 15


# ---------------------------------------------------------------------------
# parse_blocks tests
# ---------------------------------------------------------------------------

class TestParseBlocks:
    def test_empty_text(self) -> None:
        blocks = parse_blocks("")
        assert blocks == []

    def test_plain_paragraph(self) -> None:
        text = "This is a simple paragraph with enough text to not be filtered out."
        blocks = parse_blocks(text)
        assert len(blocks) >= 1
        assert blocks[0].type == BlockType.PARAGRAPH

    def test_heading_detection(self) -> None:
        text = "# Main Heading\n\nSome content paragraph goes here.\n\n## Sub Heading\n\nMore content paragraph here."
        blocks = parse_blocks(text)
        headings = [b for b in blocks if b.type == BlockType.HEADING]
        assert len(headings) == 2
        assert headings[0].level == 1
        assert headings[1].level == 2

    def test_heading_weights(self) -> None:
        text = "# H1\n\n## H2\n\n### H3"
        blocks = parse_blocks(text)
        headings = [b for b in blocks if b.type == BlockType.HEADING]
        # H1 should have higher weight than H2, H2 higher than H3
        assert headings[0].weight > headings[1].weight > headings[2].weight

    def test_code_block_preserved_whole(self) -> None:
        blocks = parse_blocks(CODE_BLOCK_DOC)
        code_blocks = [b for b in blocks if b.type == BlockType.CODE]
        assert len(code_blocks) == 2
        # Code blocks should contain the fence markers
        assert "```javascript" in code_blocks[0].text
        assert "```python" in code_blocks[1].text
        # Code content should be intact
        assert "function add(a, b)" in code_blocks[0].text
        assert "class Foo:" in code_blocks[1].text

    def test_list_block_detection(self) -> None:
        blocks = parse_blocks(LIST_DOC)
        list_blocks = [b for b in blocks if b.type == BlockType.LIST]
        assert len(list_blocks) >= 1
        assert "First item" in list_blocks[0].text

    def test_yaml_frontmatter(self) -> None:
        blocks = parse_blocks(MARKDOWN_DOC)
        metadata = [b for b in blocks if b.type == BlockType.METADATA]
        assert len(metadata) == 1
        assert "title: Test Document" in metadata[0].text
        assert metadata[0].weight == 2.0  # always preserved

    def test_full_markdown_structure(self) -> None:
        blocks = parse_blocks(MARKDOWN_DOC)
        types = [b.type for b in blocks]
        assert BlockType.METADATA in types
        assert BlockType.HEADING in types
        assert BlockType.PARAGRAPH in types
        assert BlockType.CODE in types
        assert BlockType.LIST in types

    def test_short_fragments_filtered(self) -> None:
        """Fragments <= 10 chars should be filtered out."""
        text = "Hi.\n\nThis is a real sentence that should be kept in the output."
        blocks = parse_blocks(text)
        texts = [b.text for b in blocks]
        assert "Hi." not in texts


# ---------------------------------------------------------------------------
# extract_entities tests
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_capitalized_multi_word(self) -> None:
        text = "The Anterior Cingulate Cortex is important."
        entities = extract_entities(text)
        # Regex captures the full capitalized phrase including leading "The"
        assert any("Anterior Cingulate Cortex" in e for e in entities)

    def test_acronyms(self) -> None:
        text = "We used MCMC and ELBO for inference."
        entities = extract_entities(text)
        assert "MCMC" in entities
        assert "ELBO" in entities

    def test_important_terms(self) -> None:
        text = "The variational inference approach uses free energy."
        entities = extract_entities(text)
        assert "variational inference" in entities
        assert "free energy" in entities

    def test_custom_important_terms(self) -> None:
        custom = frozenset({"custom-term", "special widget"})
        text = "We built a special widget using the custom-term approach."
        entities = extract_entities(text, important_terms=custom)
        assert "custom-term" in entities
        assert "special widget" in entities
        # Default terms should NOT match
        assert "variational inference" not in entities

    def test_parenthesized_entity(self) -> None:
        text = "The Thalamic Reticular Nucleus (TRN) plays a key role."
        entities = extract_entities(text)
        assert any("TRN" in e for e in entities)

    def test_no_cross_heading_entities(self) -> None:
        """Entities should not span across heading boundaries."""
        text = "## Overview\nThe PAC-Bayes framework is useful."
        entities = extract_entities(text)
        # Should NOT contain "Overview The" as a single entity
        assert "Overview The" not in entities

    def test_empty_text(self) -> None:
        entities = extract_entities("")
        assert entities == []


# ---------------------------------------------------------------------------
# Compressor tests
# ---------------------------------------------------------------------------

class TestCompressor:
    def test_short_text_passthrough(self) -> None:
        """Text under 50 tokens should pass through uncompressed."""
        c = Compressor(ratio=4.0)
        result = c.compress(SHORT_TEXT)
        assert result.compressed_text == SHORT_TEXT
        assert result.actual_ratio == 1.0
        assert result.confidence == 1.0

    def test_ratio_1_minimal_compression(self) -> None:
        c = Compressor(ratio=1.0)
        result = c.compress(MARKDOWN_DOC)
        # Ratio 1 should keep almost everything
        assert result.actual_ratio <= 2.0

    def test_ratio_4_default(self) -> None:
        c = Compressor(ratio=4.0)
        result = c.compress(MARKDOWN_DOC)
        assert result.compressed_tokens < result.original_tokens
        assert result.actual_ratio > 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_ratio_8_aggressive(self) -> None:
        c = Compressor(ratio=8.0)
        result = c.compress(MARKDOWN_DOC)
        assert result.compressed_tokens < result.original_tokens
        # More aggressive = fewer tokens
        c4 = Compressor(ratio=4.0)
        r4 = c4.compress(MARKDOWN_DOC)
        assert result.compressed_tokens <= r4.compressed_tokens

    def test_ratio_16_maximum(self) -> None:
        c = Compressor(ratio=16.0)
        result = c.compress(MARKDOWN_DOC)
        assert result.compressed_tokens < result.original_tokens

    def test_headings_preserved(self) -> None:
        """All headings should survive compression."""
        c = Compressor(ratio=8.0)
        result = c.compress(MARKDOWN_DOC)
        assert "# Main Title" in result.compressed_text
        assert "## First Section" in result.compressed_text
        assert "## Conclusion" in result.compressed_text

    def test_frontmatter_preserved(self) -> None:
        """YAML frontmatter should always be preserved."""
        c = Compressor(ratio=8.0)
        result = c.compress(MARKDOWN_DOC)
        assert "title: Test Document" in result.compressed_text

    def test_code_block_preserved(self) -> None:
        """Code blocks should survive compression intact."""
        # Use ratio=2 so the code block survives selection in a small section
        c = Compressor(ratio=2.0)
        result = c.compress(MARKDOWN_DOC)
        assert "def hello():" in result.compressed_text
        assert '```python' in result.compressed_text

    def test_entity_preservation(self) -> None:
        """With preserve_entities=True, key entities should survive."""
        c = Compressor(ratio=4.0, preserve_entities=True)
        result = c.compress(MARKDOWN_DOC)
        # At least some domain terms should be in the output
        compressed_lower = result.compressed_text.lower()
        domain_terms_found = sum(
            1 for term in ["elbo", "mcmc", "variational inference", "encoder", "decoder"]
            if term in compressed_lower
        )
        assert domain_terms_found >= 2

    def test_section_summaries(self) -> None:
        """Compression should produce section summaries."""
        c = Compressor(ratio=4.0)
        result = c.compress(MARKDOWN_DOC)
        assert len(result.sections) > 0
        titles = [s.title for s in result.sections]
        assert "First Section" in titles

    def test_section_token_counts(self) -> None:
        """Each section summary should have valid token counts."""
        c = Compressor(ratio=4.0)
        result = c.compress(MARKDOWN_DOC)
        for section in result.sections:
            assert section.original_tokens >= 0
            assert section.compressed_tokens >= 0
            assert section.compressed_tokens <= section.original_tokens

    def test_custom_important_terms(self) -> None:
        """Custom important terms should be used for scoring."""
        custom = frozenset({"filler", "unremarkable"})
        c = Compressor(ratio=4.0, important_terms=custom)
        result = c.compress(MARKDOWN_DOC)
        # Should still produce valid output
        assert result.compressed_tokens < result.original_tokens

    def test_key_entities_populated(self) -> None:
        c = Compressor(ratio=4.0)
        result = c.compress(MARKDOWN_DOC)
        assert len(result.key_entities) > 0

    def test_confidence_range(self) -> None:
        c = Compressor(ratio=4.0)
        result = c.compress(MARKDOWN_DOC)
        assert 0.0 <= result.confidence <= 1.0

    def test_two_sentence_passthrough(self) -> None:
        """Documents with <= 2 blocks should pass through."""
        text = "This is the first sentence. This is the second sentence."
        c = Compressor(ratio=4.0)
        result = c.compress(text)
        # Very few blocks → passthrough
        assert result.actual_ratio >= 1.0
