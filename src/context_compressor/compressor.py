"""Extractive compression engine — Phase 1.

Implements structure-aware Markdown parsing + TF-IDF sentence scoring
+ entity preservation for compressing wiki pages and carryover files
into latent chunk summaries with hierarchical section structure.

Phase 2 will replace this with a learned LCLM encoder (0.6B model from
arXiv 2606.09659) for true p(x|z) reconstruction.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class BlockType(Enum):
    """Types of content blocks recognized in Markdown."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE = "code"
    LIST = "list"
    METADATA = "metadata"  # YAML frontmatter


@dataclass
class ContentBlock:
    """A structural unit of content parsed from Markdown.

    Attributes:
        type: The structural role of this block.
        text: Raw text content of the block.
        level: Heading depth (1-6) for HEADING blocks, 0 otherwise.
        weight: Importance multiplier for scoring. Headings and metadata
            get a permanent boost; other blocks start at 1.0.
    """
    type: BlockType
    text: str
    level: int = 0
    weight: float = 1.0


@dataclass
class SectionSummary:
    """Summary of a compressed section for hierarchical output.

    Attributes:
        title: The section heading text.
        level: Heading depth (1-6).
        compressed_body: The compressed content under this heading.
        original_tokens: Token count of the original section body.
        compressed_tokens: Token count after compression.
    """
    title: str
    level: int
    compressed_body: str
    original_tokens: int
    compressed_tokens: int


@dataclass
class CompressionResult:
    """Result of compressing a single document.

    Attributes:
        original_text: The full input text.
        compressed_text: The compressed output, preserving document outline.
        original_tokens: Estimated token count of original.
        compressed_tokens: Estimated token count of compressed output.
        actual_ratio: Achieved compression ratio (original / compressed).
        key_entities: Named entities and important terms preserved.
        confidence: 0-1 estimate of compression quality (ratio accuracy
            × entity coverage).
        sections: List of section summaries for hierarchical output.
    """
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    actual_ratio: float
    key_entities: list[str]
    confidence: float
    sections: list[SectionSummary] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token estimation (cheap, no tokenizer dependency)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Markdown-aware block parsing
# ---------------------------------------------------------------------------

# Fenced code block: ``` or ~~~ with optional language tag
_CODE_FENCE_RE = re.compile(r'^(`{3,}|~{3,})', re.MULTILINE)

# Markdown heading: # through ######
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# YAML frontmatter: --- at start of file
_FRONTMATTER_RE = re.compile(r'\A---\n(.*?\n)---\n', re.DOTALL)

# Bullet or numbered list item
_LIST_ITEM_RE = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+', re.MULTILINE)

# Sentence boundary (for splitting within paragraphs)
_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'   # period/!/? followed by uppercase
    r'|(?<=\n)\n+'               # double newline
)

# Abbreviations that should NOT trigger sentence splits
_ABBREV_RE = re.compile(
    r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Fig|Eq|Ref|Vol)\.\s+'
)


def parse_blocks(text: str) -> list[ContentBlock]:
    """Parse Markdown text into structural content blocks.

    Recognizes:
    - YAML frontmatter (always preserved)
    - Fenced code blocks (preserved whole, never split)
    - Headings (always preserved, weighted by depth)
    - Bullet/numbered lists (kept or dropped as a unit)
    - Plain paragraphs (split into sentences for scoring)
    """
    blocks: list[ContentBlock] = []
    remaining = text

    # --- YAML frontmatter ---
    fm_match = _FRONTMATTER_RE.match(remaining)
    if fm_match:
        blocks.append(ContentBlock(
            type=BlockType.METADATA,
            text=fm_match.group(0).strip(),
            weight=2.0,  # always preserved
        ))
        remaining = remaining[fm_match.end():]

    # --- Process line by line, recognizing fenced code blocks ---
    lines = remaining.split('\n')
    i = 0
    current_paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        """Flush accumulated paragraph lines into blocks."""
        if not current_paragraph_lines:
            return
        para_text = '\n'.join(current_paragraph_lines).strip()
        if not para_text:
            current_paragraph_lines.clear()
            return

        # Check if this is a list block
        list_lines = [
            item for item in current_paragraph_lines if _LIST_ITEM_RE.match(item)
        ]
        if len(list_lines) > len(current_paragraph_lines) * 0.5:
            blocks.append(ContentBlock(
                type=BlockType.LIST,
                text=para_text,
                weight=0.9,  # slightly lower than prose
            ))
        else:
            # Split paragraph into sentences
            sentences = _split_paragraph_sentences(para_text)
            for sent in sentences:
                sent = sent.strip()
                if sent and len(sent) > 10:
                    blocks.append(ContentBlock(
                        type=BlockType.PARAGRAPH,
                        text=sent,
                        weight=1.0,
                    ))
        current_paragraph_lines.clear()

    while i < len(lines):
        line = lines[i]

        # Fenced code block?
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            flush_paragraph()
            fence_marker = fence_match.group(1)
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if lines[i].strip().startswith(
                    fence_marker[0] * len(fence_marker)
                ):
                    # Check it's a closing fence (same or more chars)
                    closing = lines[i].strip()
                    if (
                        closing == fence_marker[0] * len(closing)
                        and len(closing) >= len(fence_marker)
                    ):
                        i += 1
                        break
                i += 1
            blocks.append(ContentBlock(
                type=BlockType.CODE,
                text='\n'.join(code_lines),
                weight=1.1,  # slight bonus, code is often important
            ))
            continue

        # Heading?
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            # Higher weight for higher-level headings
            weight = 2.0 + (6 - level) * 0.3
            blocks.append(ContentBlock(
                type=BlockType.HEADING,
                text=line.strip(),
                level=level,
                weight=weight,
            ))
            i += 1
            continue

        # Blank line? Flush paragraph
        if not line.strip():
            flush_paragraph()
            i += 1
            continue

        # Otherwise accumulate as paragraph
        current_paragraph_lines.append(line)
        i += 1

    flush_paragraph()
    return blocks


def _split_paragraph_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, handling abbreviations.

    Avoids splitting on common abbreviations like 'e.g.', 'Dr.', etc.
    """
    # Protect abbreviations by replacing their periods temporarily
    protected = text
    for match in _ABBREV_RE.finditer(text):
        original = match.group(0)
        replacement = original.replace('. ', '.\u200B')  # zero-width space
        protected = protected.replace(original, replacement, 1)

    # Split on sentence boundaries
    raw = _SENTENCE_RE.split(protected)

    # Restore zero-width spaces
    return [s.replace('\u200B', '') for s in raw if s and s.strip()]


# ---------------------------------------------------------------------------
# Entity extraction (simple regex-based, no NER dependency)
# ---------------------------------------------------------------------------

# Capitalized multi-word phrases (potential named entities)
_ENTITY_RE = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'  # "Anterior Cingulate Cortex"
    r'|\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b'       # "ACC", "PAC-Bayes"
    r'|\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\([^)]+\))'  # "Thalamic Reticular Nucleus (TRN)"
)

# Default important terms to always preserve (domain-specific)
DEFAULT_IMPORTANT_TERMS: frozenset[str] = frozenset({
    'elbo', 'pac-bayes', 'trn', 'acc', 'mcmc', 'metropolis-hastings',
    'variational inference', 'free energy', 'predictive coding',
    'compression ratio', 'context window', 'carryover', 'markovian',
    'latent space', 'encoder', 'decoder', 'reconstruction',
    'epistemic', 'uncertainty', 'confidence', 'entropy',
    'wiki', 'neo4j', 'chroma', 'synapse', 'hermes',
})


def extract_entities(
    text: str,
    important_terms: frozenset[str] | None = None,
) -> list[str]:
    """Extract named entities and important terms from text.

    Args:
        text: Source text to scan for entities.
        important_terms: Optional set of domain terms to always detect
            (case-insensitive). Defaults to DEFAULT_IMPORTANT_TERMS.

    Returns:
        Sorted list of unique entities and terms found.
    """
    if important_terms is None:
        important_terms = DEFAULT_IMPORTANT_TERMS

    entities: set[str] = set()

    # Strip markdown heading markers and normalize newlines
    # This prevents false entities spanning heading→body boundaries
    lines = text.split('\n')
    content_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # Heading marker — insert a boundary so entities don't span
            content_lines.append('.')
        else:
            content_lines.append(line)
    normalized = ' '.join(content_lines)

    for match in _ENTITY_RE.finditer(normalized):
        entity = next(g for g in match.groups() if g is not None)
        entity = entity.strip()
        if len(entity) > 2:
            entities.add(entity)

    # Also check for important terms (case-insensitive)
    text_lower = text.lower()
    for term in important_terms:
        if term in text_lower:
            entities.add(term)

    return sorted(entities)


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r'\b[a-z][a-z0-9_\-]{1,}\b', text.lower())


def _compute_idf(texts: list[str]) -> dict[str, float]:
    """Compute inverse document frequency across text units.

    Args:
        texts: List of text segments (sentences or blocks) to compute IDF over.

    Returns:
        Mapping of token → IDF score (smoothed).
    """
    n = len(texts)
    if n == 0:
        return {}

    # Count document frequency for each term
    df: dict[str, int] = {}
    for text in texts:
        tokens = set(_tokenize(text))
        for tok in tokens:
            df[tok] = df.get(tok, 0) + 1

    # IDF = log(N / df) + 1 (smoothed)
    idf: dict[str, float] = {}
    for tok, count in df.items():
        idf[tok] = math.log((n + 1) / (count + 1)) + 1

    return idf


def score_block(
    block: ContentBlock,
    index: int,
    total_blocks: int,
    idf: dict[str, float],
    important_terms: frozenset[str] | None = None,
) -> float:
    """Score a content block for importance.

    Combines TF-IDF scoring with structural bonuses (block type weight,
    position, entity presence, structural markers).

    Args:
        block: The content block to score.
        index: Position of block in the document (0-indexed).
        total_blocks: Total number of blocks in the document.
        idf: Pre-computed IDF scores.
        important_terms: Custom important terms for entity detection.

    Returns:
        Importance score (higher = more important).
    """
    # Headings, metadata, and code blocks are always preserved
    if block.type in (BlockType.HEADING, BlockType.METADATA, BlockType.CODE):
        return float('inf')

    tokens = _tokenize(block.text)
    if not tokens:
        return 0.0

    # TF (term frequency in this block)
    tf: dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1

    # TF-IDF score
    score = 0.0
    for tok, count in tf.items():
        score += count * idf.get(tok, 1.0)

    # Normalize by block length (avoid bias toward long blocks)
    score /= math.sqrt(len(tokens))

    # Apply block type weight
    score *= block.weight

    # Bonus for blocks with entities
    entities = extract_entities(block.text, important_terms)
    if entities:
        score *= 1.2

    # Bonus for structural markers
    if re.search(
        r'\b(important|critical|key|essential|note|warning|caveat|'
        r'must|required|breaking|todo|fixme|hack)\b',
        block.text.lower(),
    ):
        score *= 1.3

    # Position bonuses
    if index == 0:
        score *= 1.15  # first block often contains summary
    if index == total_blocks - 1:
        score *= 1.05  # last block often contains conclusion

    # Code blocks get a slight bonus (usually task-relevant)
    if block.type == BlockType.CODE:
        score *= 1.1

    return score


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class Compressor:
    """Structure-aware extractive compressor — Phase 1.

    Uses Markdown-aware block parsing + TF-IDF scoring to select the most
    informative content blocks, preserving document structure (headings,
    code blocks, metadata) and named entities.

    Supports hierarchical compression: sections are compressed independently
    and the document outline is always preserved.

    Args:
        ratio: Target compression ratio (1-16). Higher = more aggressive.
        preserve_entities: If True, always include blocks containing
            named entities even if their TF-IDF score is low.
        important_terms: Custom set of domain-specific terms to always
            preserve. If None, uses DEFAULT_IMPORTANT_TERMS.
    """

    def __init__(
        self,
        ratio: float = 4.0,
        preserve_entities: bool = True,
        important_terms: frozenset[str] | None = None,
    ):
        self.ratio = ratio
        self.preserve_entities = preserve_entities
        self.important_terms = important_terms

    def compress(self, text: str) -> CompressionResult:
        """Compress text to approximately 1/ratio of original length.

        Performs hierarchical compression:
        1. Parse text into structural blocks (headings, code, lists, paragraphs)
        2. Group blocks by enclosing section (heading)
        3. Compress each section independently at the target ratio
        4. Preserve all headings to maintain document outline
        5. Reassemble with structure intact

        Args:
            text: The input text to compress (Markdown or plain text).

        Returns:
            CompressionResult with compressed text and metadata.
        """
        original_tokens = estimate_tokens(text)

        # Handle very short texts — no compression needed
        if original_tokens < 50:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                actual_ratio=1.0,
                key_entities=extract_entities(text, self.important_terms),
                confidence=1.0,
                sections=[],
            )

        # Parse into structural blocks
        blocks = parse_blocks(text)
        if len(blocks) <= 2:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                actual_ratio=1.0,
                key_entities=extract_entities(text, self.important_terms),
                confidence=0.9,
                sections=[],
            )

        # Group blocks into sections (by heading)
        sections = self._group_into_sections(blocks)

        # Compress each section independently
        compressed_parts: list[str] = []
        section_summaries: list[SectionSummary] = []

        for section_heading, section_blocks in sections:
            compressed_section, summary = self._compress_section(
                section_heading, section_blocks
            )
            compressed_parts.append(compressed_section)
            if summary:
                section_summaries.append(summary)

        compressed_text = '\n\n'.join(part for part in compressed_parts if part.strip())

        compressed_tokens = estimate_tokens(compressed_text)
        actual_ratio = max(1.0, original_tokens / max(1, compressed_tokens))

        # Compute confidence
        confidence = self._compute_confidence(
            text, compressed_text, actual_ratio
        )

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            actual_ratio=round(actual_ratio, 2),
            key_entities=extract_entities(compressed_text, self.important_terms),
            confidence=round(confidence, 3),
            sections=section_summaries,
        )

    def _group_into_sections(
        self, blocks: list[ContentBlock]
    ) -> list[tuple[ContentBlock | None, list[ContentBlock]]]:
        """Group blocks by their enclosing section heading.

        Returns a list of (heading_block_or_None, body_blocks) tuples.
        Blocks before the first heading go into a section with heading=None.
        """
        sections: list[tuple[ContentBlock | None, list[ContentBlock]]] = []
        current_heading: ContentBlock | None = None
        current_body: list[ContentBlock] = []

        for block in blocks:
            if block.type == BlockType.HEADING:
                # Save previous section
                if current_body or current_heading is not None:
                    sections.append((current_heading, current_body))
                current_heading = block
                current_body = []
            else:
                current_body.append(block)

        # Don't forget the last section
        if current_body or current_heading is not None:
            sections.append((current_heading, current_body))

        return sections

    def _compress_section(
        self,
        heading: ContentBlock | None,
        body: list[ContentBlock],
    ) -> tuple[str, SectionSummary | None]:
        """Compress a single section's body blocks.

        Always preserves the heading. Selects top-scoring body blocks
        to meet the target ratio.

        Returns:
            Tuple of (compressed section text, optional SectionSummary).
        """
        parts: list[str] = []

        # Always include the heading
        if heading:
            parts.append(heading.text)

        if not body:
            return '\n'.join(parts), None

        # Compute IDF across body blocks
        body_texts = [b.text for b in body]
        idf = _compute_idf(body_texts)

        # Score each block
        scored: list[tuple[int, float]] = []
        for i, block in enumerate(body):
            score = score_block(
                block, i, len(body), idf, self.important_terms
            )
            scored.append((i, score))

        # Always-preserve blocks (inf score): headings, metadata, code
        always_keep = {idx for idx, s in scored if s == float('inf')}

        # Determine how many additional blocks to keep
        scoreable = [(idx, s) for idx, s in scored if s != float('inf')]
        target_blocks = max(1, int(len(scoreable) / self.ratio))

        # Sort by score, take top N
        scoreable.sort(key=lambda x: x[1], reverse=True)
        selected = always_keep | {idx for idx, _ in scoreable[:target_blocks]}

        # Entity preservation: ensure entity-bearing blocks are included
        if self.preserve_entities:
            for i, block in enumerate(body):
                if i in selected:
                    continue
                if extract_entities(block.text, self.important_terms):
                    if len(selected) >= target_blocks + len(always_keep):
                        # Replace lowest-scored non-always-keep block
                        replaceable = selected - always_keep
                        if replaceable:
                            lowest = min(
                                replaceable,
                                key=lambda idx: next(
                                    (s for j, s in scored if j == idx), 0.0
                                ),
                            )
                            selected.discard(lowest)
                    selected.add(i)

        # Reconstruct in original order
        original_body_tokens = sum(estimate_tokens(b.text) for b in body)
        selected_blocks = [body[i] for i in sorted(selected)]
        body_text = ' '.join(b.text for b in selected_blocks)
        parts.append(body_text)

        compressed_body_tokens = estimate_tokens(body_text)

        summary = None
        if heading:
            summary = SectionSummary(
                title=heading.text.lstrip('#').strip(),
                level=heading.level,
                compressed_body=body_text,
                original_tokens=original_body_tokens,
                compressed_tokens=compressed_body_tokens,
            )

        return '\n'.join(parts), summary

    def _compute_confidence(
        self,
        original_text: str,
        compressed_text: str,
        actual_ratio: float,
    ) -> float:
        """Compute a 0-1 confidence score for compression quality.

        Combines ratio accuracy (did we hit the target?) with entity
        coverage (did we preserve important information?).
        """
        # How close did we get to the target ratio?
        ratio_accuracy = 1.0 - min(
            1.0, abs(actual_ratio - self.ratio) / self.ratio
        )

        # How many entities survived compression?
        entity_coverage = 1.0
        all_entities = extract_entities(original_text, self.important_terms)
        if all_entities:
            preserved = extract_entities(compressed_text, self.important_terms)
            entity_coverage = len(preserved) / len(all_entities)

        return 0.5 * ratio_accuracy + 0.5 * entity_coverage
