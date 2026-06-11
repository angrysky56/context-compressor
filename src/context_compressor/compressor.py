"""Extractive compression engine — Phase 1.

Implements TF-IDF sentence scoring + entity preservation for compressing
wiki pages and carryover files into latent chunk summaries.

Phase 2 will replace this with a learned LCLM encoder (0.6B model from
arXiv 2606.09659) for true p(x|z) reconstruction.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Result of compressing a single document."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    actual_ratio: float
    key_entities: list[str]
    confidence: float  # 0-1 estimate of compression quality


# ---------------------------------------------------------------------------
# Token estimation (cheap, no tokenizer dependency)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Split on sentence boundaries, keeping the delimiter
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|\n+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split on sentence boundaries
    raw = _SENTENCE_RE.split(text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s and len(s) > 10:  # skip very short fragments
            sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Entity extraction (simple regex-based, no NER dependency)
# ---------------------------------------------------------------------------

# Capitalized multi-word phrases (potential named entities)
_ENTITY_RE = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'  # "Anterior Cingulate Cortex"
    r'|\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b'       # "ACC", "PAC-Bayes"
    r'|\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\([^)]+\))'  # "Thalamic Reticular Nucleus (TRN)"
)

# Known important terms to always preserve
_IMPORTANT_TERMS = {
    'elbo', 'pac-bayes', 'trn', 'acc', 'mcmc', 'metropolis-hastings',
    'variational inference', 'free energy', 'predictive coding',
    'compression ratio', 'context window', 'carryover', 'markovian',
    'latent space', 'encoder', 'decoder', 'reconstruction',
    'epistemic', 'uncertainty', 'confidence', 'entropy',
    'wiki', 'neo4j', 'chroma', 'synapse', 'hermes',
}


def extract_entities(text: str) -> list[str]:
    """Extract named entities and important terms from text."""
    entities = set()

    for match in _ENTITY_RE.finditer(text):
        entity = next(g for g in match.groups() if g is not None)
        entity = entity.strip()
        if len(entity) > 2:
            entities.add(entity)

    # Also check for important terms (case-insensitive)
    text_lower = text.lower()
    for term in _IMPORTANT_TERMS:
        if term in text_lower:
            entities.add(term)

    return sorted(entities)


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r'\b[a-z][a-z0-9_\-]{1,}\b', text.lower())


def _compute_idf(sentences: list[str]) -> dict[str, float]:
    """Compute inverse document frequency across sentences."""
    n = len(sentences)
    if n == 0:
        return {}

    # Count document frequency for each term
    df: dict[str, int] = {}
    for sent in sentences:
        tokens = set(_tokenize(sent))
        for tok in tokens:
            df[tok] = df.get(tok, 0) + 1

    # IDF = log(N / df)
    idf: dict[str, float] = {}
    for tok, count in df.items():
        idf[tok] = math.log((n + 1) / (count + 1)) + 1  # smoothed

    return idf


def score_sentences(sentences: list[str], idf: dict[str, float]) -> list[tuple[int, float]]:
    """Score each sentence by TF-IDF sum."""
    scored = []
    for i, sent in enumerate(sentences):
        tokens = _tokenize(sent)
        if not tokens:
            scored.append((i, 0.0))
            continue

        # TF (term frequency in this sentence)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        # TF-IDF score
        score = 0.0
        for tok, count in tf.items():
            score += count * idf.get(tok, 1.0)

        # Normalize by sentence length (avoid bias toward long sentences)
        score /= math.sqrt(len(tokens))

        # Bonus for sentences with entities
        entities = extract_entities(sent)
        if entities:
            score *= 1.2

        # Bonus for sentences with structural markers
        if re.search(r'\b(important|critical|key|essential|note|warning|caveat)\b', sent.lower()):
            score *= 1.3

        # Slight bonus for first and last sentences (often contain summaries)
        if i == 0:
            score *= 1.15
        if i == len(sentences) - 1:
            score *= 1.05

        scored.append((i, score))

    return scored


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class Compressor:
    """Extractive compressor — Phase 1.

    Uses TF-IDF sentence scoring to select the most informative sentences,
    preserving named entities and key facts.

    Args:
        ratio: Target compression ratio (1-16). Higher = more aggressive.
        preserve_entities: If True, always include sentences containing
            named entities even if their TF-IDF score is low.
    """

    def __init__(self, ratio: float = 4.0, preserve_entities: bool = True):
        self.ratio = ratio
        self.preserve_entities = preserve_entities

    def compress(self, text: str) -> CompressionResult:
        """Compress text to approximately 1/ratio of original length."""
        original_tokens = estimate_tokens(text)

        # Handle very short texts
        if original_tokens < 50:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                actual_ratio=1.0,
                key_entities=extract_entities(text),
                confidence=1.0,
            )

        # Split into sentences
        sentences = split_sentences(text)
        if len(sentences) <= 2:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                actual_ratio=1.0,
                key_entities=extract_entities(text),
                confidence=0.9,
            )

        # Compute IDF and score sentences
        idf = _compute_idf(sentences)
        scored = score_sentences(sentences, idf)

        # Determine how many sentences to keep
        target_sentences = max(1, int(len(sentences) / self.ratio))

        # Sort by score, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_indices = set(idx for idx, _ in scored[:target_sentences])

        # If preserving entities, ensure entity-bearing sentences are included
        if self.preserve_entities:
            for i, sent in enumerate(sentences):
                if extract_entities(sent) and i not in selected_indices:
                    # Replace the lowest-scored selected sentence
                    if len(selected_indices) >= target_sentences:
                        # Find lowest scored in selected
                        lowest_idx = min(selected_indices, key=lambda idx: next(s for i, s in scored if i == idx))
                        selected_indices.discard(lowest_idx)
                    selected_indices.add(i)

        # Reconstruct in original order
        selected_indices = sorted(selected_indices)
        compressed_sentences = [sentences[i] for i in selected_indices]
        compressed_text = ' '.join(compressed_sentences)

        compressed_tokens = estimate_tokens(compressed_text)
        actual_ratio = max(1.0, original_tokens / max(1, compressed_tokens))

        # Confidence: how well did we hit the target?
        ratio_accuracy = 1.0 - min(1.0, abs(actual_ratio - self.ratio) / self.ratio)
        entity_coverage = 1.0
        all_entities = extract_entities(text)
        if all_entities:
            preserved = extract_entities(compressed_text)
            entity_coverage = len(preserved) / len(all_entities)

        confidence = 0.5 * ratio_accuracy + 0.5 * entity_coverage

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            actual_ratio=round(actual_ratio, 2),
            key_entities=extract_entities(compressed_text),
            confidence=round(confidence, 3),
        )
