"""Skill Retrieval — keyword-based context chunking for token efficiency.

Instead of injecting entire documents (100K+ chars) into every API call,
this module splits documents into topic chunks and retrieves only the
chunks relevant to the user's message.

Architecture:
  1. At session start, large docs are split into SkillChunks by section headers.
  2. Each chunk has a keyword set for fast matching.
  3. On each user message, score chunks against message keywords.
  4. Only top-N chunks (up to a token budget) are injected into the system prompt.

No external dependencies — pure keyword matching, zero API calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum chars of context to inject per message (keeps prompt lean)
MAX_CONTEXT_CHARS = 12_000

# ── Text cleaning ─────────────────────────────────────────────────────────────


def clean_whitespace(text: str) -> str:
    """Collapse excessive whitespace that wastes tokens.

    - Collapses runs of 3+ spaces to a single space
    - Strips trailing spaces from each line
    - Collapses 3+ blank lines to 2
    """
    # Collapse multi-space runs (preserves intentional 2-space indents)
    text = re.sub(r" {3,}", " ", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ── Skill chunks ──────────────────────────────────────────────────────────────


@dataclass
class SkillChunk:
    """A topic chunk from a knowledge document."""

    title: str
    content: str
    keywords: set[str] = field(default_factory=set)

    @property
    def char_count(self) -> int:
        return len(self.content)


def split_into_chunks(text: str, min_chunk_chars: int = 200) -> list[SkillChunk]:
    """Split a document into chunks by markdown headers or bold section markers.

    Recognizes:
      - # / ## / ### headers
      - **Bold Section Title** on its own line
      - ALL CAPS LINES (>5 chars)
    """
    text = clean_whitespace(text)

    # Split on section boundaries
    pattern = re.compile(
        r"^(?=#{1,3}\s+.+$)"           # markdown headers
        r"|^(?=\*\*[A-Z].+\*\*\s*$)"   # **Bold Title**
        r"|^(?=[A-Z][A-Z\s:]{4,}$)",    # ALL CAPS HEADER
        re.MULTILINE,
    )

    parts = pattern.split(text)
    chunks: list[SkillChunk] = []

    for part in parts:
        part = part.strip()
        if not part or len(part) < min_chunk_chars:
            # Merge tiny fragments into previous chunk
            if chunks and len(part) > 10:
                chunks[-1].content += "\n\n" + part
                chunks[-1].keywords |= _extract_keywords(part)
            continue

        # Extract title from first line
        first_line = part.split("\n", 1)[0].strip().lstrip("#").strip().strip("*").strip()
        title = first_line[:80] if first_line else f"Section {len(chunks) + 1}"

        keywords = _extract_keywords(part)
        chunks.append(SkillChunk(title=title, content=part, keywords=keywords))

    if not chunks and text.strip():
        # Document has no section structure — treat as single chunk
        chunks.append(SkillChunk(
            title="General Reference",
            content=text,
            keywords=_extract_keywords(text),
        ))

    logger.info(
        "Split document into %d chunks (total %d chars)",
        len(chunks),
        sum(c.char_count for c in chunks),
    )
    return chunks


# ── Keyword extraction ────────────────────────────────────────────────────────

# Lab-specific terms that are strong signal for topic matching
_LAB_TERMS = {
    # Buffers & reagents
    "pbs", "facs", "buffer", "lysis", "staining", "blocking", "wash",
    "fixation", "permeabilization", "tris", "ripa", "sds", "bsa", "fbs",
    "edta", "hepes", "dmso", "tween", "triton", "formaldehyde", "pfa",
    "paraformaldehyde", "methanol", "ethanol", "rnase", "dnase",
    "protease", "inhibitor", "antibody", "isotype", "igg",
    # Techniques
    "centrifug", "incubat", "vortex", "pipett", "aliquot", "dilut",
    "resuspend", "pellet", "supernatant", "aspirat", "filter",
    "sort", "gating", "compensation", "viability", "zombie",
    # Equipment
    "cytometer", "centrifuge", "incubator", "microscope", "spectrophotometer",
    "nanodrop", "bioanalyzer", "thermocycler", "pcr",
    # Cell biology
    "cell", "cells", "confluenc", "passage", "trypsin", "detach",
    "harvest", "count", "viability", "dead", "live", "apoptosis",
    "lineage", "stem", "progenitor", "hematopoietic", "hspc",
    "bone marrow", "lin", "sca1", "cd117", "cd150", "cd105",
    # Assays
    "bca", "bradford", "protein", "concentration", "absorbance",
    "elisa", "western", "jess", "blot", "gel", "electrophoresis",
    "hcr", "fish", "probe", "hybridization", "amplification",
    # Calculations
    "dilution", "c1v1", "molarity", "molar", "stock", "working",
    "volume", "weight", "concentration", "master mix",
    # Sample prep
    "fractionation", "separation", "isolation", "purification",
    "magnetic", "column", "beads", "selection", "depletion",
    "enrichment",
}

# Common words to ignore
_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has",
    "will", "can", "are", "was", "were", "been", "being", "not", "but",
    "all", "each", "any", "our", "your", "its", "how", "what", "when",
    "where", "which", "who", "why", "use", "used", "using", "also",
    "into", "about", "after", "before", "between", "during", "should",
    "would", "could", "may", "might", "must", "shall", "then", "than",
    "very", "just", "only", "most", "more", "some", "such", "these",
    "those", "other", "please", "thanks", "okay", "yes", "sure",
}


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text for matching."""
    words = set(re.findall(r"[a-z][a-z0-9/()-]+", text.lower()))
    # Keep lab terms and non-trivial words
    result = set()
    for w in words:
        if w in _STOP_WORDS or len(w) < 3:
            continue
        # Check if word matches or starts with a lab term
        if w in _LAB_TERMS:
            result.add(w)
            continue
        for term in _LAB_TERMS:
            if w.startswith(term) or term.startswith(w):
                result.add(w)
                break
        else:
            # Keep non-trivial words (4+ chars)
            if len(w) >= 4:
                result.add(w)
    return result


# ── Retrieval ─────────────────────────────────────────────────────────────────


class SkillIndex:
    """Keyword index over document chunks for fast retrieval."""

    def __init__(self, chunks: list[SkillChunk] | None = None) -> None:
        self._chunks: list[SkillChunk] = chunks or []

    def add_document(self, text: str, source: str = "") -> int:
        """Split a document and add its chunks to the index."""
        new_chunks = split_into_chunks(text)
        if source:
            for c in new_chunks:
                c.title = f"[{source}] {c.title}"
        self._chunks.extend(new_chunks)
        return len(new_chunks)

    def retrieve(
        self,
        query: str,
        max_chars: int = MAX_CONTEXT_CHARS,
        min_score: float = 0.1,
    ) -> str:
        """Retrieve relevant chunks for a query, up to max_chars total.

        Returns concatenated chunk text.
        """
        if not self._chunks:
            return ""

        query_kw = _extract_keywords(query)
        if not query_kw:
            return ""

        scored: list[tuple[float, SkillChunk]] = []
        for chunk in self._chunks:
            if not chunk.keywords:
                continue
            overlap = query_kw & chunk.keywords
            if not overlap:
                continue
            # Score: Jaccard-like but weighted toward query coverage
            score = len(overlap) / len(query_kw)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        result_parts: list[str] = []
        total = 0
        for score, chunk in scored:
            if score < min_score:
                break
            if total + chunk.char_count > max_chars:
                # Try to fit a truncated version
                remaining = max_chars - total
                if remaining > 500:
                    result_parts.append(chunk.content[:remaining] + "\n[...truncated]")
                    total += remaining
                break
            result_parts.append(chunk.content)
            total += chunk.char_count

        if result_parts:
            logger.debug(
                "Skill retrieval: %d/%d chunks matched (%.1fK chars)",
                len(result_parts), len(self._chunks), total / 1000,
            )

        return "\n\n---\n\n".join(result_parts)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def total_chars(self) -> int:
        return sum(c.char_count for c in self._chunks)
