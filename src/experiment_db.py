"""
ChromaDB experiment database client.

Provides async access to the pre-built vector database of OpenProject
work packages (experiments). Supports:
  - Exact lookup by experiment number
  - Semantic search by free-text query
  - Structured field extraction from experiment documents
  - Natural language trigger detection for "open project" commands
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from typing import Optional

from .config import CHROMA_DB_PATH

logger = logging.getLogger(__name__)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(name="experiments")
        logger.info("ChromaDB connected: %d documents in 'experiments'", _collection.count())
    return _collection


async def _run(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# ── Lookup functions ──────────────────────────────────────────────────────────


def _get_by_number_sync(number: str) -> Optional[dict]:
    """Fetch experiment by number — tries wp_{number} ID first, then metadata filters."""
    col = _get_collection()
    # Try direct document ID
    result = col.get(ids=[f"wp_{number}"], include=["documents", "metadatas"])
    if result["documents"]:
        return {"document": result["documents"][0], "metadata": result["metadatas"][0]}
    # Fallback: metadata filter on experiment_id
    result = col.get(
        where={"experiment_id": str(number)},
        include=["documents", "metadatas"],
    )
    if result["documents"]:
        return {"document": result["documents"][0], "metadata": result["metadatas"][0]}
    # Fallback: metadata filter on id (int)
    try:
        result = col.get(
            where={"id": int(number)},
            include=["documents", "metadatas"],
        )
        if result["documents"]:
            return {"document": result["documents"][0], "metadata": result["metadatas"][0]}
    except (ValueError, TypeError):
        pass
    return None


async def get_experiment_by_number(number: str) -> Optional[dict]:
    return await _run(_get_by_number_sync, number)


def _search_sync(query: str, n_results: int = 5) -> list[dict]:
    col = _get_collection()
    results = col.query(query_texts=[query], n_results=n_results)
    if not results["documents"] or not results["documents"][0]:
        return []
    out = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        out.append({"document": doc, "metadata": meta, "distance": dist})
    return out


async def search_experiments(query: str, n_results: int = 5) -> list[dict]:
    return await _run(_search_sync, query, n_results)


# ── Trigger detection ─────────────────────────────────────────────────────────

_OPEN_PROJECT_RE = re.compile(
    r"open\s+project\s+(?:experiment\s+)?(\d+)",
    re.IGNORECASE,
)

_OPEN_PROJECT_TEXT_RE = re.compile(
    r"open\s+project\s+(.+)",
    re.IGNORECASE,
)


def parse_open_project(text: str) -> Optional[dict]:
    """Detect 'open project' trigger in text.

    Returns:
      {"type": "number", "value": "547"}    — for "open project experiment 547"
      {"type": "search", "value": "DNA..."} — for "open project DNA sequencing"
      None                                  — if no trigger detected
    """
    m = _OPEN_PROJECT_RE.search(text)
    if m:
        return {"type": "number", "value": m.group(1)}
    m = _OPEN_PROJECT_TEXT_RE.search(text)
    if m:
        query = m.group(1).strip()
        if query:
            return {"type": "search", "value": query}
    return None


# ── Field extraction ──────────────────────────────────────────────────────────

_FIELD_PATTERNS = {
    "objective": re.compile(
        r"(?:objective|goal|aim)\s*[:\-–—]\s*(.+?)(?:\n|$)", re.IGNORECASE
    ),
    "controls_and_conc": re.compile(
        r"(?:controls?\s*(?:&|and)\s*conc\.?|controls?\s*(?:&|and)\s*concentrations?)"
        r"\s*[:\-–—]\s*(.+?)(?:\n\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    "total_samples": re.compile(
        r"(?:total\s+(?:number\s+of\s+)?samples?)\s*[:\-–—]\s*(.+?)(?:\n|$)",
        re.IGNORECASE,
    ),
}


def extract_field(document: str, field_name: str) -> str:
    """Extract a named field from the experiment document text."""
    key = field_name.lower().replace(" ", "_").replace("&", "and")
    for pattern_key, pattern in _FIELD_PATTERNS.items():
        if pattern_key in key or key in pattern_key:
            m = pattern.search(document)
            if m:
                return m.group(1).strip()
    return ""


def extract_key_fields(document: str, metadata: dict) -> dict:
    """Extract all standard fields for sheet population."""
    return {
        "experiment_id": str(metadata.get("experiment_id", metadata.get("id", ""))),
        "subject": metadata.get("subject", ""),
        "status": metadata.get("status", ""),
        "type": metadata.get("type", ""),
        "assignee": metadata.get("assignee", ""),
        "project": metadata.get("project", ""),
        "objective": extract_field(document, "objective"),
        "controls_and_conc": extract_field(document, "controls_and_conc"),
        "total_samples": extract_field(document, "total_samples"),
    }


# ── System prompt for project mini-session ────────────────────────────────────

PROJECT_SYSTEM_PROMPT = """\
You are a lab assistant with deep expertise in experiment data retrieval.
You have been loaded with the full data record for an experiment from the project database.

Your responsibilities:
- Answer questions about this experiment accurately, citing the data provided.
- Extract specific data when asked: plate layouts, primer names/sequences, reagent lists, \
sample descriptions, concentrations, conditions, controls, or any other structured data.
- When the researcher asks to "add to the sheet" or "put in the sheet", extract the \
requested data and format it as a structured list that can be appended to the experiment sheet.
- Prefix sheet-destined data with [SHEET_DATA] on its own line, followed by the data in a \
clear tabular format (one item per line, pipe-separated columns).

Experiment data is provided below. Only use information from this data — do not invent details.

Rules:
- Always respond in English.
- Be concise and precise.
- If the requested data is not in the experiment record, say so clearly.

=== EXPERIMENT DATA ===
Subject: {subject}
ID: {experiment_id}
Status: {status}
Type: {type}
Assignee: {assignee}
Project: {project}

=== FULL DESCRIPTION ===
{document}
"""
