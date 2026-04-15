"""
Protocol loader module.

Downloads a protocol .docx from Google Drive, extracts all text (body paragraphs
+ table cells), and loads the companion knowledge Google Doc if one exists.

Returns the text components separately so that build_system_prompt() in
claude_client.py can format them into the Protocol Expert system prompt.

IMPORTANT — why we extract tables separately:
    doc.paragraphs from python-docx only covers body text.
    Table cells are NOT included. Buffer recipes almost always live in tables,
    so iterating doc.tables is critical for the /buffer functionality to work.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

from docx import Document

from .google_client import download_docx, find_companion_doc_id, get_doc_text

logger = logging.getLogger(__name__)


def extract_docx_text(docx_bytes: bytes) -> str:
    """Extract all text from a .docx file — body paragraphs + table contents.

    Table rows are formatted as pipe-separated cells so Claude can read
    them as structured data (e.g. reagent | stock concentration | final volume).
    """
    doc = Document(io.BytesIO(docx_bytes))
    parts: list[str] = []

    # Body paragraphs (headings, steps, notes)
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)

    # Tables — critical for buffer recipes
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


async def load_protocol(
    file_id: str,
    file_name: str,
    modified_time: str,
    parent_folder_id: str = "",
) -> tuple[str, str, str, str, Optional[str]]:
    """Load a protocol from Drive and return all components for the skill.

    Args:
        file_id:       Google Drive file ID of the .docx protocol.
        file_name:     Display name (e.g. "Western_Blot_v3.docx").
        modified_time: Drive modifiedTime string (ISO 8601).

    Returns:
        (protocol_text, companion_text, protocol_name, protocol_version, companion_doc_id)

        protocol_text:    Full extracted text from the .docx.
        companion_text:   Text from the companion knowledge Google Doc (empty str if none).
        protocol_name:    Filename without extension (e.g. "Western_Blot_v3").
        protocol_version: Filename + last-modified date for traceability.
        companion_doc_id: Google Doc id of the companion doc, or None.
    """
    protocol_name = os.path.splitext(file_name)[0]
    # Use only the date part (YYYY-MM-DD) of the ISO 8601 modifiedTime
    mod_date = modified_time[:10] if modified_time else "unknown"
    protocol_version = f"{file_name} (modified {mod_date})"

    # Download and parse .docx
    docx_bytes = await download_docx(file_id)
    protocol_text = extract_docx_text(docx_bytes)
    logger.info("Loaded protocol '%s' (%d chars)", protocol_name, len(protocol_text))

    # Load companion knowledge doc if it exists
    companion_text = ""
    companion_doc_id: Optional[str] = None
    try:
        companion_doc_id = await find_companion_doc_id(protocol_name, parent_folder_id)
        if companion_doc_id:
            companion_text = await get_doc_text(companion_doc_id)
            logger.info(
                "Loaded companion doc for '%s' (%d chars)", protocol_name, len(companion_text)
            )
    except Exception:
        logger.warning(
            "Could not load companion doc for '%s'", protocol_name, exc_info=True
        )

    return protocol_text, companion_text, protocol_name, protocol_version, companion_doc_id
