"""
Google API client — Drive, Sheets, Docs.

All public functions are async. The underlying google-api-python-client library
is synchronous; calls are wrapped with run_in_executor to avoid blocking the
Telegram bot event loop.

Authentication: Google Service Account (headless — no browser OAuth).
Setup:
  1. Google Cloud Console → Enable Drive API + Sheets API + Docs API
  2. Create a Service Account → download JSON key → save as service_account.json
  3. Share the "Lab Assistant" Drive folder and the Lab Assistant Sheets file
     with the service account email address.
"""

from __future__ import annotations

import asyncio
import functools
import io
import logging
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .config import (
    DRIVE_PROTOCOLS_FOLDER_ID,
    DRIVE_SESSION_REPORTS_FOLDER_ID,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    SHEET_LAB_JOURNAL,
    SHEETS_SPREADSHEET_ID,
)

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
]

# Module-level service cache — built lazily on first call.
_services: dict = {}


def _get_service(name: str, version: str):
    """Return (and cache) a Google API service client."""
    key = f"{name}_{version}"
    if key not in _services:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE, scopes=_SCOPES
        )
        _services[key] = build(name, version, credentials=creds)
    return _services[key]


async def _run(func, *args, **kwargs):
    """Run a synchronous Google API call in the default thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# ── Drive ─────────────────────────────────────────────────────────────────────


def _list_docx_sync() -> list[dict]:
    svc = _get_service("drive", "v3")
    q = (
        f"'{DRIVE_PROTOCOLS_FOLDER_ID}' in parents"
        " and mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'"
        " and trashed=false"
    )
    result = svc.files().list(
        q=q,
        fields="files(id, name, modifiedTime)",
        orderBy="name",
    ).execute()
    return result.get("files", [])


async def list_protocols() -> list[dict]:
    """Return all .docx protocol files in the Protocols Drive folder.

    Each item: {"id": str, "name": str, "modifiedTime": str}
    """
    return await _run(_list_docx_sync)


def _download_file_sync(file_id: str) -> bytes:
    svc = _get_service("drive", "v3")
    buf = io.BytesIO()
    request = svc.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


async def download_docx(file_id: str) -> bytes:
    """Download a .docx blob file from Drive and return raw bytes."""
    return await _run(_download_file_sync, file_id)


def _find_companion_sync(protocol_stem: str) -> Optional[str]:
    """Return the Google Doc id of a companion knowledge doc, or None."""
    svc = _get_service("drive", "v3")
    # Escape single quotes in the name to avoid query injection
    safe_stem = protocol_stem.replace("'", "\\'")
    q = (
        f"'{DRIVE_PROTOCOLS_FOLDER_ID}' in parents"
        " and mimeType='application/vnd.google-apps.document'"
        f" and name contains '{safe_stem}_context'"
        " and trashed=false"
    )
    result = svc.files().list(q=q, fields="files(id, name)").execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


async def find_companion_doc_id(protocol_stem: str) -> Optional[str]:
    """Find the companion knowledge Google Doc for a protocol.

    Naming convention: {protocol_name}_context  (Google Doc, not .docx)
    Returns doc_id string, or None if not found.
    """
    return await _run(_find_companion_sync, protocol_stem)


# ── Docs ─────────────────────────────────────────────────────────────────────


def _extract_googledoc_text(doc: dict) -> str:
    """Extract plain text from a Google Docs API document response."""
    parts: list[str] = []

    def process_elements(elements: list) -> None:
        for el in elements:
            if "paragraph" in el:
                for run in el["paragraph"].get("elements", []):
                    if "textRun" in run:
                        parts.append(run["textRun"].get("content", ""))
            elif "table" in el:
                for row in el["table"].get("tableRows", []):
                    for cell in row.get("tableCells", []):
                        process_elements(cell.get("content", []))

    process_elements(doc.get("body", {}).get("content", []))
    return "".join(parts)


def _read_doc_sync(doc_id: str) -> str:
    svc = _get_service("docs", "v1")
    doc = svc.documents().get(documentId=doc_id).execute()
    return _extract_googledoc_text(doc)


async def get_doc_text(doc_id: str) -> str:
    """Read all text content from a Google Doc and return as a plain string."""
    return await _run(_read_doc_sync, doc_id)


def _append_doc_sync(doc_id: str, text: str) -> None:
    svc = _get_service("docs", "v1")
    # Fetch the doc to get the current end index.
    doc = svc.documents().get(documentId=doc_id).execute()
    # The body content always ends with one implicit character; insert just before it.
    end_index = doc["body"]["content"][-1]["endIndex"] - 1
    svc.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": end_index},
                        "text": "\n" + text,
                    }
                }
            ]
        },
    ).execute()


async def append_doc_text(doc_id: str, text: str) -> None:
    """Append text to the end of a Google Doc."""
    await _run(_append_doc_sync, doc_id, text)


def _create_doc_sync(title: str, folder_id: str) -> str:
    """Create a new Google Doc in the specified Drive folder. Returns the doc id."""
    svc = _get_service("drive", "v3")
    file_metadata = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id],
    }
    file = svc.files().create(body=file_metadata, fields="id").execute()
    return file["id"]


async def create_session_doc(title: str) -> str:
    """Create a session report Google Doc in the Session Reports folder.

    Returns the new doc's id.
    """
    return await _run(_create_doc_sync, title, DRIVE_SESSION_REPORTS_FOLDER_ID)


def get_doc_url(doc_id: str) -> str:
    """Return the web URL for a Google Doc (no API call needed)."""
    return f"https://docs.google.com/document/d/{doc_id}/edit"


# ── Sheets ────────────────────────────────────────────────────────────────────


def _read_sheet_sync(sheet_name: str, range_: str) -> list[list]:
    svc = _get_service("sheets", "v4")
    result = (
        svc.spreadsheets()
        .values()
        .get(
            spreadsheetId=SHEETS_SPREADSHEET_ID,
            range=f"{sheet_name}!{range_}",
        )
        .execute()
    )
    return result.get("values", [])


async def read_sheet(sheet_name: str, range_: str = "A:Z") -> list[list]:
    """Read all rows from a Google Sheet tab."""
    return await _run(_read_sheet_sync, sheet_name, range_)


def _append_row_sync(sheet_name: str, row: list) -> None:
    svc = _get_service("sheets", "v4")
    svc.spreadsheets().values().append(
        spreadsheetId=SHEETS_SPREADSHEET_ID,
        range=f"{sheet_name}!A:A",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": [row]},
    ).execute()


async def append_sheet_row(sheet_name: str, row: list) -> None:
    """Append a single row to a Google Sheet tab."""
    await _run(_append_row_sync, sheet_name, row)
