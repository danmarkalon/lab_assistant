"""
Google API client — Drive, Sheets, Docs.

Drive folder layout expected:
  Lab Assistant/                  ← DRIVE_ROOT_FOLDER_ID
  ├── Cell Fractionation/         ← one subfolder per protocol
  │   ├── protocol.docx           ← the protocol file
  │   └── protocol_context        ← optional companion knowledge Google Doc
  ├── Another Protocol/
  │   └── ...
  └── experiments/                ← created by setup.py; session reports go here

Authentication: Google Service Account (headless — no browser OAuth).
All public functions are async (sync Google API wrapped in run_in_executor).
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
    DRIVE_EXPERIMENTS_SUBFOLDER,
    DRIVE_ROOT_FOLDER_ID,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    SHEET_LAB_JOURNAL,
    SHEET_STOCK_ORDERS,
    SHEET_RECEIVED,
    SHEETS_SPREADSHEET_ID,
)

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
]

_services: dict = {}


def _get_service(name: str, version: str):
    key = f"{name}_{version}"
    if key not in _services:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE, scopes=_SCOPES
        )
        _services[key] = build(name, version, credentials=creds)
    return _services[key]


async def _run(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# ── Drive — protocol discovery ────────────────────────────────────────────────


def _list_docx_sync() -> list[dict]:
    """Find all .docx files accessible to the service account.

    Returns each file with its parent folder id so the companion doc
    can be searched in the same subfolder later.
    """
    svc = _get_service("drive", "v3")
    q = (
        "mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'"
        " and trashed=false"
    )
    result = svc.files().list(
        q=q,
        fields="files(id, name, modifiedTime, parents)",
        orderBy="name",
    ).execute()
    return result.get("files", [])


async def list_protocols() -> list[dict]:
    """Return all .docx protocol files the service account can see.

    Each item: {"id": str, "name": str, "modifiedTime": str, "parents": list[str]}
    The first element of "parents" is the subfolder the file lives in.
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


def _find_companion_sync(protocol_stem: str, parent_folder_id: str) -> Optional[str]:
    """Search for a companion knowledge Google Doc in the same folder as the protocol."""
    svc = _get_service("drive", "v3")
    safe_stem = protocol_stem.replace("'", "\\'")
    q = (
        f"'{parent_folder_id}' in parents"
        " and mimeType='application/vnd.google-apps.document'"
        f" and name contains '{safe_stem}_context'"
        " and trashed=false"
    )
    result = svc.files().list(q=q, fields="files(id, name)").execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


async def find_companion_doc_id(
    protocol_stem: str, parent_folder_id: str
) -> Optional[str]:
    """Find the companion knowledge Google Doc for a protocol.

    Searches in the same folder as the protocol .docx.
    Naming convention: {protocol_stem}_context  (Google Doc)
    Returns doc_id or None.
    """
    return await _run(_find_companion_sync, protocol_stem, parent_folder_id)


# ── Drive — folder management ─────────────────────────────────────────────────


def _get_or_create_subfolder_sync(parent_id: str, name: str) -> str:
    """Find a subfolder by name inside parent_id, creating it if absent. Returns folder id."""
    svc = _get_service("drive", "v3")
    safe_name = name.replace("'", "\\'")
    q = (
        f"'{parent_id}' in parents"
        " and mimeType='application/vnd.google-apps.folder'"
        f" and name='{safe_name}'"
        " and trashed=false"
    )
    result = svc.files().list(q=q, fields="files(id)").execute()
    files = result.get("files", [])
    if files:
        return files[0]["id"]
    # Create the folder
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = svc.files().create(body=metadata, fields="id").execute()
    logger.info("Created Drive subfolder '%s' (id=%s)", name, folder["id"])
    return folder["id"]


async def get_or_create_experiments_folder() -> str:
    """Return the id of the 'experiments' subfolder, creating it if needed."""
    return await _run(
        _get_or_create_subfolder_sync, DRIVE_ROOT_FOLDER_ID, DRIVE_EXPERIMENTS_SUBFOLDER
    )


# ── Docs ──────────────────────────────────────────────────────────────────────


def _extract_googledoc_text(doc: dict) -> str:
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
    return await _run(_read_doc_sync, doc_id)


def _append_doc_sync(doc_id: str, text: str) -> None:
    svc = _get_service("docs", "v1")
    doc = svc.documents().get(documentId=doc_id).execute()
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
    await _run(_append_doc_sync, doc_id, text)


def _create_doc_sync(title: str, folder_id: str) -> str:
    svc = _get_service("drive", "v3")
    file_metadata = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id],
    }
    file = svc.files().create(body=file_metadata, fields="id").execute()
    return file["id"]


async def create_session_doc(title: str) -> str:
    """Create a session report Google Doc inside the 'experiments' subfolder."""
    experiments_folder_id = await get_or_create_experiments_folder()
    return await _run(_create_doc_sync, title, experiments_folder_id)


def get_doc_url(doc_id: str) -> str:
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
    await _run(_append_row_sync, sheet_name, row)
