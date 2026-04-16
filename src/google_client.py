"""
Google API client — Drive, Sheets, Docs.

Drive folder layout expected:
  Lab Assistant/                  ← DRIVE_ROOT_FOLDER_ID
  ├── Cell Fractionation/         ← one subfolder per protocol
  │   ├── protocol.docx           ← the protocol file (first .docx found)
  │   ├── protocol_context        ← optional companion knowledge Google Doc
  │   └── experiments/            ← auto-created; one session doc per run
  │       └── YYYY-MM-DD — Protocol — Researcher
  ├── Another Protocol/
  │   └── ...
  └── Lab Assistant (Sheets)      ← general experiment tracking

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


def _list_protocol_folders_sync() -> list[dict]:
    """List protocol folders (direct subfolders of root, excluding Sheets files).

    For each subfolder, finds the first .docx OR Google Doc inside it.
    Returns only folders that contain at least one protocol file.

    Each item: {
        "folder_id": str,    # the protocol subfolder ID
        "name": str,         # folder name shown in the menu
        "id": str,           # file ID (for download/read)
        "docx_name": str,    # filename (or Google Doc title)
        "modifiedTime": str, # file modifiedTime
        "is_gdoc": bool,     # True if native Google Doc (read via Docs API)
    }
    """
    svc = _get_service("drive", "v3")

    # List direct subfolders of root
    q = (
        f"'{DRIVE_ROOT_FOLDER_ID}' in parents"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    result = svc.files().list(q=q, fields="files(id, name)", orderBy="name").execute()
    folders = result.get("files", [])

    protocols = []
    for folder in folders:
        # Try .docx first, then fall back to Google Doc
        docx_q = (
            f"'{folder['id']}' in parents"
            " and mimeType='application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document'"
            " and trashed=false"
        )
        docx_res = svc.files().list(
            q=docx_q,
            fields="files(id, name, modifiedTime)",
            orderBy="name",
            pageSize=1,
        ).execute()
        docx_files = docx_res.get("files", [])

        if docx_files:
            f = docx_files[0]
            protocols.append({
                "folder_id": folder["id"],
                "name": folder["name"],
                "id": f["id"],
                "docx_name": f["name"],
                "modifiedTime": f.get("modifiedTime", ""),
                "is_gdoc": False,
            })
        else:
            # Fall back to native Google Doc
            gdoc_q = (
                f"'{folder['id']}' in parents"
                " and mimeType='application/vnd.google-apps.document'"
                " and trashed=false"
            )
            gdoc_res = svc.files().list(
                q=gdoc_q,
                fields="files(id, name, modifiedTime)",
                orderBy="name",
                pageSize=1,
            ).execute()
            gdoc_files = gdoc_res.get("files", [])
            if gdoc_files:
                f = gdoc_files[0]
                protocols.append({
                    "folder_id": folder["id"],
                    "name": folder["name"],
                    "id": f["id"],
                    "docx_name": f["name"],
                    "modifiedTime": f.get("modifiedTime", ""),
                    "is_gdoc": True,
                })

    return protocols


async def list_protocols() -> list[dict]:
    """Return all protocol folders that contain a .docx file.

    Each item: {"folder_id", "name", "id" (docx), "docx_name", "modifiedTime"}
    """
    return await _run(_list_protocol_folders_sync)


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


def _find_companion_sync(folder_name: str, parent_folder_id: str) -> Optional[str]:
    """Search for a companion knowledge Google Doc in the protocol folder.

    Tries two naming conventions:
      1. {folder_name}_context   (preferred — matches folder name)
      2. Any Google Doc whose name contains '_context'  (fallback)
    """
    svc = _get_service("drive", "v3")

    # Try exact folder-name convention first
    safe = folder_name.replace("'", "\\'")
    q = (
        f"'{parent_folder_id}' in parents"
        " and mimeType='application/vnd.google-apps.document'"
        f" and name='{safe}_context'"
        " and trashed=false"
    )
    result = svc.files().list(q=q, fields="files(id, name)").execute()
    files = result.get("files", [])
    if files:
        return files[0]["id"]

    # Fallback: any doc containing '_context' in this folder
    q2 = (
        f"'{parent_folder_id}' in parents"
        " and mimeType='application/vnd.google-apps.document'"
        " and name contains '_context'"
        " and trashed=false"
    )
    result2 = svc.files().list(q=q2, fields="files(id, name)").execute()
    files2 = result2.get("files", [])
    return files2[0]["id"] if files2 else None


async def find_companion_doc_id(
    folder_name: str, parent_folder_id: str
) -> Optional[str]:
    """Find the companion knowledge Google Doc for a protocol.

    Naming convention: {folder_name}_context  (Google Doc in the protocol folder)
    Falls back to any doc containing '_context' in the folder.
    Returns doc_id or None.
    """
    return await _run(_find_companion_sync, folder_name, parent_folder_id)


def _find_experiments_doc_sync(folder_name: str, parent_folder_id: str) -> Optional[str]:
    """Search for an experiments log Google Doc in the protocol folder.

    Naming conventions (tried in order):
      1. {folder_name}_experiments
      2. Any Google Doc whose name contains '_experiments'
    """
    svc = _get_service("drive", "v3")
    safe = folder_name.replace("'", "\\'")
    q = (
        f"'{parent_folder_id}' in parents"
        " and mimeType='application/vnd.google-apps.document'"
        f" and name='{safe}_experiments'"
        " and trashed=false"
    )
    result = svc.files().list(q=q, fields="files(id, name)").execute()
    files = result.get("files", [])
    if files:
        return files[0]["id"]

    q2 = (
        f"'{parent_folder_id}' in parents"
        " and mimeType='application/vnd.google-apps.document'"
        " and name contains '_experiments'"
        " and trashed=false"
    )
    result2 = svc.files().list(q=q2, fields="files(id, name)").execute()
    files2 = result2.get("files", [])
    return files2[0]["id"] if files2 else None


async def find_experiments_doc_id(
    folder_name: str, parent_folder_id: str
) -> Optional[str]:
    """Find the experiments log Google Doc for a protocol.

    Naming convention: {folder_name}_experiments (Google Doc in the protocol folder)
    Returns doc_id or None.
    """
    return await _run(_find_experiments_doc_sync, folder_name, parent_folder_id)


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


async def get_or_create_experiments_folder(protocol_folder_id: str) -> str:
    """Return the id of the 'experiments' subfolder inside a protocol folder."""
    return await _run(
        _get_or_create_subfolder_sync, protocol_folder_id, DRIVE_EXPERIMENTS_SUBFOLDER
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


async def create_session_doc(title: str, protocol_folder_id: str) -> str:
    """Create a session report Google Doc inside the protocol's 'experiments' subfolder."""
    experiments_folder_id = await get_or_create_experiments_folder(protocol_folder_id)
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
