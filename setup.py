"""
One-time setup script — run once after placing service_account.json.

Creates:
  1. 'experiments' subfolder inside the root Lab Assistant Drive folder
  2. Google Sheets file 'Lab Assistant' with three configured tabs:
       - Lab Journal
       - Stock Orders
       - Received Supplies
  3. Prints the SHEETS_SPREADSHEET_ID to add to your .env

Usage:
    python setup.py
"""

import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from google.oauth2 import service_account
from googleapiclient.discovery import build

from src.config import (
    DRIVE_EXPERIMENTS_SUBFOLDER,
    DRIVE_ROOT_FOLDER_ID,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    SHEET_LAB_JOURNAL,
    SHEET_STOCK_ORDERS,
    SHEET_RECEIVED,
)

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_services():
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE, scopes=_SCOPES
    )
    drive = build("drive", "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds)
    return drive, sheets


def ensure_experiments_folder(drive) -> str:
    q = (
        f"'{DRIVE_ROOT_FOLDER_ID}' in parents"
        " and mimeType='application/vnd.google-apps.folder'"
        f" and name='{DRIVE_EXPERIMENTS_SUBFOLDER}'"
        " and trashed=false"
    )
    res = drive.files().list(q=q, fields="files(id, name)").execute()
    files = res.get("files", [])
    if files:
        folder_id = files[0]["id"]
        print(f"✅  '{DRIVE_EXPERIMENTS_SUBFOLDER}' folder already exists  (id={folder_id})")
        return folder_id

    metadata = {
        "name": DRIVE_EXPERIMENTS_SUBFOLDER,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [DRIVE_ROOT_FOLDER_ID],
    }
    folder = drive.files().create(body=metadata, fields="id").execute()
    folder_id = folder["id"]
    print(f"✅  Created '{DRIVE_EXPERIMENTS_SUBFOLDER}' folder  (id={folder_id})")
    return folder_id


def create_lab_sheets(drive, sheets) -> str:
    # Check if it already exists
    q = (
        f"'{DRIVE_ROOT_FOLDER_ID}' in parents"
        " and mimeType='application/vnd.google-apps.spreadsheet'"
        " and name='Lab Assistant'"
        " and trashed=false"
    )
    res = drive.files().list(q=q, fields="files(id)").execute()
    existing = res.get("files", [])
    if existing:
        spreadsheet_id = existing[0]["id"]
        print(f"✅  'Lab Assistant' spreadsheet already exists  (id={spreadsheet_id})")
        return spreadsheet_id

    # Create the spreadsheet
    spreadsheet = sheets.spreadsheets().create(
        body={
            "properties": {"title": "Lab Assistant"},
            "sheets": [
                {"properties": {"title": SHEET_LAB_JOURNAL,  "index": 0}},
                {"properties": {"title": SHEET_STOCK_ORDERS, "index": 1}},
                {"properties": {"title": SHEET_RECEIVED,     "index": 2}},
            ],
        }
    ).execute()
    spreadsheet_id = spreadsheet["spreadsheetId"]
    print(f"✅  Created 'Lab Assistant' spreadsheet  (id={spreadsheet_id})")

    # Move it into the root Drive folder
    file = drive.files().get(fileId=spreadsheet_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents", []))
    drive.files().update(
        fileId=spreadsheet_id,
        addParents=DRIVE_ROOT_FOLDER_ID,
        removeParents=previous_parents,
        fields="id, parents",
    ).execute()
    print(f"   Moved to Lab Assistant Drive folder.")

    # Write headers
    headers = {
        SHEET_LAB_JOURNAL: [
            "Exp Name", "Date", "Researcher", "Protocol",
            "Protocol Version", "Objective / Target", "Session Doc Link", "Status",
        ],
        SHEET_STOCK_ORDERS: [
            "Item Name", "Catalog #", "Quantity", "Unit", "Supplier",
            "Status", "Requested By", "Date Requested", "Date Ordered", "Date Arrived",
        ],
        SHEET_RECEIVED: [
            "Item", "Lot #", "Quantity", "Unit", "Expiry Date",
            "Storage Location", "Date Received", "Received By", "Linked Order Row",
        ],
    }
    data = [
        {"range": f"{sheet}!A1", "values": [cols]}
        for sheet, cols in headers.items()
    ]
    sheets.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"valueInputOption": "RAW", "data": data},
    ).execute()
    print(f"   Headers written to all 3 sheets.")
    return spreadsheet_id


def main():
    print("\n=== Lab Assistant — one-time setup ===\n")

    if not DRIVE_ROOT_FOLDER_ID:
        print("❌  DRIVE_ROOT_FOLDER_ID is not set in .env — aborting.")
        sys.exit(1)

    try:
        drive, sheets = get_services()
    except Exception as exc:
        print(f"❌  Could not authenticate with Google: {exc}")
        print("    Make sure GOOGLE_SERVICE_ACCOUNT_FILE is set and the file exists.")
        sys.exit(1)

    ensure_experiments_folder(drive)
    spreadsheet_id = create_lab_sheets(drive, sheets)

    print(f"\n📋  Add this line to your .env:\n")
    print(f"    SHEETS_SPREADSHEET_ID={spreadsheet_id}\n")
    print("Setup complete. Share the spreadsheet with your service account email if needed.")


if __name__ == "__main__":
    main()
