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
    import os
    spreadsheet_id = os.getenv("SHEETS_SPREADSHEET_ID", "").strip()

    if not spreadsheet_id:
        print("\n⚠️   SHEETS_SPREADSHEET_ID is not set in .env.")
        print("    Please do the following manually:")
        print("    1. Go to sheets.google.com → create a blank spreadsheet")
        print("    2. Name it 'Lab Assistant'")
        print("    3. Create 3 tabs: 'Lab Journal', 'Stock Orders', 'Received Supplies'")
        print("    4. Share it with: lab-assistant@oneylab-assistnant.iam.gserviceaccount.com (Editor)")
        print("    5. Copy the ID from the URL: docs.google.com/spreadsheets/d/<ID>/edit")
        print("    6. Add to .env:  SHEETS_SPREADSHEET_ID=<ID>")
        print("    7. Re-run:  HTTPLIB2_CA_CERTS=/etc/ssl/certs/ca-certificates.crt python3 setup.py\n")
        sys.exit(0)

    print(f"✅  Using spreadsheet  (id={spreadsheet_id})")

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
