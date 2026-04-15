"""
Configuration module.

Loads all settings from a .env file at project root.
Copy .env.example to .env and fill in your keys before running.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file: src/ -> lab_assistant/)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# ── API Keys ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# ── Google Service Account ────────────────────────────────────────────────────
GOOGLE_SERVICE_ACCOUNT_FILE: str = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    str(Path(__file__).parent.parent / "service_account.json"),
)

# ── Google Drive ──────────────────────────────────────────────────────────────
# Single root folder — all protocol subfolders and the experiments subfolder live here.
DRIVE_ROOT_FOLDER_ID: str = os.environ.get("DRIVE_ROOT_FOLDER_ID", "")
# Name of the subfolder that gets created inside the root for session reports.
DRIVE_EXPERIMENTS_SUBFOLDER = "experiments"

# ── Google Sheets ─────────────────────────────────────────────────────────────
SHEETS_SPREADSHEET_ID: str = os.environ.get("SHEETS_SPREADSHEET_ID", "")
SHEET_LAB_JOURNAL = "Lab Journal"
SHEET_STOCK_ORDERS = "Stock Orders"
SHEET_RECEIVED = "Received Supplies"

# ── Gemini ────────────────────────────────────────────────────────────────────
# gemini-2.0-flash: free tier (1500 req/day), 1M token context, vision + audio.
# Switch to gemini-1.5-pro for higher quality on complex reasoning tasks.
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# ── Team ─────────────────────────────────────────────────────────────────────
# Map Telegram user_id (int) -> researcher display name.
# Add every lab member here so their name appears in Sheets / session reports.
TEAM_MEMBERS: dict[int, str] = {
    # 123456789: "Dr. Smith",
    # 987654321: "Dr. Jones",
}


def get_researcher_name(user_id: int) -> str:
    """Return the researcher name for a Telegram user_id, or a fallback."""
    return TEAM_MEMBERS.get(user_id, f"Researcher_{user_id}")
