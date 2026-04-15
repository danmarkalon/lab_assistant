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
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# ── Google Service Account ────────────────────────────────────────────────────
GOOGLE_SERVICE_ACCOUNT_FILE: str = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    str(Path(__file__).parent.parent / "service_account.json"),
)

# ── Google Drive ──────────────────────────────────────────────────────────────
DRIVE_PROTOCOLS_FOLDER_ID: str = os.environ.get("DRIVE_PROTOCOLS_FOLDER_ID", "")
DRIVE_SESSION_REPORTS_FOLDER_ID: str = os.environ.get(
    "DRIVE_SESSION_REPORTS_FOLDER_ID", ""
)

# ── Google Sheets ─────────────────────────────────────────────────────────────
SHEETS_SPREADSHEET_ID: str = os.environ.get("SHEETS_SPREADSHEET_ID", "")
SHEET_LAB_JOURNAL = "Lab Journal"
SHEET_STOCK_ORDERS = "Stock Orders"
SHEET_RECEIVED = "Received Supplies"

# ── Claude ────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# ── Whisper ───────────────────────────────────────────────────────────────────
# Set to a BCP-47 language code (e.g. "en", "he") to force a language and
# improve accuracy.  None = auto-detect (handles multilingual labs).
WHISPER_LANGUAGE: str | None = None

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
