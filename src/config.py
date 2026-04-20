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

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.environ.get(
    "CHROMA_DB_PATH",
    str(Path(__file__).parent.parent / "chroma_db"),
)

# ── Gemini ────────────────────────────────────────────────────────────────────
# gemini-2.0-flash: free tier (1500 req/day), 1M token context, vision + audio.
# Switch to gemini-1.5-pro for higher quality on complex reasoning tasks.
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# ── Allowlist ─────────────────────────────────────────────────────────────────
# Telegram user IDs allowed to use the bot.
# Set ALLOWED_USER_IDS in .env as a comma-separated list, e.g.:
#   ALLOWED_USER_IDS=123456789,987654321
# Leave empty to allow everyone (not recommended for production).
_raw_ids = os.environ.get("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: set[int] = (
    {int(x.strip()) for x in _raw_ids.split(",") if x.strip()}
    if _raw_ids.strip()
    else set()
)


def is_allowed(user_id: int) -> bool:
    """Return True if this user is allowed. If allowlist is empty, allow all."""
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def get_researcher_name(user_id: int) -> str:
    """Fallback name — prefer user_settings.get_researcher_name() which checks saved name first."""
    return f"Researcher_{user_id}"
