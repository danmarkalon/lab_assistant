"""
Per-user settings store — persisted to user_settings.json at project root.

Settings are keyed by Telegram user_id (as string).
The file is created automatically on first write.

Available settings per user:
  name        : Display name used in session reports and Sheets (str)
  gemini_model: Override the default Gemini model for this user (str | None)

Usage:
    from .user_settings import get_setting, set_setting, get_all_settings

    name = get_setting(user_id, "name", default="Dan")
    set_setting(user_id, "name", "Dr. Cohen")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SETTINGS_FILE = Path(__file__).parent.parent / "user_settings.json"

# Available Gemini models (free tier first)
AVAILABLE_MODELS = [
    ("gemini-2.0-flash", "Gemini 2.0 Flash — fast, free tier"),
    ("gemini-1.5-flash", "Gemini 1.5 Flash — fast, free tier"),
    ("gemini-1.5-pro",   "Gemini 1.5 Pro — higher quality, limited free quota"),
]

_DEFAULT_SETTINGS: dict[str, Any] = {
    "name": None,          # None → falls back to Telegram first name
    "gemini_model": None,  # None → uses GEMINI_MODEL from .env
}


def _load() -> dict:
    if _SETTINGS_FILE.exists():
        try:
            return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not read user_settings.json — starting fresh")
    return {}


def _save(data: dict) -> None:
    try:
        _SETTINGS_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.error("Could not save user_settings.json: %s", exc)


def get_setting(user_id: int, key: str, default: Any = None) -> Any:
    data = _load()
    user_data = data.get(str(user_id), {})
    return user_data.get(key, _DEFAULT_SETTINGS.get(key, default))


def set_setting(user_id: int, key: str, value: Any) -> None:
    data = _load()
    uid = str(user_id)
    if uid not in data:
        data[uid] = {}
    data[uid][key] = value
    _save(data)


def get_all_settings(user_id: int) -> dict:
    data = _load()
    defaults = dict(_DEFAULT_SETTINGS)
    defaults.update(data.get(str(user_id), {}))
    return defaults


def get_researcher_name(user_id: int, telegram_first_name: Optional[str] = None) -> str:
    """Return display name: saved setting → Telegram first name → fallback ID."""
    saved = get_setting(user_id, "name")
    if saved:
        return saved
    if telegram_first_name:
        return telegram_first_name
    return f"Researcher_{user_id}"


def get_user_model(user_id: int) -> Optional[str]:
    """Return per-user model override, or None to use the global default."""
    return get_setting(user_id, "gemini_model")
