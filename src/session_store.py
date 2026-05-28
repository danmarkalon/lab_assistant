"""Persistent session state for resume across bot restarts.

Stores minimal session metadata to JSON so multi-day experiments can be
picked up after a process restart without losing the experiment sheet tab.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def _user_path(user_id: int) -> Path:
    return _STORE_DIR / f"{user_id}.json"


def save_session(user_id: int, state: dict) -> None:
    """Persist session state for a user."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = _user_path(user_id)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("Saved session state for user %d", user_id)


def load_session(user_id: int) -> Optional[dict]:
    """Load persisted session state, or None if no saved session."""
    path = _user_path(user_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load session for user %d: %s", user_id, exc)
        return None


def delete_session(user_id: int) -> None:
    """Remove persisted session state (called on end_session)."""
    path = _user_path(user_id)
    try:
        path.unlink(missing_ok=True)
        logger.debug("Deleted session state for user %d", user_id)
    except OSError:
        pass


def list_sessions(user_id: int) -> list[dict]:
    """List all saved sessions for a user (currently just one, but future-proof)."""
    state = load_session(user_id)
    if state:
        return [state]
    return []
