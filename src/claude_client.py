"""
AI client module — Google Gemini.

Drop-in replacement for the previous Anthropic implementation.
Public interface (ConversationHistory, build_system_prompt, send_message,
send_message_with_image, call_claude) is identical — no other modules change.

Model: gemini-2.0-flash  (free tier: 1500 req/day, 1M token context window)
Handles: text, images (JPEG/PNG/GIF/WEBP), system prompts, conversation history.

Gemini message format differences from Anthropic:
  - role "assistant" → "model"
  - parts are dicts: {"text": "..."} or {"inline_data": {"mime_type": ..., "data": ...}}
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Awaitable, Callable, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .config import GEMINI_API_KEY, GEMINI_MODEL

# Must patch certifi BEFORE constructing the genai Client so its internal
# httpx.AsyncClient uses the corporate CA bundle instead of the bundled certs.
_ca = __import__("os").environ.get("HTTPLIB2_CA_CERTS") or __import__("os").environ.get("SSL_CERT_FILE")
if _ca:
    try:
        import certifi
        certifi.where = lambda: _ca  # type: ignore[method-assign]
    except ImportError:
        pass

_client = genai.Client(api_key=GEMINI_API_KEY)
logger = logging.getLogger(__name__)

# Maximum number of conversation turns kept in rolling history.
MAX_HISTORY_TURNS: int = 10

# ── Model fallback chain ──────────────────────────────────────────────────────
# On RPM quota exhaustion the bot tries each model in order.
# Each has its own separate free-tier quota pool, effectively stacking them.
# Primary is taken from .env (GEMINI_MODEL). Fallbacks are tried in sequence.
_FALLBACK_MODELS: list[str] = [
    GEMINI_MODEL,
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]
# Deduplicate while preserving order (in case GEMINI_MODEL is already in list)
_seen: set[str] = set()
MODEL_CHAIN: list[str] = [m for m in _FALLBACK_MODELS if not (_seen.add(m) or m in _seen)]  # type: ignore[func-returns-value]

# Per-process pacing delay (seconds). 0 at startup; bumps to 4s on first RPM
# hit so all subsequent calls are spaced to stay within 15 RPM.
_throttle_delay: float = 0.0


def _is_rpm_error(exc: genai_errors.ClientError) -> bool:
    """True for per-minute quota exhaustion (recoverable by slowing down)."""
    return exc.code == 429 and "PerMinute" in str(exc)


def _is_daily_error(exc: genai_errors.ClientError) -> bool:
    """True for daily quota exhaustion (no recovery until tomorrow)."""
    return exc.code == 429 and not _is_rpm_error(exc)


def _parse_retry_delay(exc: genai_errors.ClientError) -> float:
    """Extract the server-suggested retry delay in seconds; default 30."""
    try:
        details = exc.args[1].get("error", {}).get("details", []) if exc.args else []
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                return float(d.get("retryDelay", "30s").rstrip("s"))
    except Exception:
        pass
    return 30.0

# ── Base system prompt ────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
You are a highly skilled lab assistant specializing in molecular biology and cell biology.
You are supporting a researcher during active bench work.

Your responsibilities:
- Answer questions about protocols precisely, citing the relevant step when possible.
- Guide buffer preparation: read the recipe from the provided protocol and calculate \
exact volumes and weights for the researcher's target amount.
- Log protocol deviations clearly and objectively.
- Help with lab calculations: dilutions (C1V1 = C2V2), molarity, stock solution prep, \
unit conversions.
- Flag potential issues or known failure points when you are aware of them.

Rules:
- Always respond in English, regardless of the language used in the input.
- Be concise and precise. Avoid unnecessary filler.
- If you are uncertain about something, say so clearly rather than guessing.
- Never invent protocol steps or reagent concentrations — only cite what is in the \
provided protocol text."""


# ── Conversation history ──────────────────────────────────────────────────────


class ConversationHistory:
    """Rolling list of messages for a single bot session (Gemini format).

    Gemini uses role="model" (not "assistant") and parts as a list of dicts:
      {"role": "user",  "parts": [{"text": "..."}, ...]}
      {"role": "model", "parts": [{"text": "..."}]}

    History is automatically trimmed to MAX_HISTORY_TURNS after each complete
    turn (user + model) to keep per-request token costs bounded.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def add_user(self, content: str | list) -> None:
        if isinstance(content, str):
            self.messages.append({"role": "user", "parts": [{"text": content}]})
        else:
            # Multimodal — content is already a list of part dicts
            self.messages.append({"role": "user", "parts": content})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "model", "parts": [{"text": text}]})
        self._trim()

    def _trim(self) -> None:
        """Drop oldest turns so history never exceeds MAX_HISTORY_TURNS pairs."""
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def clear(self) -> None:
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)


# ── System prompt builder ─────────────────────────────────────────────────────


def build_system_prompt(
    protocol_text: Optional[str] = None,
    companion_text: Optional[str] = None,
    protocol_name: Optional[str] = None,
    protocol_version: Optional[str] = None,
) -> str:
    """Assemble a system prompt, optionally embedding a protocol + companion knowledge."""
    parts = [BASE_SYSTEM_PROMPT]

    if protocol_text:
        header = f"=== PROTOCOL: {protocol_name or 'Unknown'}"
        if protocol_version:
            header += f" | version: {protocol_version}"
        header += " ==="
        parts.append(f"\n{header}\n{protocol_text}")

    if companion_text:
        parts.append(
            "\n=== PROTOCOL KNOWLEDGE BASE ===\n"
            "(Accumulated notes, rationale, and known issues from previous runs)\n"
            f"{companion_text}"
        )

    return "\n\n".join(parts)


# ── Internal helper ───────────────────────────────────────────────────────────


def _make_config(system_prompt: str, max_tokens: int) -> genai_types.GenerateContentConfig:
    """Build a GenerateContentConfig with system prompt and token limit."""
    return genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
    )


# ── Gemini API calls ──────────────────────────────────────────────────────────


def _friendly_api_error(exc: genai_errors.ClientError) -> str:
    if exc.code == 429:
        if _is_rpm_error(exc):
            return "⚠️ Too many requests per minute — slowing down. Please try again in a moment."
        return "⚠️ Daily AI quota reached. Please try again tomorrow, or ask the admin to enable billing."
    if exc.code == 503:
        return "⚠️ AI service temporarily unavailable. Please try again in a moment."
    return f"⚠️ AI error ({exc.code}): {exc.message}"


async def _generate_with_fallback(
    contents: list[dict],
    config: genai_types.GenerateContentConfig,
    notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
) -> str:
    """Core Gemini call with exponential backoff and model cascade.

    Strategy:
    - Pacing: sleep _throttle_delay before each call (0 until first RPM hit).
    - 503 (traffic spike): notify user, wait 10s, retry same model once.
    - RPM 429: exponential backoff (1s → 2s → 4s, capped 60s), bump _throttle_delay
               to 4s for all future calls, then cascade to next model in chain.
    - Daily 429: immediate friendly error — no retry possible.
    - Other errors: immediate friendly error.
    """
    global _throttle_delay
    last_exc: genai_errors.ClientError | None = None
    backoff = 1.0

    for model in MODEL_CHAIN:
        # Up to 3 attempts per model (handles transient 503s and a couple of RPM spikes)
        for attempt in range(3):
            if _throttle_delay > 0:
                await asyncio.sleep(_throttle_delay)
            try:
                response = await _client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
                if model != MODEL_CHAIN[0]:
                    logger.info("Succeeded on fallback model %s", model)
                return response.text
            except genai_errors.ClientError as exc:
                last_exc = exc

                # Daily quota — no point retrying any model
                if _is_daily_error(exc):
                    logger.warning("Daily quota hit on %s", model)
                    return _friendly_api_error(exc)

                # 503 traffic spike — retry same model after 10s
                if exc.code == 503:
                    if attempt < 2:
                        logger.warning("503 on %s attempt %d, retrying in 10s", model, attempt + 1)
                        if attempt == 0 and notify_retry:
                            await notify_retry()
                        await asyncio.sleep(10)
                        continue
                    logger.warning("503 persists on %s after 3 attempts, cascading", model)
                    break  # try next model

                # RPM quota — exponential backoff then cascade to next model
                if _is_rpm_error(exc):
                    wait = min(backoff, 60.0)
                    backoff = min(backoff * 2, 60.0)
                    _throttle_delay = max(_throttle_delay, 4.0)
                    logger.warning("RPM hit on %s, backoff %.0fs (throttle now %.0fs)", model, wait, _throttle_delay)
                    if attempt == 0 and notify_retry:
                        await notify_retry()
                    await asyncio.sleep(wait)
                    continue  # retry same model first, cascade after 3 failures

                # Other error (auth, not-found, etc.) — fail immediately
                logger.error("Gemini API error on %s: %s", model, exc)
                return _friendly_api_error(exc)

        # All attempts exhausted for this model — cascade
        if model != MODEL_CHAIN[-1]:
            logger.warning("Cascading from %s to next model", model)

    logger.error("All models in fallback chain failed")
    return _friendly_api_error(last_exc) if last_exc else "⚠️ AI unavailable. Please try again."


async def send_message(
    history: ConversationHistory,
    user_text: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
) -> str:
    """Send a text message, update history, and return Gemini's reply."""
    history.add_user(user_text)
    cfg = _make_config(system_prompt or BASE_SYSTEM_PROMPT, max_tokens)
    reply = await _generate_with_fallback(history.messages, cfg, notify_retry)
    if not reply.startswith("⚠️"):
        history.add_assistant(reply)
    return reply


async def send_message_with_image(
    history: ConversationHistory,
    image_bytes: bytes,
    user_text: str,
    media_type: str = "image/jpeg",
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
) -> str:
    """Send an image + text message, update history, and return Gemini's reply."""
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    parts = [
        {"inline_data": {"mime_type": media_type, "data": image_data}},
        {"text": user_text},
    ]
    history.add_user(parts)
    cfg = _make_config(system_prompt or BASE_SYSTEM_PROMPT, max_tokens)
    reply = await _generate_with_fallback(history.messages, cfg, notify_retry)
    if not reply.startswith("⚠️"):
        history.add_assistant(reply)
    return reply


async def call_claude(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Standalone Gemini call (no history update) — used for summaries and notes.

    Accepts both Gemini ("parts") and legacy Anthropic ("content") message formats.
    """
    gemini_messages = []
    for msg in messages:
        if "parts" in msg:
            gemini_messages.append(msg)
        else:
            role = "model" if msg.get("role") == "assistant" else "user"
            gemini_messages.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    cfg = _make_config(system_prompt, max_tokens)
    return await _generate_with_fallback(gemini_messages, cfg)
