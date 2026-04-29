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
# Only models confirmed to have free-tier quota on this key.
# gemini-2.5-flash: primary (best quality, occasionally 503s under high load)
# gemini-2.5-flash-lite: reliable fallback with separate quota pool
_FALLBACK_MODELS: list[str] = [
    GEMINI_MODEL,
    "gemini-2.5-flash-lite",
]
# Deduplicate while preserving order
MODEL_CHAIN: list[str] = list(dict.fromkeys(_FALLBACK_MODELS))

# Per-process pacing delay (seconds). 0 at startup; bumps to 4s on first RPM
# hit so all subsequent calls are spaced to stay within 15 RPM.
_throttle_delay: float = 0.0


def _is_rpm_error(exc: genai_errors.APIError) -> bool:
    """True for per-minute quota exhaustion (recoverable by slowing down)."""
    return getattr(exc, 'code', 0) == 429 and "PerMinute" in str(exc)


def _is_daily_error(exc: genai_errors.APIError) -> bool:
    """True for daily quota exhaustion (no recovery until tomorrow)."""
    return getattr(exc, 'code', 0) == 429 and not _is_rpm_error(exc)


def _parse_retry_delay(exc: genai_errors.APIError) -> float:
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
You are supporting a researcher during active bench work via Telegram on a small phone screen.

Your responsibilities:
- Answer questions about protocols precisely, citing the relevant step when possible.
- Guide buffer preparation: read the recipe from the provided protocol and calculate \
exact volumes and weights for the researcher's target amount.
- Log protocol deviations clearly and objectively.
- Help with lab calculations: dilutions (C1V1 = C2V2), molarity, stock solution prep, \
unit conversions.
- Flag potential issues or known failure points when you are aware of them.

Formatting rules (CRITICAL — the researcher reads your replies on a small phone screen):
- Keep replies SHORT. Skip pleasantries, filler, and restatements.
- Use plain text by default. Only bold a word with **word** when it truly needs emphasis.
- NEVER use nested formatting like ***text*** or **_text_**.
- Use line breaks generously to separate sections — whitespace aids readability.
- For lists, use simple lines with a dash or bullet, one item per line.
- For tables/data, use one value per line with a clear label:
    Lin(-): 2.785×10⁶ in 1mL
    Origin: 4.4×10⁶ in 200µL
- For calculations, show: formula → result. One line each.
- NO long paragraphs. If a paragraph exceeds 3 lines, break it up.
- Use emoji sparingly for section markers (⚠️ for warnings, ✅ for done).

Observation tagging:
When the researcher mentions a factual observation about the experiment — such as cell \
confluency, cell origin/passage, reagent appearance, timing, temperature readings, pH, \
color changes, unexpected results, or any concrete detail worth recording — include an \
observation tag in your reply on its own line:
[OBS: <one-sentence factual note of what the researcher reported>]
Only tag genuine experimental observations, NOT questions, greetings, or commands. \
You may include multiple [OBS: ...] tags if several observations are mentioned. \
Continue your normal reply around the tag(s).

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
    general_methods_text: Optional[str] = None,
    is_facs: bool = False,
) -> str:
    """Assemble a system prompt, optionally embedding a protocol + companion knowledge."""
    parts = [BASE_SYSTEM_PROMPT]

    if general_methods_text:
        parts.append(
            "\n=== GENERAL LABORATORY METHODS ===\n"
            "(Cross-method knowledge: buffers, BCA, Jess Western, SOPs)\n"
            f"{general_methods_text}"
        )

    if protocol_text:
        header = f"=== PROTOCOL: {protocol_name or 'Unknown'}"
        if protocol_version:
            header += f" | version: {protocol_version}"
        header += " ==="
        parts.append(f"\n{header}\n{protocol_text}")

    if companion_text:
        parts.append(
            "\n=== METHOD-SPECIFIC KNOWLEDGE BASE ===\n"
            "(Development history, expert notes, and known issues for this method)\n"
            f"{companion_text}"
        )

    if is_facs:
        parts.append(_FACS_CALCULATOR_PROMPT)

    return "\n\n".join(parts)


# ── FACS Calculator Prompt ────────────────────────────────────────────────────

_FACS_CALCULATOR_PROMPT = """\
=== FACS CELL DATA EXTRACTION ===

When the researcher provides cell count data (from photos, text, or voice),
extract it into a [CELL_DATA] block. The system calculates everything automatically.

FORMAT (pipe-separated, one line per fraction):
[CELL_DATA]
Treatment | Fraction | Concentration (cells/mL) | Volume (mL)
PBS | Origin | 5e6 | 1
PBS | Lin(-) | 2e6 | 3
PBS | Lin(+) | 50e6 | 5
5mg/kg | Origin | 6e6 | 2
5mg/kg | Lin(-) | 5.5e6 | 5
5mg/kg | Lin(+) | 46e6 | 2
[/CELL_DATA]

RULES:
- Use scientific notation for concentration: 5e6 = 5×10⁶
- Concentration in cells/mL, volume in mL
- Fractions must be exactly: Origin, Lin(-), Lin(+)
- Include ALL treatment groups and ALL fractions
- First line after [CELL_DATA] is the header — include it exactly as shown
- Do NOT calculate antibody volumes, master mixes, or plate layouts — the system does this
- If data is unclear or incomplete, ASK the researcher to clarify
- ALWAYS output [CELL_DATA] when cell count information is available
"""


# ── Internal helper ───────────────────────────────────────────────────────────


def _make_config(system_prompt: str, max_tokens: int) -> genai_types.GenerateContentConfig:
    """Build a GenerateContentConfig with system prompt and token limit."""
    return genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
    )


# ── Gemini API calls ──────────────────────────────────────────────────────────


def _friendly_api_error(exc: genai_errors.APIError) -> str:
    code = getattr(exc, 'code', 0)
    if code == 429:
        if _is_rpm_error(exc):
            return "⚠️ Too many requests per minute — slowing down. Please try again in a moment."
        return "⚠️ Daily AI quota reached. Please try again tomorrow, or ask the admin to enable billing."
    if code == 503:
        return "⚠️ AI service temporarily unavailable. Please try again in a moment."
    return f"⚠️ AI error ({code}): {getattr(exc, 'message', str(exc))}"


async def _generate_with_fallback(
    contents: list[dict],
    config: genai_types.GenerateContentConfig,
    notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
) -> str:
    """Core Gemini call with exponential backoff and model cascade.

    Strategy:
    - Pacing: sleep _throttle_delay before each call (0 until first RPM hit).
    - 503 (model overloaded): notify user once, cascade to next model immediately.
    - RPM 429: exponential backoff (1s → 2s → 4s, capped 60s), bump _throttle_delay
               to 4s for all future calls, then cascade to next model in chain.
    - Daily 429: immediate friendly error — no retry possible.
    - Other errors: immediate friendly error.
    """
    global _throttle_delay
    last_exc: genai_errors.APIError | None = None
    notified = False
    backoff = 1.0

    # Pass 1: fast cascade (503 = skip immediately, RPM = backoff then skip).
    # Pass 2: wait 15s, then retry the full chain again (503 spikes are brief).
    # Pass 3: wait 30s, final attempt on full chain.
    for chain_pass in range(3):
        if chain_pass > 0:
            wait = 15 if chain_pass == 1 else 30
            logger.warning("Pass %d: all models failed, waiting %ds then retrying full chain", chain_pass + 1, wait)
            if not notified and notify_retry:
                notified = True
                await notify_retry()
            await asyncio.sleep(wait)

        for model in MODEL_CHAIN:
            # Up to 3 attempts per model for RPM backoff; 503 cascades immediately
            for attempt in range(3):
                if _throttle_delay > 0:
                    await asyncio.sleep(_throttle_delay)
                try:
                    logger.debug("Trying model %s (pass %d, attempt %d)", model, chain_pass + 1, attempt + 1)
                    response = await _client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                    if model != MODEL_CHAIN[0]:
                        logger.info("Succeeded on fallback model %s", model)
                    return response.text
                except (genai_errors.ClientError, genai_errors.ServerError) as exc:
                    last_exc = exc

                    err_code = getattr(exc, 'code', 0)
                    logger.warning("API error %d (%s) on %s pass %d attempt %d",
                                   err_code, type(exc).__name__, model, chain_pass + 1, attempt + 1)

                    # Daily quota on this model — cascade to next, only stop if last
                    if _is_daily_error(exc):
                        logger.warning("Daily quota hit on %s, cascading", model)
                        if model == MODEL_CHAIN[-1]:
                            return _friendly_api_error(exc)
                        break  # try next model

                    # 503 — model is overloaded, cascade to next model immediately
                    if err_code == 503:
                        logger.warning("503 on %s, cascading to next model", model)
                        if not notified and notify_retry:
                            notified = True
                            await notify_retry()
                        break  # cascade

                    # RPM quota — exponential backoff, then cascade after 3 failures
                    if _is_rpm_error(exc):
                        wait = min(backoff, 60.0)
                        backoff = min(backoff * 2, 60.0)
                        _throttle_delay = max(_throttle_delay, 4.0)
                        logger.warning("RPM hit on %s, backoff %.0fs (throttle now %.0fs)", model, wait, _throttle_delay)
                        if not notified and notify_retry:
                            notified = True
                            await notify_retry()
                        await asyncio.sleep(wait)
                        continue  # retry same model first, cascade after 3 failures

                    # Other error (auth, not-found, etc.) — fail immediately
                    logger.error("Gemini API error on %s: %s", model, exc)
                    return _friendly_api_error(exc)

                except Exception as exc:
                    # Unexpected error (SSL, connection, timeout, etc.)
                    logger.error("Unexpected %s on %s pass %d attempt %d: %s",
                                 type(exc).__name__, model, chain_pass + 1, attempt + 1, exc)
                    last_exc = exc  # type: ignore[assignment]
                    break  # cascade to next model

            # All attempts exhausted for this model — cascade
            if model != MODEL_CHAIN[-1]:
                logger.warning("Cascading from %s to next model", model)

    logger.error("All models in fallback chain failed after %d passes", 3)
    return _friendly_api_error(last_exc) if last_exc else "⚠️ AI unavailable. Please try again."


async def send_message(
    history: ConversationHistory,
    user_text: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
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
    max_tokens: int = 1024,
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


async def describe_image(
    image_bytes: bytes,
    instruction: str = "",
    media_type: str = "image/jpeg",
    notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
) -> str:
    """Standalone vision call — extracts text/data from an image.

    Uses a minimal prompt and no conversation history so the image
    is processed once and never re-sent on subsequent turns.
    Returns a text description that can be fed into the main agent.
    """
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    prompt = instruction or (
        "Extract ALL text, numbers, labels, and measurements visible in this image. "
        "Preserve the structure (tables, lists, groupings). "
        "If handwritten, transcribe carefully. Output only the extracted content."
    )
    contents = [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": media_type, "data": image_data}},
                {"text": prompt},
            ],
        }
    ]
    cfg = _make_config("You are a precise image-to-text extraction tool.", 1024)
    return await _generate_with_fallback(contents, cfg, notify_retry)
