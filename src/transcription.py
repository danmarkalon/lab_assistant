"""
Voice transcription — Google Gemini.

Gemini 2.0 Flash accepts OGG Opus audio natively (Telegram's voice format).
No separate transcription service or audio conversion needed — one API key
handles everything.

Supports any spoken language; Gemini auto-detects and transcribes faithfully.
"""

import asyncio
import base64
import logging

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .config import GEMINI_API_KEY, GEMINI_MODEL

_ca = __import__("os").environ.get("HTTPLIB2_CA_CERTS") or __import__("os").environ.get("SSL_CERT_FILE")
if _ca:
    try:
        import certifi
        certifi.where = lambda: _ca  # type: ignore[method-assign]
    except ImportError:
        pass

_client = genai.Client(api_key=GEMINI_API_KEY)
logger = logging.getLogger(__name__)

_FALLBACK_MODELS = [GEMINI_MODEL, "gemini-2.5-flash-lite"]


async def transcribe_ogg(ogg_bytes: bytes) -> str:
    """Transcribe an OGG Opus voice message using Gemini.

    Retries on 503 errors and falls back to lite model.

    Args:
        ogg_bytes: Raw bytes of the .ogg voice file downloaded from Telegram.

    Returns:
        Transcript string, stripped of leading/trailing whitespace.
    """
    audio_part = genai_types.Part(
        inline_data=genai_types.Blob(
            mime_type="audio/ogg",
            data=base64.b64encode(ogg_bytes).decode("utf-8"),
        )
    )
    contents = [
        genai_types.Content(
            role="user",
            parts=[
                audio_part,
                genai_types.Part(text=(
                    "Transcribe this audio accurately. "
                    "Output only the transcribed text, nothing else."
                )),
            ],
        )
    ]

    last_exc = None
    for model in _FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = await _client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                )
                return response.text.strip()
            except genai_errors.ServerError as exc:
                last_exc = exc
                logger.warning("Transcription 503 on %s attempt %d", model, attempt + 1)
                if attempt == 0:
                    await asyncio.sleep(3)
            except Exception as exc:
                last_exc = exc
                logger.warning("Transcription error on %s: %s", model, exc)
                break  # non-retryable, try next model

    raise last_exc or RuntimeError("All transcription models failed")
