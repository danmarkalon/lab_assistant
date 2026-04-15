"""
Voice transcription — Google Gemini.

Gemini 2.0 Flash accepts OGG Opus audio natively (Telegram's voice format).
No separate transcription service or audio conversion needed — one API key
handles everything.

Supports any spoken language; Gemini auto-detects and transcribes faithfully.
"""

import base64
import logging

from google import genai
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


async def transcribe_ogg(ogg_bytes: bytes) -> str:
    """Transcribe an OGG Opus voice message using Gemini.

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
    response = await _client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
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
        ],
    )
    return response.text.strip()
