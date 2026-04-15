"""
Voice transcription module — OpenAI Whisper-1.

Telegram voice messages are sent as OGG Opus (.oga / .ogg).
Whisper-1 accepts OGG natively; no audio conversion is needed.

Language is auto-detected by default (WHISPER_LANGUAGE = None in config).
Set WHISPER_LANGUAGE to a BCP-47 code (e.g. "en", "he") to force a language
and improve speed + accuracy when the lab language is known.
"""

import io

from openai import AsyncOpenAI

from .config import OPENAI_API_KEY, WHISPER_LANGUAGE

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def transcribe_ogg(ogg_bytes: bytes) -> str:
    """Transcribe an OGG Opus audio buffer and return the transcript as a string.

    Args:
        ogg_bytes: Raw bytes of the .ogg voice file downloaded from Telegram.

    Returns:
        Transcript string, stripped of leading/trailing whitespace.
    """
    buf = io.BytesIO(ogg_bytes)
    # The filename extension tells Whisper the audio format.
    buf.name = "voice.ogg"

    kwargs: dict = {
        "model": "whisper-1",
        "file": buf,
        "response_format": "text",
    }
    if WHISPER_LANGUAGE:
        kwargs["language"] = WHISPER_LANGUAGE

    result = await _client.audio.transcriptions.create(**kwargs)

    # response_format="text" returns a plain string directly
    if isinstance(result, str):
        return result.strip()
    return result.text.strip()
