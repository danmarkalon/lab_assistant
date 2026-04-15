"""
Voice transcription — Google Gemini.

Gemini 2.0 Flash accepts OGG Opus audio natively (Telegram's voice format).
No separate transcription service or audio conversion needed — one API key
handles everything.

Supports any spoken language; Gemini auto-detects and transcribes faithfully.
"""

import base64
import logging

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)
logger = logging.getLogger(__name__)


async def transcribe_ogg(ogg_bytes: bytes) -> str:
    """Transcribe an OGG Opus voice message using Gemini.

    Args:
        ogg_bytes: Raw bytes of the .ogg voice file downloaded from Telegram.

    Returns:
        Transcript string, stripped of leading/trailing whitespace.
    """
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    audio_part = {
        "inline_data": {
            "mime_type": "audio/ogg",
            "data": base64.b64encode(ogg_bytes).decode("utf-8"),
        }
    }
    response = await model.generate_content_async(
        contents=[
            {
                "role": "user",
                "parts": [
                    audio_part,
                    {
                        "text": (
                            "Transcribe this audio accurately. "
                            "Output only the transcribed text, nothing else."
                        )
                    },
                ],
            }
        ]
    )
    return response.text.strip()
