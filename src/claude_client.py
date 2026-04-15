"""
Claude AI integration module.

Responsibilities:
- ConversationHistory: rolling message list for a session
- build_system_prompt(): assembles BASE + protocol text + companion knowledge
- send_message(): text-only Claude call
- send_message_with_image(): multimodal (image + text) Claude call

All functions are async (AsyncAnthropic).
Claude always responds in English regardless of input language.
"""

from __future__ import annotations

import base64
from typing import Optional

import anthropic

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

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
    """Rolling list of messages for a single bot session.

    Each message is a dict with keys "role" ("user" or "assistant") and "content".
    User content can be a plain string or a list of content blocks (for multimodal).
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def add_user(self, content: str | list) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

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
    """Assemble a system prompt, optionally embedding a protocol + companion knowledge.

    Args:
        protocol_text:    Full text extracted from the protocol .docx (body + tables).
        companion_text:   Text from the companion knowledge Google Doc, if it exists.
        protocol_name:    Display name of the protocol (e.g. "Western Blot").
        protocol_version: Version string (filename or Drive modifiedTime).

    Returns:
        A single system prompt string ready to pass to the Claude API.
    """
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


# ── Claude API calls ──────────────────────────────────────────────────────────


async def send_message(
    history: ConversationHistory,
    user_text: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
) -> str:
    """Send a text message, update history, and return Claude's reply."""
    history.add_user(user_text)
    response = await _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt or BASE_SYSTEM_PROMPT,
        messages=history.messages,
    )
    reply: str = response.content[0].text
    history.add_assistant(reply)
    return reply


async def send_message_with_image(
    history: ConversationHistory,
    image_bytes: bytes,
    user_text: str,
    media_type: str = "image/jpeg",
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
) -> str:
    """Send an image + text message, update history, and return Claude's reply.

    Telegram photos arrive as JPEG — media_type default is "image/jpeg".
    Supported: image/jpeg, image/png, image/gif, image/webp.
    """
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        },
        {"type": "text", "text": user_text},
    ]
    history.add_user(content)
    response = await _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt or BASE_SYSTEM_PROMPT,
        messages=history.messages,
    )
    reply: str = response.content[0].text
    history.add_assistant(reply)
    return reply
