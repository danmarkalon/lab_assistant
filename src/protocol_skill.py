"""
Protocol Expert Skill — ProtocolSession.

A ProtocolSession represents one active experiment session. It bundles:
  - The Protocol Expert system prompt (base rules + protocol text + companion knowledge)
  - A rolling ConversationHistory for the lifetime of the session
  - All session-specific handlers (/buffer, /deviation, /refine, end_session)

Usage:
    session = await ProtocolSession.create(protocol_dict, researcher_name, objective)
    reply   = await session.handle_message("What temperature for the lysis step?")
    reply   = await session.handle_deviation("Used 0.5% Triton X-100 instead of 1%")
    reply   = await session.handle_refine("Reducing Triton to 0.5% works fine for HEK cells")
    summary, doc_url = await session.end_session()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

from .claude_client import (
    BASE_SYSTEM_PROMPT,
    ConversationHistory,
    build_system_prompt,
    call_claude,
    send_message,
    send_message_with_image,
)
from .config import SHEET_LAB_JOURNAL
from .google_client import (
    append_doc_text,
    append_sheet_row,
    find_experiments_doc_id,
    get_doc_url,
)
from .protocol_loader import load_protocol

logger = logging.getLogger(__name__)

_TZ = ZoneInfo("Asia/Jerusalem")

# ── Prompt templates ──────────────────────────────────────────────────────────

_DEVIATION_PREFIX = (
    "PROTOCOL DEVIATION LOGGED: {description}\n\n"
    "Please:\n"
    "1. Acknowledge the deviation.\n"
    "2. Identify which protocol step it affects (cite from the protocol if possible).\n"
    "3. Note any potential impact on results or downstream steps."
)

_REFINE_INSTRUCTION = (
    "Based on the session context and the finding below, draft a single concise "
    "knowledge note for this protocol's knowledge base.\n\n"
    "Format exactly as one line:\n"
    "[{date}, {researcher}] <finding in one or two sentences, actionable for future runs>\n\n"
    "Output only the formatted note — no preamble, no explanation.\n\n"
    "Finding: {finding}"
)

_SUMMARY_INSTRUCTION = (
    "Generate a structured session report in English with the following sections:\n\n"
    "1. **Work Completed** — key steps performed during this session.\n"
    "2. **Protocol Deviations** — any deviations logged (or 'None').\n"
    "3. **Buffer Preparations** — buffers prepared and their final volumes (or 'None').\n"
    "4. **Key Observations** — important findings, issues, or unexpected results.\n"
    "5. **Action Items** — follow-up steps needed.\n\n"
    "Be concise and factual. Do not invent information not discussed in the session."
)


# ── ProtocolSession ───────────────────────────────────────────────────────────


class ProtocolSession:
    """Active experiment session with the Protocol Expert skill loaded."""

    def __init__(
        self,
        protocol_name: str,
        protocol_version: str,
        companion_doc_id: Optional[str],
        experiments_doc_id: Optional[str],
        researcher_name: str,
        objective: str,
        system_prompt: str,
        protocol_folder_id: str = "",
        folder_name: str = "",
    ) -> None:
        self.protocol_name = protocol_name
        self.protocol_version = protocol_version
        self.companion_doc_id = companion_doc_id
        self.experiments_doc_id = experiments_doc_id
        self.researcher_name = researcher_name
        self.objective = objective
        self.system_prompt = system_prompt
        self.protocol_folder_id = protocol_folder_id
        self.folder_name = folder_name
        self.history = ConversationHistory()
        self.session_date = datetime.now(_TZ).strftime("%Y-%m-%d")
        self.session_time = datetime.now(_TZ).strftime("%H:%M")
        # Append-only log of key session events (deviations, notes, buffers).
        # Used for summary generation so we don't pay for the full history.
        self._event_log: list[str] = []

    @classmethod
    async def create(
        cls,
        protocol: dict,
        researcher_name: str,
        objective: str,
    ) -> "ProtocolSession":
        """Download the protocol + companion knowledge and build the skill.

        Args:
            protocol:        dict with keys 'id', 'name', 'modifiedTime'
                             (as returned by google_client.list_protocols).
            researcher_name: Display name of the researcher for this session.
            objective:       Session objective / target as typed by the researcher.
        """
        (
            protocol_text,
            companion_text,
            protocol_name,
            protocol_version,
            companion_doc_id,
        ) = await load_protocol(
            file_id=protocol["id"],
            file_name=protocol["docx_name"],
            modified_time=protocol.get("modifiedTime", ""),
            parent_folder_id=protocol.get("folder_id", ""),
            folder_name=protocol.get("name", ""),
            is_gdoc=protocol.get("is_gdoc", False),
        )

        folder_name = protocol.get("name", "")
        folder_id = protocol.get("folder_id", "")

        # Find pre-created experiments log doc
        experiments_doc_id = await find_experiments_doc_id(folder_name, folder_id)
        if experiments_doc_id:
            logger.info("Found experiments doc for '%s' (id=%s)", folder_name, experiments_doc_id)
        else:
            logger.warning("No experiments doc found for '%s' — live logging disabled", folder_name)

        system_prompt = build_system_prompt(
            protocol_text=protocol_text,
            companion_text=companion_text if companion_text.strip() else None,
            protocol_name=protocol_name,
            protocol_version=protocol_version,
        )

        session = cls(
            protocol_name=protocol_name,
            protocol_version=protocol_version,
            companion_doc_id=companion_doc_id,
            experiments_doc_id=experiments_doc_id,
            researcher_name=researcher_name,
            objective=objective,
            system_prompt=system_prompt,
            protocol_folder_id=folder_id,
            folder_name=folder_name,
        )

        # Write session header to experiments doc immediately
        await session._doc_write_header()

        return session

    # ── Live doc append ──────────────────────────────────────────────────────

    async def _doc_append(self, text: str) -> None:
        """Append text to the experiments doc. Silently logs on failure."""
        if not self.experiments_doc_id:
            return
        try:
            await append_doc_text(self.experiments_doc_id, text)
        except Exception as exc:
            logger.warning("Live-append to experiments doc failed: %s", exc)

    async def _doc_write_header(self) -> None:
        """Write the session header block at the start of a new session."""
        header = (
            f"\n{'═' * 60}\n"
            f"SESSION: {self.protocol_name}\n"
            f"Date: {self.session_date}  {self.session_time}\n"
            f"Researcher: {self.researcher_name}\n"
            f"Objective: {self.objective}\n"
            f"{'═' * 60}\n"
        )
        await self._doc_append(header)

    @property
    def experiments_doc_url(self) -> str:
        if self.experiments_doc_id:
            return get_doc_url(self.experiments_doc_id)
        return ""

    # ── Message routing ───────────────────────────────────────────────────────

    async def handle_message(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
        notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        """Route a text or image+text message through the Protocol Expert."""
        if image_bytes:
            reply = await send_message_with_image(
                self.history,
                image_bytes,
                text,
                system_prompt=self.system_prompt,
                notify_retry=notify_retry,
            )
        else:
            reply = await send_message(self.history, text, system_prompt=self.system_prompt, notify_retry=notify_retry)

        # Live-append the exchange to the experiments doc
        ts = datetime.now(_TZ).strftime("%H:%M")
        user_label = "[📷 Image] " if image_bytes else ""
        await self._doc_append(
            f"\n[{ts}] Researcher: {user_label}{text}\n"
            f"[{ts}] Assistant: {reply}\n"
        )
        return reply

    async def handle_deviation(self, description: str) -> str:
        """Log a protocol deviation and get Claude's acknowledgement + impact assessment."""
        self._event_log.append(f"[DEVIATION] {description}")
        prompt = _DEVIATION_PREFIX.format(description=description)
        reply = await send_message(self.history, prompt, system_prompt=self.system_prompt)
        ts = datetime.now(_TZ).strftime("%H:%M")
        await self._doc_append(
            f"\n[{ts}] ⚠️ DEVIATION: {description}\n"
            f"[{ts}] Assistant: {reply}\n"
        )
        return reply

    # ── Knowledge refinement ──────────────────────────────────────────────────

    async def handle_refine(self, finding: str) -> str:
        """Draft a dated knowledge note and append it to the companion Google Doc.

        Uses a standalone Claude call (not session history) so the drafted note
        does not pollute the conversation.
        """
        instruction = _REFINE_INSTRUCTION.format(
            date=self.session_date,
            researcher=self.researcher_name,
            finding=finding,
        )
        note = await call_claude(
            messages=[{"role": "user", "content": instruction}],
            system_prompt=self.system_prompt,
            max_tokens=256,
        )

        if self.companion_doc_id:
            try:
                await append_doc_text(self.companion_doc_id, note)
                return f"✅ Knowledge note saved to companion doc:\n\n_{note}_"
            except Exception as exc:
                logger.error("Could not append to companion doc: %s", exc)
                return (
                    f"✅ Knowledge note drafted (Drive save failed: {exc}):\n\n_{note}_"
                )
        else:
            return (
                f"✅ Knowledge note drafted.\n\n"
                f"_(To enable persistent saving, create a Google Doc named "
                f"'{self.protocol_name}_context' in the Protocols Drive folder.)_\n\n"
                f"_{note}_"
            )

    # ── Session end ───────────────────────────────────────────────────────────

    async def _generate_summary(self) -> str:
        """Ask Gemini to generate a structured summary from the event log + recent history.

        Uses a compact representation instead of full history to save tokens:
        - Key events (deviations, notes, findings) from _event_log
        - Last few turns of recent history for context
        """
        event_block = ""
        if self._event_log:
            event_block = "Key events this session:\n" + "\n".join(self._event_log) + "\n\n"

        # Include last 6 messages (3 turns) from history for recent context
        recent = self.history.messages[-6:] if len(self.history.messages) > 6 else self.history.messages
        recent_block = ""
        if recent:
            lines = []
            for msg in recent:
                role = "Researcher" if msg["role"] == "user" else "Assistant"
                text = " ".join(p.get("text", "") for p in msg["parts"] if isinstance(p, dict))
                if text.strip():
                    lines.append(f"{role}: {text.strip()[:300]}")
            if lines:
                recent_block = "Recent conversation (last 3 turns):\n" + "\n".join(lines) + "\n\n"

        summary_request = (
            f"{event_block}{recent_block}"
            f"Researcher: {self.researcher_name}\n"
            f"Objective: {self.objective}\n\n"
            f"{_SUMMARY_INSTRUCTION}"
        )
        return await call_claude(
            messages=[{"role": "user", "content": summary_request}],
            system_prompt=self.system_prompt,
            max_tokens=2048,
        )

    async def end_session(self) -> tuple[str, str]:
        """Close the session: generate summary, append to experiments doc, log to Lab Journal.

        Returns:
            (summary_text, experiments_doc_url)
        """
        summary = await self._generate_summary()

        # Append summary to the experiments doc
        ts = datetime.now(_TZ).strftime("%H:%M")
        summary_block = (
            f"\n{'─' * 40}\n"
            f"[{ts}] SESSION SUMMARY\n"
            f"{'─' * 40}\n\n"
            f"{summary}\n"
            f"\n{'═' * 60}\n"
            f"END OF SESSION\n"
            f"{'═' * 60}\n"
        )
        await self._doc_append(summary_block)

        doc_url = self.experiments_doc_url

        await append_sheet_row(
            SHEET_LAB_JOURNAL,
            [
                f"{self.protocol_name} — {self.session_date}",  # Exp Name
                self.session_date,                               # Date
                self.researcher_name,                            # Researcher
                self.protocol_name,                              # Protocol
                self.protocol_version,                           # Protocol Version
                self.objective,                                  # Objective / Target
                doc_url,                                         # Experiments Doc Link
                "Completed",                                     # Status
            ],
        )

        logger.info("Session ended: '%s' — experiments doc at %s", self.protocol_name, doc_url or "(none)")
        return summary, doc_url
