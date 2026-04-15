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
from datetime import datetime, timezone
from typing import Optional

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
    create_session_doc,
    get_doc_url,
)
from .protocol_loader import load_protocol

logger = logging.getLogger(__name__)

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
        researcher_name: str,
        objective: str,
        system_prompt: str,
    ) -> None:
        self.protocol_name = protocol_name
        self.protocol_version = protocol_version
        self.companion_doc_id = companion_doc_id
        self.researcher_name = researcher_name
        self.objective = objective
        self.system_prompt = system_prompt
        self.history = ConversationHistory()
        self.session_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
            file_name=protocol["name"],
            modified_time=protocol.get("modifiedTime", ""),
        )

        system_prompt = build_system_prompt(
            protocol_text=protocol_text,
            companion_text=companion_text if companion_text.strip() else None,
            protocol_name=protocol_name,
            protocol_version=protocol_version,
        )

        return cls(
            protocol_name=protocol_name,
            protocol_version=protocol_version,
            companion_doc_id=companion_doc_id,
            researcher_name=researcher_name,
            objective=objective,
            system_prompt=system_prompt,
        )

    # ── Message routing ───────────────────────────────────────────────────────

    async def handle_message(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
    ) -> str:
        """Route a text or image+text message through the Protocol Expert.

        Updates conversation history so Claude retains full session context.
        """
        if image_bytes:
            return await send_message_with_image(
                self.history,
                image_bytes,
                text,
                system_prompt=self.system_prompt,
            )
        return await send_message(self.history, text, system_prompt=self.system_prompt)

    async def handle_deviation(self, description: str) -> str:
        """Log a protocol deviation and get Claude's acknowledgement + impact assessment."""
        prompt = _DEVIATION_PREFIX.format(description=description)
        return await send_message(self.history, prompt, system_prompt=self.system_prompt)

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
        """Ask Claude to generate a structured summary from the session history."""
        return await call_claude(
            messages=self.history.messages + [
                {"role": "user", "content": _SUMMARY_INSTRUCTION}
            ],
            system_prompt=self.system_prompt,
            max_tokens=2048,
        )

    async def end_session(self) -> tuple[str, str]:
        """Close the session: generate summary, save to Google Doc, log to Lab Journal.

        Returns:
            (summary_text, doc_url)
        """
        summary = await self._generate_summary()

        doc_title = (
            f"{self.protocol_name} — {self.session_date} — {self.researcher_name}"
        )
        doc_id = await create_session_doc(doc_title)

        report = (
            f"Protocol: {self.protocol_name}\n"
            f"Version: {self.protocol_version}\n"
            f"Date: {self.session_date}\n"
            f"Researcher: {self.researcher_name}\n"
            f"Objective: {self.objective}\n"
            f"\n{'=' * 60}\nSESSION SUMMARY\n{'=' * 60}\n\n"
            f"{summary}"
        )
        await append_doc_text(doc_id, report)

        doc_url = get_doc_url(doc_id)
        await append_sheet_row(
            SHEET_LAB_JOURNAL,
            [
                f"{self.protocol_name} — {self.session_date}",  # Exp Name
                self.session_date,                               # Date
                self.researcher_name,                            # Researcher
                self.protocol_name,                              # Protocol
                self.protocol_version,                           # Protocol Version
                self.objective,                                  # Objective / Target
                doc_url,                                         # Session Doc Link
                "Completed",                                     # Status
            ],
        )

        logger.info("Session ended: '%s' — report at %s", doc_title, doc_url)
        return summary, doc_url
