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
import re
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
    append_experiment_rows,
    append_sheet_row,
    create_experiment_tab,
    find_experiments_sheet_id,
    get_sheet_url,
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
        researcher_name: str,
        objective: str,
        system_prompt: str,
        protocol_folder_id: str = "",
        folder_name: str = "",
        experiments_spreadsheet_id: Optional[str] = None,
    ) -> None:
        self.protocol_name = protocol_name
        self.protocol_version = protocol_version
        self.companion_doc_id = companion_doc_id
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
        # Experiments spreadsheet (per-protocol, user-created)
        self._exp_spreadsheet_id = experiments_spreadsheet_id
        self._exp_tab_title: str = ""     # set in create() after tab creation
        self._exp_tab_sheet_id: int = 0   # gid for URL linking

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

        # Find pre-created experiments spreadsheet
        exp_sheet_id = await find_experiments_sheet_id(folder_name, folder_id)
        if exp_sheet_id:
            logger.info("Found experiments spreadsheet for '%s' (id=%s)", folder_name, exp_sheet_id)
        else:
            logger.warning("No experiments spreadsheet for '%s' — live logging disabled", folder_name)

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
            researcher_name=researcher_name,
            objective=objective,
            system_prompt=system_prompt,
            protocol_folder_id=folder_id,
            folder_name=folder_name,
            experiments_spreadsheet_id=exp_sheet_id,
        )

        # Create a new tab for this experiment and write header
        await session._sheet_init()

        return session

    # ── Live experiment sheet logging ────────────────────────────────────────

    async def _sheet_init(self) -> None:
        """Create a new tab in the experiments spreadsheet and write the header."""
        if not self._exp_spreadsheet_id:
            return
        self._exp_tab_title = f"{self.session_date} — {self.researcher_name}"
        try:
            self._exp_tab_sheet_id = await create_experiment_tab(
                self._exp_spreadsheet_id, self._exp_tab_title
            )
            # Write general info + section headers
            await append_experiment_rows(
                self._exp_spreadsheet_id,
                self._exp_tab_title,
                [
                    # ── General Info ──
                    ["GENERAL INFO"],
                    ["Protocol", self.protocol_name],
                    ["Version", self.protocol_version],
                    ["Date", self.session_date],
                    ["Time", self.session_time],
                    ["Researcher", self.researcher_name],
                    ["Objective", self.objective],
                    [],
                    # ── Column headers for log entries ──
                    ["Time", "Section", "Content"],
                ],
            )
        except Exception as exc:
            logger.error("Failed to create experiment tab: %s", exc)
            self._exp_spreadsheet_id = None  # disable further writes

    async def _sheet_log(self, section: str, content: str) -> None:
        """Append a single structured row to the experiment tab."""
        if not self._exp_spreadsheet_id:
            return
        ts = datetime.now(_TZ).strftime("%H:%M")
        try:
            await append_experiment_rows(
                self._exp_spreadsheet_id, self._exp_tab_title,
                [[ts, section, content]],
            )
        except Exception as exc:
            logger.warning("Live-append to experiment sheet failed: %s", exc)

    @property
    def experiments_sheet_url(self) -> str:
        if self._exp_spreadsheet_id:
            return get_sheet_url(self._exp_spreadsheet_id, self._exp_tab_sheet_id)
        return ""

    # ── Message routing ───────────────────────────────────────────────────────

    async def handle_message(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
        notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        """Route a text or image+text message through the Protocol Expert.

        Automatically detects [OBS: ...] tags in the AI response, logs them
        to the event log and experiment sheet, and strips the tags from the
        reply shown to the user.
        """
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

        # Extract auto-detected observations
        observations = re.findall(r"\[OBS:\s*(.+?)\]", reply)
        for obs in observations:
            self._event_log.append(f"[NOTE] {obs}")
            await self._sheet_log("📝 Auto-Note", obs)

        # Strip [OBS: ...] tags from the user-facing reply
        clean_reply = re.sub(r"\s*\[OBS:\s*.+?\]\s*", "\n", reply).strip()

        return clean_reply

    async def handle_deviation(self, description: str) -> str:
        """Log a protocol deviation and get Claude's acknowledgement + impact assessment."""
        self._event_log.append(f"[DEVIATION] {description}")
        prompt = _DEVIATION_PREFIX.format(description=description)
        reply = await send_message(self.history, prompt, system_prompt=self.system_prompt)
        await self._sheet_log("⚠️ Deviation", description)
        return reply

    async def log_note(self, note: str) -> None:
        """Record a note to the experiment sheet."""
        await self._sheet_log("📝 Note", note)

    async def log_buffer(self, buffer_name: str, details: str) -> None:
        """Record a buffer preparation to the experiment sheet."""
        await self._sheet_log("🧪 Buffer Prep", f"{buffer_name}: {details}")

    async def log_dilution(self, details: str) -> None:
        """Record a dilution/calculation to the experiment sheet."""
        await self._sheet_log("🔬 Dilution/Calc", details)

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
        """Close the session: generate summary, append to experiment sheet, log to Lab Journal.

        Returns:
            (summary_text, experiments_sheet_url)
        """
        summary = await self._generate_summary()

        # Write summary to experiment sheet
        await self._sheet_log("📋 Summary", summary)
        await self._sheet_log("🔚 Session End", "")

        sheet_url = self.experiments_sheet_url

        await append_sheet_row(
            SHEET_LAB_JOURNAL,
            [
                f"{self.protocol_name} — {self.session_date}",  # Exp Name
                self.session_date,                               # Date
                self.researcher_name,                            # Researcher
                self.protocol_name,                              # Protocol
                self.protocol_version,                           # Protocol Version
                self.objective,                                  # Objective / Target
                sheet_url,                                       # Experiment Sheet Link
                "Completed",                                     # Status
            ],
        )

        logger.info("Session ended: '%s' — experiment sheet at %s", self.protocol_name, sheet_url or "(none)")
        return summary, sheet_url
