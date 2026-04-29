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
)
from .config import SHEET_LAB_JOURNAL
from .facs_calculator import (
    compute_facs,
    format_sheet_rows,
    format_telegram_summary,
    parse_cell_data,
)
from .google_client import (
    append_doc_text,
    append_experiment_rows,
    append_sheet_row,
    create_experiment_tab,
    find_experiments_sheet_id,
    get_sheet_url,
    load_general_methods,
)
from .protocol_loader import load_protocol
from .skill_retrieval import SkillIndex, clean_whitespace

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
        self._plate_layout_written: bool = False  # track if FACS plate layout has been written
        self._skill_index: SkillIndex = SkillIndex()  # keyword-based context retrieval

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

        is_facs = "bone marrow" in folder_name.lower() and "facs" in folder_name.lower()

        # Build a lean base system prompt with just protocol text (always relevant).
        # Companion + general methods go into the SkillIndex for per-message retrieval.
        system_prompt = build_system_prompt(
            protocol_text=clean_whitespace(protocol_text) if protocol_text else None,
            protocol_name=protocol_name,
            protocol_version=protocol_version,
            is_facs=is_facs,
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

        # Build skill index from companion doc + cross-method knowledge
        if companion_text and companion_text.strip():
            n = session._skill_index.add_document(clean_whitespace(companion_text), source=folder_name)
            logger.info("Indexed %d chunks from companion doc for '%s'", n, folder_name)

        general_methods_text = await load_general_methods()
        if general_methods_text and general_methods_text.strip():
            n = session._skill_index.add_document(clean_whitespace(general_methods_text), source="General Methods")
            logger.info("Indexed %d chunks from general_methods (%d total chunks)", n, session._skill_index.chunk_count)

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
            self._exp_tab_sheet_id, self._exp_tab_title = await create_experiment_tab(
                self._exp_spreadsheet_id, self._exp_tab_title
            )
            # Use method-specific template if available
            if self._is_facs_method():
                await self._sheet_init_facs()
            else:
                await self._sheet_init_default()
        except Exception as exc:
            logger.error("Failed to create experiment tab: %s", exc)
            self._exp_spreadsheet_id = None  # disable further writes

    def _is_facs_method(self) -> bool:
        """Check if this session is a Bone Marrow FACS experiment."""
        name = (self.folder_name or self.protocol_name).lower()
        return "bone marrow" in name and "facs" in name

    async def _sheet_init_default(self) -> None:
        """Write the default experiment sheet header."""
        await append_experiment_rows(
            self._exp_spreadsheet_id,
            self._exp_tab_title,
            [
                ["GENERAL INFO"],
                ["Protocol", self.protocol_name],
                ["Version", self.protocol_version],
                ["Date", self.session_date],
                ["Time", self.session_time],
                ["Researcher", self.researcher_name],
                ["Objective", self.objective],
                [],
                ["Time", "Section", "Content"],
            ],
        )

    async def _sheet_init_facs(self) -> None:
        """Write the FACS experiment sheet with structured layout.

        Creates a lean template with headers and reference data.
        The calculator agent populates sample table, plate layout,
        and master mix sections via [CALC_DATA] blocks.
        """
        rows = [
            # ── General Info ──
            ["GENERAL INFO"],
            ["Protocol", self.protocol_name],
            ["Version", self.protocol_version],
            ["Date", self.session_date],
            ["Time", self.session_time],
            ["Researcher", self.researcher_name],
            ["Objective", self.objective],
            [],
            # ── FACs Plate Layout (header only — bot fills via CALC_DATA) ──
            ["FACs PLATE LAYOUT"],
            ["(Will be populated by calculator agent based on treatment groups)"],
            [],
            # ── Sample Table (header only — bot fills via CALC_DATA) ──
            ["SAMPLE TABLE"],
            ["sample type", "Treatment", "Fraction", "IF condition",
             "Expected cells", "Actual cells", "Volume (µL)", "Resuspension vol", "Comments"],
            [],
            # ── Antibody Reference ──
            ["ANTIBODY PANEL"],
            ["Abs", "Fluorophore", "vol/1×10⁶ cells (µL)", "Laser", "Detector"],
            ["Biotin (Anti-lineage)", "Vio-Bright", "0.5", "488", "525-40"],
            ["SCA1", "PerCP-Vio 770", "2", "488", "690-50"],
            ["CD117", "PE", "2", "561", "585-42"],
            ["CD16/CD32", "PE-Vio", "2", "561", "610-20"],
            ["CD105", "PE-Vio770", "2", "561", "780-60"],
            ["CD41", "APC-Vio770", "2", "638", "780-60"],
            ["CD150", "BV605", "2", "405", "525-40"],
            ["SNIPER", "AF647", "use on origin only", "638", "660-10"],
            [],
            ["IgG ISOTYPE CONTROLS"],
            ["PE", "2 µL/1×10⁶"],
            ["PerCP-Vio700", "2 µL/1×10⁶"],
            ["PE-Vio770", "2 µL/1×10⁶"],
            ["APC-Vio770", "2 µL/1×10⁶"],
            [],
            # ── Master Mix (header only — bot fills via CALC_DATA) ──
            ["ANTIBODY MASTER MIX — ALL AB POOL"],
            ["(Calculator will compute based on actual cell counts)"],
            [],
            ["IgG CONTROL POOL"],
            ["(Calculator will compute based on actual cell counts)"],
            [],
            ["LIN(+) TUBES"],
            ["(Calculator will compute based on actual cell counts)"],
            [],
            # ── Zombie Staining (header only — bot fills via CALC_DATA) ──
            ["ZOMBIE STAINING"],
            ["(Calculator will compute based on number of samples)"],
            [],
            # ── Calculator Results ──
            ["CALCULATOR RESULTS"],
            [],
            # ── Session Log ──
            ["SESSION LOG"],
            ["Time", "Section", "Content"],
        ]
        await append_experiment_rows(
            self._exp_spreadsheet_id, self._exp_tab_title, rows,
        )

    async def _write_plate_layout(self, treatments: list[str]) -> None:
        """Write the FACS plate layout to the experiment sheet.

        Generates a standard 96-well plate layout with:
        - Row A: Single stain controls (from first treatment group's Lin(-))
        - Row B: Origin/unselected BM (All Abs, IgG, unstained per treatment)
        - Row C: Lin(-) cells (All Abs, IgG, unstained per treatment)
        - Row D: Lin(+) cells (noted as FACS tubes, not wells)
        """
        if not self._exp_spreadsheet_id or self._plate_layout_written:
            return

        n = len(treatments)
        # Build column headers: 3 wells per treatment (All Abs, IgG, Unstained)
        header = ["", "FACs plate", ""]
        col_num = 1
        for _ in treatments:
            header.extend([str(col_num), str(col_num + 1), str(col_num + 2)])
            col_num += 3
        # Pad to 12 columns
        while len(header) < 15:
            header.append("")

        # Row A: single stains
        row_a = ["1. single stains", "1", "A",
                 "Untreated", "Zombie", "Biotin VB",
                 "Sca1 PerCP", "CD117 PE", "CD16/32 PE-Vio",
                 "CD105 PE-V770", "CD41 APC-V770", "CD150 BV605",
                 "SNIPER AF647", "", ""]

        # Row B: Origin per treatment
        row_b = ["2. Origin (unselected BM)", "2", "B"]
        for t in treatments:
            row_b.extend([f"{t} All Abs+Z+Lin", f"{t} IgG+Z+Lin", f"{t} unstained"])
        while len(row_b) < 15:
            row_b.append("")

        # Row C: Lin(-) per treatment
        row_c = ["3. Lin(-) cells", "3", "C"]
        for t in treatments:
            row_c.extend([f"{t} All Abs+Z", f"{t} IgG+Z", f"{t} unstained"])
        while len(row_c) < 15:
            row_c.append("")

        # Row D: Lin(+) — FACS tubes
        row_d = ["4. Lin(+) cells", "", "D"]
        for t in treatments:
            row_d.extend([f"{t} All Abs+Z (TUBE)", f"{t} IgG+Z", f"{t} unstained"])
        while len(row_d) < 15:
            row_d.append("")

        # Empty rows E-H
        rows_empty = [["", "", r] for r in "EFGH"]

        # Treatment labels row
        label_row = ["", "", ""]
        for t in treatments:
            label_row.extend([t, "", ""])
        while len(label_row) < 15:
            label_row.append("")

        # Sample table with per-treatment rows
        sample_header = ["SAMPLE TABLE"]
        sample_cols = ["Sample type", "Fraction", "Treatment", "IF conditions",
                       "Expected cells", "Actual cells", "Volume", "Comments"]

        sample_rows = [sample_header, sample_cols]
        sample_rows.append(["Single stains", "Lin(-) from first group", treatments[0],
                            "Specific Abs + unstained + IgG", "", "75K each",
                            "*2µL each Ab, *0.5µL biotin", ""])
        for t in treatments:
            sample_rows.append(["Origin", "Unselected BM", t,
                                "1.All Abs  2.IgG  3.Unstained", "",
                                "1. 1×10⁶  2. 0.1×10⁶  3. 0.1×10⁶",
                                "Only ~1% are cells of interest", ""])
            sample_rows.append(["Lin(-)", "Selected BM", t,
                                "1.All Abs  2.IgG  3.Unstained", "",
                                "1. ALL remaining  2. 100K  3. 100K", "", ""])
            sample_rows.append(["Lin(+)", "Selected BM", t,
                                "1.All Abs  2.IgG  3.Unstained", "",
                                "1. 5×10⁶ (TUBE)  2. 0.2×10⁶  3. 0.2×10⁶",
                                "All Abs in FACS tubes", ""])

        all_rows = (
            [header, row_a, row_b, row_c, row_d]
            + rows_empty
            + [label_row, []]
            + sample_rows
        )

        await append_experiment_rows(
            self._exp_spreadsheet_id, self._exp_tab_title, all_rows,
        )
        self._plate_layout_written = True
        await self._sheet_log("📋 Plate Layout", f"Generated for {n} groups: {', '.join(treatments)}")
        logger.info("Wrote FACS plate layout for treatments: %s", treatments)

    def _parse_treatments(self, text: str) -> list[str]:
        """Extract treatment group names from user text."""
        lower = text.lower()
        treatments = []
        # Common patterns: "PBS and 5mg/kg", "two samples - PBS and 5mg/kg"
        # Look for explicit group names
        patterns = [
            r"(?:samples?|groups?|treatments?)\s*[-:—]\s*(.+)",
            r"(?:have|are|using)\s+(?:\w+\s+)?(?:samples?|groups?)\s*[-:—]?\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                groups_text = m.group(1)
                # Split on "and", ",", "+"
                parts = re.split(r"\s+and\s+|,\s*|\+\s*", groups_text)
                treatments = [p.strip().rstrip(".") for p in parts if p.strip()]
                break

        if not treatments:
            # Fallback: look for known treatment keywords
            known = ["pbs", "vehicle", "control", "untreated"]
            dose_pat = re.findall(r"\d+\s*(?:mg/?kg|µg|ug|nm|µm)", lower)
            for k in known:
                if k in lower:
                    treatments.append(k.upper() if k == "pbs" else k.capitalize())
            treatments.extend(dose_pat)

        return treatments

    async def _write_calc_table(self, calc_lines: list[str]) -> None:
        """Write calculator results to the experiment sheet.

        calc_lines: list of pipe-separated strings from [CALC_DATA] blocks.
        Each line is split on '|' into columns.
        """
        if not self._exp_spreadsheet_id:
            return
        rows = []
        for line in calc_lines:
            cells = [c.strip() for c in line.split("|")]
            if any(cells):
                rows.append(cells)
        if rows:
            await self._sheet_log("🧮 Calculator", f"Added {len(rows)} rows")
            await append_experiment_rows(
                self._exp_spreadsheet_id, self._exp_tab_title, rows,
            )

    async def _write_calc_rows(self, rows: list[list[str]]) -> None:
        """Write pre-formatted calculation rows to the experiment sheet."""
        if not self._exp_spreadsheet_id or not rows:
            return
        try:
            await self._sheet_log("🧮 Calculator", f"Added {len(rows)} rows")
            await append_experiment_rows(
                self._exp_spreadsheet_id, self._exp_tab_title, rows,
            )
        except Exception as exc:
            logger.error("Failed to write calc rows to sheet: %s", exc)

    @staticmethod
    def _extract_calc_fallback(reply: str) -> list[str]:
        """Extract calculation data from plain-text LLM reply as pipe-separated lines.

        Looks for patterns like:
          - "Label: value" lines (grouped into sections by headers)
          - Bullet points with calculations
        Returns pipe-separated lines suitable for _write_calc_table.
        """
        lines: list[str] = []
        current_section = ""
        for raw_line in reply.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            # Detect section headers (bold markdown or ALL CAPS)
            header_match = re.match(r"^\*\*(.+?)\*\*:?$", stripped)
            if header_match:
                current_section = header_match.group(1).strip()
                lines.append(current_section)
                continue
            if stripped.isupper() and len(stripped) > 3:
                current_section = stripped
                lines.append(current_section)
                continue
            # Detect "Label: value" or "- Label: value" patterns
            kv_match = re.match(
                r"^[-•*]?\s*(.+?):\s+(.+)$", stripped
            )
            if kv_match:
                key = kv_match.group(1).strip().lstrip("*").rstrip("*")
                val = kv_match.group(2).strip()
                # Skip lines that are just narrative explanations
                if len(val) > 100 or val.endswith("?"):
                    continue
                lines.append(f"{key} | {val}")

        # Only return if we got meaningful structured data (at least 3 data lines)
        data_lines = [l for l in lines if "|" in l]
        return lines if len(data_lines) >= 3 else []

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

    # ── FACS message helpers ────────────────────────────────────────────────

    def _build_prompt(self, text: str) -> str:
        """Build message-specific system prompt with relevant skill context."""
        skill_context = self._skill_index.retrieve(text)
        if not skill_context:
            return self.system_prompt
        return (
            self.system_prompt
            + "\n\n=== RELEVANT KNOWLEDGE (retrieved for this message) ===\n"
            + skill_context
        )

    # ── Message routing ───────────────────────────────────────────────────────

    async def handle_message(
        self,
        text: str,
        notify_retry: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        """Route a text message through the Protocol Expert.

        Images are pre-processed by the vision agent in handlers.py and arrive
        here as text descriptions — no image bytes ever enter session history.

        Automatically detects [OBS: ...] tags in the AI response, logs them
        to the event log and experiment sheet, and strips the tags from the
        reply shown to the user.

        For FACS sessions: parses cell data from LLM reply, runs deterministic
        calculations in Python, writes results to sheet, and appends a formatted
        summary to the reply.
        """
        prompt = self._build_prompt(text)

        reply = await send_message(self.history, text, system_prompt=prompt, notify_retry=notify_retry)

        # Extract auto-detected observations
        observations = re.findall(r"\[OBS:\s*(.+?)\]", reply)
        for obs in observations:
            self._event_log.append(f"[NOTE] {obs}")
            await self._sheet_log("📝 Auto-Note", obs)

        calc_summary = ""

        if self._is_facs_method():
            # Auto-generate plate layout from LLM reply if not written yet
            if not self._plate_layout_written:
                treatments = self._parse_treatments(text) or self._parse_treatments(reply)
                if treatments:
                    try:
                        await self._write_plate_layout(treatments)
                    except Exception as exc:
                        logger.warning("Failed to write plate layout: %s", exc)

            # Parse cell data and run code-based calculations
            cell_data = parse_cell_data(reply)
            if cell_data:
                # Also write plate layout if we haven't yet (treatments from cell data)
                if not self._plate_layout_written:
                    treatments = list(dict.fromkeys(d.treatment for d in cell_data))
                    if treatments:
                        try:
                            await self._write_plate_layout(treatments)
                        except Exception as exc:
                            logger.warning("Failed to write plate layout: %s", exc)

                results = compute_facs(cell_data)
                if results.samples:
                    # Write to experiment sheet
                    rows = format_sheet_rows(results)
                    await self._write_calc_rows(rows)
                    # Format summary for Telegram
                    calc_summary = format_telegram_summary(results)
                    logger.info("FACS calculator: %d samples, %d warnings",
                                len(results.samples), len(results.warnings))

        # Strip internal tags from the user-facing reply
        clean_reply = re.sub(r"\s*\[OBS:\s*.+?\]\s*", "\n", reply).strip()
        clean_reply = re.sub(
            r"\s*\[CELL_DATA\]\s*\n.*?(?:\[/CELL_DATA\]|\Z)",
            "\n", clean_reply, flags=re.DOTALL,
        ).strip()
        clean_reply = re.sub(
            r"\s*\[CALC_DATA\]\s*\n.*?(?:\[/CALC_DATA\]|\Z)",
            "\n", clean_reply, flags=re.DOTALL,
        ).strip()

        if calc_summary:
            clean_reply = clean_reply + "\n\n" + calc_summary

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
