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
    _fmt as _facs_fmt,
)
from .google_client import (
    append_doc_text,
    append_experiment_rows,
    append_sheet_row,
    batch_format,
    clear_range,
    create_experiment_tab,
    find_experiments_sheet_id,
    get_sheet_url,
    load_general_methods,
    read_sheet_rows,
    set_column_widths,
    write_range,
    COLORS,
    TREATMENT_COLORS,
    _cell_format,
    _repeat_cell_request,
    _update_borders_request,
    _merge_cells_request,
    _rgb,
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
        self._known_treatments: list[str] = []  # accumulated treatment groups
        self._skill_index: SkillIndex = SkillIndex()  # keyword-based context retrieval
        self._user_id: int = 0  # set externally for session persistence

    # ── Session persistence ──────────────────────────────────────────────────

    def to_state_dict(self) -> dict:
        """Serialize minimal session state for JSON persistence."""
        return {
            "protocol_name": self.protocol_name,
            "protocol_version": self.protocol_version,
            "protocol_folder_id": self.protocol_folder_id,
            "folder_name": self.folder_name,
            "spreadsheet_id": self._exp_spreadsheet_id or "",
            "tab_title": self._exp_tab_title,
            "tab_sheet_id": self._exp_tab_sheet_id,
            "objective": self.objective,
            "researcher_name": self.researcher_name,
            "session_date": self.session_date,
            "plate_layout_written": self._plate_layout_written,
            "known_treatments": self._known_treatments,
        }

    def save_state(self) -> None:
        """Persist current session state to disk (call after sheet writes)."""
        if not self._user_id:
            return
        from .session_store import save_session
        save_session(self._user_id, self.to_state_dict())

    @classmethod
    async def resume(
        cls,
        state: dict,
        user_id: int,
    ) -> "ProtocolSession":
        """Reconstruct a session from persisted state without creating a new tab.

        Re-indexes the protocol and reads back recent sheet rows as context.
        """
        from .protocol_loader import load_protocol

        # Build protocol dict for load_protocol
        folder_name = state["folder_name"]
        folder_id = state["protocol_folder_id"]

        # Load protocol text for skill indexing
        protocol_text = ""
        companion_text = ""
        protocol_name = state["protocol_name"]
        protocol_version = state["protocol_version"]

        # Try to reload protocol from Drive
        try:
            from .google_client import list_protocols
            protocols = await list_protocols()
            # Find matching protocol
            for p in protocols:
                if p.get("name") == folder_name or p.get("folder_id") == folder_id:
                    protocol_text, companion_text, protocol_name, protocol_version, _ = (
                        await load_protocol(
                            file_id=p["id"],
                            file_name=p["docx_name"],
                            modified_time=p.get("modifiedTime", ""),
                            parent_folder_id=p.get("folder_id", ""),
                            folder_name=p.get("name", ""),
                            is_gdoc=p.get("is_gdoc", False),
                        )
                    )
                    break
        except Exception as exc:
            logger.warning("Could not reload protocol for resume: %s", exc)

        is_facs = "bone marrow" in folder_name.lower() and "facs" in folder_name.lower()

        system_prompt = build_system_prompt(
            protocol_text=None,
            protocol_name=protocol_name,
            protocol_version=protocol_version,
            is_facs=is_facs,
        )

        session = cls(
            protocol_name=protocol_name,
            protocol_version=protocol_version,
            companion_doc_id=None,
            researcher_name=state["researcher_name"],
            objective=state["objective"],
            system_prompt=system_prompt,
            protocol_folder_id=folder_id,
            folder_name=folder_name,
            experiments_spreadsheet_id=state.get("spreadsheet_id") or None,
        )

        # Restore persisted state (skip tab creation)
        session._exp_tab_title = state["tab_title"]
        session._exp_tab_sheet_id = state["tab_sheet_id"]
        session._plate_layout_written = state.get("plate_layout_written", False)
        session._known_treatments = state.get("known_treatments", [])
        session.session_date = state.get("session_date", session.session_date)
        session._user_id = user_id

        # Re-index protocol
        if protocol_text and protocol_text.strip():
            n = session._skill_index.add_document(clean_whitespace(protocol_text), source="Protocol")
            logger.info("Resume: indexed %d protocol chunks for '%s'", n, protocol_name)

        # Index general methods
        general_methods_text = await load_general_methods()
        if general_methods_text and general_methods_text.strip():
            session._skill_index.add_document(clean_whitespace(general_methods_text), source="General Methods")

        # Read back recent rows from the experiment sheet as conversation context
        if session._exp_spreadsheet_id and session._exp_tab_title:
            try:
                rows = await read_sheet_rows(
                    session._exp_spreadsheet_id, session._exp_tab_title, max_rows=25
                )
                if rows:
                    # Format as readable context and inject into history
                    context_lines = []
                    for row in rows:
                        line = " | ".join(str(c) for c in row if c)
                        if line.strip():
                            context_lines.append(line)
                    if context_lines:
                        context_text = "\n".join(context_lines)
                        session.history.add_user(
                            f"[SESSION RESUMED — Previous experiment log from {session.session_date}]\n\n"
                            f"{context_text}"
                        )
                        session.history.add_assistant(
                            f"Session resumed. I can see the previous log entries from {session.session_date}. "
                            "I'll continue from where we left off. What would you like to do next?"
                        )
                        logger.info("Resume: injected %d sheet rows as context", len(context_lines))
            except Exception as exc:
                logger.warning("Resume: could not read back sheet rows: %s", exc)

        # Log the resume event to the sheet
        await session._sheet_log("🔄 Session Resumed", f"by {session.researcher_name}")

        return session

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

        # Build a lean base system prompt WITHOUT protocol text.
        # Protocol, companion, and general methods all go into the SkillIndex
        # so only relevant sections are retrieved per message (saves tokens).
        system_prompt = build_system_prompt(
            protocol_text=None,  # indexed in SkillIndex instead
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

        # Index protocol text for per-message retrieval (highest priority source)
        if protocol_text and protocol_text.strip():
            n = session._skill_index.add_document(clean_whitespace(protocol_text), source="Protocol")
            logger.info("Indexed %d chunks from protocol for '%s'", n, protocol_name)

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

        # Persist state so session can be resumed
        session.save_state()

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
                ["General info"],
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
        # Format header row
        sid = self._exp_tab_sheet_id
        fmt = [
            _repeat_cell_request(sid, 0, 1, 0, 3,
                                 _cell_format(bold=True, underline=True, bg_hex=COLORS["header_bg"])),
            _update_borders_request(sid, 0, 7, 0, 2),
            _repeat_cell_request(sid, 8, 9, 0, 3,
                                 _cell_format(bold=True, bg_hex=COLORS["header_bg"])),
        ]
        try:
            await batch_format(self._exp_spreadsheet_id, fmt)
        except Exception:
            pass

    async def _sheet_init_facs(self) -> None:
        """Write the FACS experiment sheet with structured layout.

        Fixed row layout (0-indexed for API, 1-indexed in comments):
        Row 0-6:   General info
        Row 7:     empty
        Row 8:     Plate layout header
        Row 9-17:  Plate layout (8 rows A-H + treatment labels)
        Row 18:    empty
        Row 19:    Sample table header
        Row 20:    Sample table column headers
        Row 21-38: Sample table data (up to 18 rows)
        Row 39:    empty
        Row 40:    Antibody panel header
        Row 41:    Ab panel column headers
        Row 42-50: Ab panel data
        Row 51:    empty
        Row 52:    IgG isotype header
        Row 53-57: IgG data
        Row 58:    empty
        Row 59:    Calculator results header (populated after cell counts)
        Row 60+:   Calculator fills in
        """
        # ── Row mapping for later updates ──
        self._row_plate_header = 8
        self._row_plate_start = 9
        self._row_sample_header = 19
        self._row_sample_cols = 20
        self._row_sample_data = 21
        self._row_ab_panel = 40
        self._row_igg_panel = 52
        self._row_calc_header = 59
        self._row_calc_start = 60

        rows = [
            # Row 0-6: General info
            ["General info"],
            ["Protocol", self.protocol_name],
            ["Version", self.protocol_version],
            ["Date", self.session_date],
            ["Time", self.session_time],
            ["Researcher", self.researcher_name],
            ["Objective", self.objective],
            [],
            # Row 8: Plate layout header
            ["Plate layout"],
            # Row 9: placeholder — will be filled by _write_plate_layout
            ["(awaiting treatment groups)"],
            [], [], [], [], [], [], [],
            [],
            # Row 19-20: Sample table
            ["Sample table"],
            ["Sample type", "Treatment", "Fraction", "Staining condition",
             "Expected cells", "Actual cells", "Volume (µL)", "Resuspension vol", "Comments"],
            # Rows 21-38: sample data placeholder
        ]
        # Pad sample rows
        for _ in range(18):
            rows.append([])
        rows.append([])  # Row 39: empty

        # Row 40-50: Antibody panel
        rows.extend([
            ["Antibody panel"],
            ["Antibody", "Fluorophore", "µL / 1×10⁶ cells", "Laser", "Detector"],
            ["Biotin (Anti-lineage)", "Vio-Bright", "0.5", "488", "525-40"],
            ["SCA1", "PerCP-Vio 770", "2", "488", "690-50"],
            ["CD117", "PE", "2", "561", "585-42"],
            ["CD16/CD32", "PE-Vio", "2", "561", "610-20"],
            ["CD105", "PE-Vio770", "2", "561", "780-60"],
            ["CD41", "APC-Vio770", "2", "638", "780-60"],
            ["CD150", "BV605", "2", "405", "525-40"],
            ["SNIPER", "AF647", "use on Origin only", "638", "660-10"],
            [],
        ])
        # Row 52-57: IgG isotype panel
        rows.extend([
            ["IgG isotype controls"],
            ["Isotype", "Volume per 1×10⁶ cells"],
            ["PE", "2 µL"],
            ["PerCP-Vio700", "2 µL"],
            ["PE-Vio770", "2 µL"],
            ["APC-Vio770", "2 µL"],
            [],
        ])
        # Row 59: Calculator results placeholder
        rows.extend([
            ["Calculator results"],
            ["(will be populated after cell count data is provided)"],
        ])

        await append_experiment_rows(
            self._exp_spreadsheet_id, self._exp_tab_title, rows,
        )

        # ── Apply formatting ──
        sid = self._exp_tab_sheet_id
        fmt_requests: list[dict] = []

        # Section headers: bold + underline, gray background
        header_rows = [0, 8, 19, 40, 52, 59]
        for r in header_rows:
            fmt_requests.append(_repeat_cell_request(
                sid, r, r + 1, 0, 10,
                _cell_format(bold=True, underline=True, bg_hex=COLORS["header_bg"], font_size=11),
            ))

        # Column header rows: bold, light gray bg
        col_header_rows = [20, 41, 53]
        for r in col_header_rows:
            fmt_requests.append(_repeat_cell_request(
                sid, r, r + 1, 0, 10,
                _cell_format(bold=True, bg_hex=COLORS["header_bg"]),
            ))

        # Table borders: Ab panel (rows 41-50), IgG panel (rows 53-57), sample table header
        fmt_requests.append(_update_borders_request(sid, 41, 51, 0, 5))
        fmt_requests.append(_update_borders_request(sid, 53, 57, 0, 2))
        fmt_requests.append(_update_borders_request(sid, 20, 21, 0, 9))

        # General info borders
        fmt_requests.append(_update_borders_request(sid, 0, 7, 0, 2))

        # Column widths
        try:
            await set_column_widths(self._exp_spreadsheet_id, sid, [
                (0, 1, 160),   # A: labels
                (1, 2, 130),   # B
                (2, 3, 120),   # C
                (3, 4, 140),   # D
                (4, 5, 120),   # E
                (5, 6, 100),   # F
                (6, 7, 100),   # G
                (7, 8, 120),   # H
                (8, 9, 100),   # I
            ])
        except Exception:
            pass  # non-critical

        try:
            await batch_format(self._exp_spreadsheet_id, fmt_requests)
        except Exception as exc:
            logger.warning("Sheet formatting failed (non-critical): %s", exc)

    async def _write_plate_layout(self, treatments: list[str]) -> None:
        """Write the FACS plate layout to the experiment sheet with color coding.

        Layout (rows 9-17 in the sheet):
        - Row 9:  Column numbers (1-12)
        - Row 10: Row A — Single stain controls (light blue)
        - Row 11: Row B — Origin samples (color per treatment)
        - Row 12: Row C — Lin(-) samples (color per treatment)
        - Row 13: Row D — Lin(+) samples (color per treatment, noted as FACS tubes)
        - Row 14-17: Rows E-H (empty)

        Colors: single stains = light blue, each treatment = distinct pastel color.
        Fractions: Origin = bold, Lin(-) = italic, Lin(+) = underline.
        """
        if not self._exp_spreadsheet_id:
            return

        n = len(treatments)
        plate_start = getattr(self, "_row_plate_start", 9)

        # Build plate layout data
        # Column headers
        col_headers = ["", ""]
        for i in range(1, 13):
            col_headers.append(str(i))

        # Row A: Single stain controls
        single_stains = [
            "Untreated", "Zombie", "Biotin VB", "Sca1 PerCP",
            "CD117 PE", "CD16/32 PE-Vio", "CD105 PE-V770",
            "CD41 APC-V770", "CD150 BV605", "SNIPER AF647",
        ]
        row_a = ["Single stains", "A"] + single_stains
        while len(row_a) < 14:
            row_a.append("")

        # Row B: Origin per treatment
        row_b = ["Origin", "B"]
        for t in treatments:
            row_b.extend([f"{t}\nAll Abs+Z+Lin", f"{t}\nIgG+Z+Lin", f"{t}\nUnstained"])
        while len(row_b) < 14:
            row_b.append("")

        # Row C: Lin(-) per treatment
        row_c = ["Lin(−)", "C"]
        for t in treatments:
            row_c.extend([f"{t}\nAll Abs+Z", f"{t}\nIgG+Z", f"{t}\nUnstained"])
        while len(row_c) < 14:
            row_c.append("")

        # Row D: Lin(+) per treatment — FACS tubes
        row_d = ["Lin(+)", "D"]
        for t in treatments:
            row_d.extend([f"{t}\nAll Abs+Z\n(TUBE)", f"{t}\nIgG+Z", f"{t}\nUnstained"])
        while len(row_d) < 14:
            row_d.append("")

        # Rows E-H: empty
        rows_eh = [["", chr(ord("E") + i)] + [""] * 12 for i in range(4)]

        # Treatment label row
        label_row = ["Treatment", ""]
        for t in treatments:
            label_row.extend([t, "", ""])
        while len(label_row) < 14:
            label_row.append("")

        all_rows = [col_headers, row_a, row_b, row_c, row_d] + rows_eh + [label_row]

        # Write data to the plate layout range
        end_row_idx = plate_start + len(all_rows)
        col_letter = chr(ord("A") + len(col_headers) - 1)
        await write_range(
            self._exp_spreadsheet_id,
            self._exp_tab_title,
            f"A{plate_start + 1}:{col_letter}{end_row_idx}",
            all_rows,
        )

        # ── Apply formatting ──
        sid = self._exp_tab_sheet_id
        fmt: list[dict] = []

        # Column header row (row numbers)
        fmt.append(_repeat_cell_request(
            sid, plate_start, plate_start + 1, 0, 14,
            _cell_format(bold=True),
        ))

        # Row labels column (A): bold
        fmt.append(_repeat_cell_request(
            sid, plate_start, end_row_idx, 0, 1,
            _cell_format(bold=True),
        ))

        # Single stains row (A): light blue background
        fmt.append(_repeat_cell_request(
            sid, plate_start + 1, plate_start + 2, 2, 12,
            _cell_format(bg_hex=COLORS["single_stain"]),
        ))

        # Origin row: bold text for fraction identification
        fmt.append(_repeat_cell_request(
            sid, plate_start + 2, plate_start + 3, 0, 1,
            _cell_format(bold=True),
        ))

        # Lin(-) row: italic text for fraction identification
        fmt.append(_repeat_cell_request(
            sid, plate_start + 3, plate_start + 4, 0, 1,
            _cell_format(italic=True),
        ))

        # Lin(+) row: underline text for fraction identification
        fmt.append(_repeat_cell_request(
            sid, plate_start + 4, plate_start + 5, 0, 1,
            _cell_format(underline=True),
        ))

        # Color each treatment's columns
        for ti, _t in enumerate(treatments):
            color = TREATMENT_COLORS[ti % len(TREATMENT_COLORS)]
            col_start = 2 + ti * 3
            col_end = col_start + 3
            # Apply to rows B, C, D (origin, lin-, lin+)
            for row_offset in range(2, 5):  # rows B=+2, C=+3, D=+4
                fmt.append(_repeat_cell_request(
                    sid, plate_start + row_offset, plate_start + row_offset + 1,
                    col_start, min(col_end, 14),
                    _cell_format(bg_hex=color),
                ))

        # Borders around the entire plate layout
        fmt.append(_update_borders_request(
            sid, plate_start, plate_start + 9, 0, 14,
        ))

        # Treatment label row: bold
        label_row_idx = plate_start + 9
        fmt.append(_repeat_cell_request(
            sid, label_row_idx, label_row_idx + 1, 0, 14,
            _cell_format(bold=True),
        ))

        try:
            await batch_format(self._exp_spreadsheet_id, fmt)
        except Exception as exc:
            logger.warning("Plate layout formatting failed: %s", exc)

        self._plate_layout_written = True
        self.save_state()
        logger.info("Wrote FACS plate layout for treatments: %s", treatments)

    def _parse_treatments(self, text: str) -> list[str]:
        """Extract treatment group names from user text or objective."""
        lower = text.lower()
        treatments = []
        # Common patterns: "PBS and 5mg/kg", "two samples - PBS and 5mg/kg"
        # Look for explicit group names
        patterns = [
            r"(?:samples?|groups?|treatments?)\s*[-:—]\s*(.+)",
            r"(?:have|are|using)\s+(?:\w+\s+)?(?:samples?|groups?)\s*[-:—]?\s*(.+)",
            # "PBS and 5mg/kg" standalone
            r"\b(pbs\s+and\s+\d+\s*mg/?kg)\b",
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
            # Fallback: look for known treatment keywords + dose patterns
            known = ["pbs", "vehicle", "control", "untreated"]
            dose_pat = re.findall(r"\d+\s*(?:mg/?kg|µg|ug|nm|µm)", lower)
            for k in known:
                if k in lower:
                    treatments.append(k.upper() if k == "pbs" else k.capitalize())
            treatments.extend(dose_pat)

        # Normalize: capitalize PBS, strip whitespace
        normalized = []
        for t in treatments:
            t = t.strip()
            if t.lower() == "pbs":
                t = "PBS"
            normalized.append(t)

        return normalized

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
        """Write calculator results to the fixed calculator section of the sheet."""
        if not self._exp_spreadsheet_id or not rows:
            return
        try:
            calc_start = getattr(self, "_row_calc_start", 60)
            end_row = calc_start + len(rows)
            # Determine max columns
            max_cols = max(len(r) for r in rows) if rows else 8
            col_letter = chr(ord("A") + max_cols - 1)
            await write_range(
                self._exp_spreadsheet_id,
                self._exp_tab_title,
                f"A{calc_start + 1}:{col_letter}{end_row}",
                rows,
            )

            # Format section headers and table borders
            sid = self._exp_tab_sheet_id
            fmt: list[dict] = []

            # Find header rows in the data and apply formatting
            for i, row in enumerate(rows):
                if row and len(row) == 1 and row[0]:
                    # Section header: bold + underline
                    fmt.append(_repeat_cell_request(
                        sid, calc_start + i, calc_start + i + 1, 0, max_cols,
                        _cell_format(bold=True, underline=True, bg_hex=COLORS["header_bg"]),
                    ))

            # Borders around the entire calc section
            if len(rows) > 1:
                fmt.append(_update_borders_request(
                    sid, calc_start, end_row, 0, max_cols,
                ))

            if fmt:
                await batch_format(self._exp_spreadsheet_id, fmt)

            logger.info("Wrote %d calc rows to sheet (rows %d-%d)", len(rows), calc_start, end_row)
        except Exception as exc:
            logger.error("Failed to write calc rows to sheet: %s", exc)

    async def _write_sample_table(self, results) -> None:
        """Write sample data to the fixed sample table section (rows 21+)."""
        if not self._exp_spreadsheet_id:
            return
        sample_start = getattr(self, "_row_sample_data", 21)

        rows: list[list[str]] = []
        # Single stains row
        if results.treatments:
            rows.append([
                "Single stains", results.treatments[0], "Lin(−)",
                "10 individual Abs + unstained",
                f"{_facs_fmt(results.single_stain_total)}",
                "", "75K each", "", "",
            ])

        for s in results.samples:
            tube = " (TUBE)" if s.get("tube") else ""
            rows.append([
                s["fraction"], s["treatment"], s["fraction"],
                f"All Abs / IgG / Unstained",
                f"{_facs_fmt(s['needed'])}",
                f"{_facs_fmt(s['total_cells'])}",
                f"{_facs_fmt(s['all_abs'])}{tube} / {_facs_fmt(s['igg'])} / {_facs_fmt(s['unstained'])}",
                "", "",
            ])

        if not rows:
            return

        end_row = sample_start + len(rows)
        await write_range(
            self._exp_spreadsheet_id,
            self._exp_tab_title,
            f"A{sample_start + 1}:I{end_row}",
            rows,
        )

        # Format: borders + fraction styling
        sid = self._exp_tab_sheet_id
        fmt: list[dict] = []
        fmt.append(_update_borders_request(sid, sample_start, end_row, 0, 9))

        # Color rows by treatment
        if results.treatments:
            for i, row in enumerate(rows):
                treatment = row[1] if len(row) > 1 else ""
                ti = next(
                    (j for j, t in enumerate(results.treatments) if t == treatment),
                    0,
                )
                color = TREATMENT_COLORS[ti % len(TREATMENT_COLORS)]
                fmt.append(_repeat_cell_request(
                    sid, sample_start + i, sample_start + i + 1, 0, 9,
                    _cell_format(bg_hex=color),
                ))

        try:
            await batch_format(self._exp_spreadsheet_id, fmt)
        except Exception as exc:
            logger.warning("Sample table formatting failed: %s", exc)

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
            # Auto-generate plate layout from text, reply, or objective
            need_plate = not self._plate_layout_written
            if self._plate_layout_written:
                low = text.lower()
                if any(kw in low for kw in ("missing", "update plate", "fix plate",
                                             "plate layout", "add treatment", "add group")):
                    need_plate = True

            if need_plate:
                # Gather treatments from all sources and merge with known
                new_treatments = (
                    self._parse_treatments(text)
                    or self._parse_treatments(reply)
                    or self._parse_treatments(self.objective)
                )
                if new_treatments:
                    merged = list(dict.fromkeys(
                        self._known_treatments + new_treatments
                    ))
                    try:
                        await self._write_plate_layout(merged)
                        self._known_treatments = merged
                    except Exception as exc:
                        logger.warning("Failed to write plate layout: %s", exc)

            # Parse cell data and run code-based calculations
            cell_data = parse_cell_data(reply)
            if cell_data:
                # Update plate layout from cell data treatments if needed
                data_treatments = list(dict.fromkeys(d.treatment for d in cell_data))
                if data_treatments:
                    merged = list(dict.fromkeys(
                        self._known_treatments + data_treatments
                    ))
                    if merged != self._known_treatments or not self._plate_layout_written:
                        try:
                            await self._write_plate_layout(merged)
                            self._known_treatments = merged
                        except Exception as exc:
                            logger.warning("Failed to write plate layout: %s", exc)

                results = compute_facs(cell_data)
                if results.samples:
                    # Write calculator results to sheet
                    rows = format_sheet_rows(results)
                    await self._write_calc_rows(rows)
                    # Write sample table to fixed section
                    await self._write_sample_table(results)
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

        # Remove persisted session state (experiment is complete)
        if self._user_id:
            from .session_store import delete_session
            delete_session(self._user_id)

        return summary, sheet_url
