"""FACS Calculator — deterministic cell count calculations.

Parses cell data from LLM responses and computes:
- Sample table (cells per condition)
- Antibody master mix volumes
- IgG control pool volumes
- Zombie staining volumes

All calculations follow the Bone Marrow FACS protocol rules.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Superscript helpers ───────────────────────────────────────────────────────

_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _parse_sup(s: str) -> int:
    return int("".join(_SUP.get(c, "") for c in s)) if s else 6


# ── Antibody panel ────────────────────────────────────────────────────────────
# (name, fluorophore, µL per 1×10⁶ cells)

AB_PANEL: list[tuple[str, str, float]] = [
    ("Biotin (Anti-lineage)", "Vio-Bright", 0.5),
    ("SCA1", "PerCP-Vio770", 2.0),
    ("CD117", "PE", 2.0),
    ("CD16/CD32", "PE-Vio", 2.0),
    ("CD105", "PE-Vio770", 2.0),
    ("CD41", "APC-Vio770", 2.0),
    ("CD150", "BV605", 2.0),
]
SNIPER = ("SNIPER", "AF647", 2.0)  # Origin-only

IGG_PANEL: list[tuple[str, float]] = [
    ("IgG PE", 2.0),
    ("IgG PerCP-Vio700", 2.0),
    ("IgG PE-Vio770", 2.0),
    ("IgG APC-Vio770", 2.0),
]

# ── Protocol constants ────────────────────────────────────────────────────────

ORIGIN_ALL_ABS = 1_000_000
ORIGIN_IGG = 100_000
ORIGIN_UNSTAINED = 100_000

LINNEG_IGG = 100_000
LINNEG_UNSTAINED = 100_000
SINGLE_STAIN_CELLS = 75_000
N_SINGLE_STAINS = 10

LINPOS_ALL_ABS = 5_000_000
LINPOS_IGG = 200_000
LINPOS_UNSTAINED = 200_000

STAINING_VOL_UL = 100
OVERAGE = 1.1
ZOMBIE_DILUTION = 1000
ZOMBIE_VOL_PER_SAMPLE = 100  # µL


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FractionData:
    treatment: str
    fraction: str          # "Origin", "Lin(-)", "Lin(+)"
    concentration: float   # cells/mL
    volume_ml: float

    @property
    def total_cells(self) -> float:
        return self.concentration * self.volume_ml


@dataclass
class CalculationResults:
    treatments: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    single_stain_total: float = 0
    ab_mix: list[dict] = field(default_factory=list)
    ab_mix_buffer: float = 0
    ab_mix_total: float = 0
    n_ab_wells: int = 0
    igg_mix: list[dict] = field(default_factory=list)
    igg_mix_buffer: float = 0
    igg_mix_total: float = 0
    n_igg_wells: int = 0
    zombie_samples: int = 0
    zombie_working_vol: float = 0
    zombie_stock_vol: float = 0
    warnings: list[str] = field(default_factory=list)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_cell_data(text: str) -> list[FractionData]:
    """Extract cell count data from LLM response.

    Tries [CELL_DATA] tags first, then falls back to natural-language parsing.
    """
    m = re.search(r"\[CELL_DATA\]\s*\n(.*?)\[/CELL_DATA\]", text, re.DOTALL)
    if m:
        result = _parse_block(m.group(1))
        if result:
            logger.info("Parsed %d fractions from [CELL_DATA] tags", len(result))
            return result
    result = _parse_natural(text)
    if result:
        logger.info("Parsed %d fractions from natural language", len(result))
    return result


def _parse_block(block: str) -> list[FractionData]:
    results: list[FractionData] = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("treatment"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        try:
            frac = _norm_fraction(parts[1])
            conc = _parse_num(parts[2])
            vol = _parse_num(parts[3])
            if frac and conc > 0 and vol > 0:
                results.append(FractionData(parts[0].strip(), frac, conc, vol))
        except (ValueError, IndexError):
            continue
    return results


_FRAC_RE = re.compile(
    r"(Origin|Lin\s*\([+-]\)|unselected\s*BM)\s*:\s*"
    r"([\d.,]+)\s*[×x]\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*"
    r"(?:cells?/?m[lL])?\s*[,;]\s*"
    r"([\d.,]+)\s*(?:m[lL])",
    re.IGNORECASE,
)

_FRACTION_NAMES = {"origin", "lin(-)", "lin(+)", "unselected bm"}


def _parse_natural(text: str) -> list[FractionData]:
    results: list[FractionData] = []
    current_treatment = ""

    for raw in text.splitlines():
        stripped = raw.strip().lstrip("-•* ")
        if not stripped:
            continue

        # Treatment header: standalone short label, not a fraction name
        clean = stripped.replace("*", "").strip().rstrip(":").strip()
        low = clean.lower()
        if (
            clean
            and len(clean) < 30
            and low not in _FRACTION_NAMES
            and not any(x in low for x in ("×10", "x10", "e6", "e7", "cells", "ml"))
            and any(c.isalnum() for c in clean)
            and ":" in stripped  # must look like a header with colon
        ):
            # Only accept as header if the rest after colon is empty
            after_colon = stripped.split(":", 1)[1].replace("*", "").strip()
            if not after_colon:
                current_treatment = clean
                continue

        if not current_treatment:
            continue

        m = _FRAC_RE.search(stripped)
        if m:
            frac = _norm_fraction(m.group(1))
            conc = float(m.group(2).replace(",", ".")) * (10 ** _parse_sup(m.group(3)))
            vol = float(m.group(4).replace(",", "."))
            if frac:
                results.append(FractionData(current_treatment, frac, conc, vol))

    return results


def _norm_fraction(s: str) -> str:
    s = s.strip()
    low = s.lower()
    if "origin" in low or "unselected" in low:
        return "Origin"
    if "lin" in low and "-" in low:
        return "Lin(-)"
    if "lin" in low and "+" in low:
        return "Lin(+)"
    return s


def _parse_num(s: str) -> float:
    s = s.strip().lower()
    s = re.sub(r"\s*(cells?/?ml|ml|µl|ul)\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.replace(",", ".")
    if "e" in s:
        return float(s)
    m = re.search(r"([\d.]+)\s*[×x]\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", s)
    if m:
        return float(m.group(1)) * (10 ** _parse_sup(m.group(2)))
    return float(s)


# ── Calculations ──────────────────────────────────────────────────────────────

def compute_facs(data: list[FractionData]) -> CalculationResults:
    res = CalculationResults()
    treatments = list(dict.fromkeys(d.treatment for d in data))  # preserve order
    res.treatments = treatments
    if not treatments:
        return res

    first_treatment = treatments[0]

    # Cells going to All-Abs and IgG conditions (for master-mix totals)
    abs_cells: list[tuple[str, str, float]] = []
    igg_cells: list[tuple[str, str, float]] = []

    for treatment in treatments:
        for frac in ("Origin", "Lin(-)", "Lin(+)"):
            fd = next((d for d in data if d.treatment == treatment and d.fraction == frac), None)
            if not fd:
                res.warnings.append(f"Missing data: {treatment} {frac}")
                continue

            total = fd.total_cells
            s: dict = {
                "treatment": treatment,
                "fraction": frac,
                "concentration": fd.concentration,
                "volume_ml": fd.volume_ml,
                "total_cells": total,
            }

            if frac == "Origin":
                s["all_abs"] = ORIGIN_ALL_ABS
                s["igg"] = ORIGIN_IGG
                s["unstained"] = ORIGIN_UNSTAINED
                s["needed"] = ORIGIN_ALL_ABS + ORIGIN_IGG + ORIGIN_UNSTAINED

            elif frac == "Lin(-)":
                reserved = LINNEG_IGG + LINNEG_UNSTAINED
                if treatment == first_treatment:
                    reserved += SINGLE_STAIN_CELLS * N_SINGLE_STAINS
                    res.single_stain_total = SINGLE_STAIN_CELLS * N_SINGLE_STAINS
                remaining = max(0, total - reserved)
                s["all_abs"] = remaining
                s["igg"] = LINNEG_IGG
                s["unstained"] = LINNEG_UNSTAINED
                if treatment == first_treatment:
                    s["single_stains"] = res.single_stain_total
                s["needed"] = total  # uses everything

            else:  # Lin(+)
                s["all_abs"] = min(LINPOS_ALL_ABS, total)
                s["igg"] = LINPOS_IGG
                s["unstained"] = LINPOS_UNSTAINED
                s["tube"] = True
                s["needed"] = s["all_abs"] + LINPOS_IGG + LINPOS_UNSTAINED

            if total < s["needed"]:
                res.warnings.append(
                    f"⚠️ {treatment} {frac}: need {_fmt(s['needed'])} but only have {_fmt(total)}"
                )

            abs_cells.append((treatment, frac, s["all_abs"]))
            igg_cells.append((treatment, frac, s["igg"]))
            res.samples.append(s)

    # ── Ab Master Mix ─────────────────────────────────────────────────────
    origin_cells = sum(n for _, f, n in abs_cells if f == "Origin")
    total_abs = sum(n for _, _, n in abs_cells)
    res.n_ab_wells = len(abs_cells)

    total_ab_vol = 0.0
    for name, fluor, vpm in AB_PANEL:
        v = vpm * total_abs / 1e6 * OVERAGE
        res.ab_mix.append({"name": name, "fluorophore": fluor,
                           "vol_per_1M": vpm, "total_cells": total_abs,
                           "total_vol": round(v, 1)})
        total_ab_vol += v

    sv = SNIPER[2] * origin_cells / 1e6 * OVERAGE
    res.ab_mix.append({"name": SNIPER[0], "fluorophore": SNIPER[1],
                       "vol_per_1M": SNIPER[2], "total_cells": origin_cells,
                       "total_vol": round(sv, 1), "origin_only": True})
    total_ab_vol += sv

    target = STAINING_VOL_UL * res.n_ab_wells * OVERAGE
    res.ab_mix_buffer = round(max(0, target - total_ab_vol), 1)
    res.ab_mix_total = round(total_ab_vol + res.ab_mix_buffer, 1)

    # ── IgG Pool ──────────────────────────────────────────────────────────
    total_igg = sum(n for _, _, n in igg_cells)
    res.n_igg_wells = len(igg_cells)

    total_igg_vol = 0.0
    for iname, vpm in IGG_PANEL:
        v = vpm * total_igg / 1e6 * OVERAGE
        res.igg_mix.append({"name": iname, "vol_per_1M": vpm,
                            "total_cells": total_igg, "total_vol": round(v, 1)})
        total_igg_vol += v

    target_igg = STAINING_VOL_UL * res.n_igg_wells * OVERAGE
    res.igg_mix_buffer = round(max(0, target_igg - total_igg_vol), 1)
    res.igg_mix_total = round(total_igg_vol + res.igg_mix_buffer, 1)

    # ── Zombie ────────────────────────────────────────────────────────────
    # All samples except unstained: AllAbs + IgG per fraction + single stains
    n_zombie = res.n_ab_wells + res.n_igg_wells + N_SINGLE_STAINS
    res.zombie_samples = n_zombie
    res.zombie_working_vol = round(ZOMBIE_VOL_PER_SAMPLE * n_zombie * OVERAGE, 1)
    res.zombie_stock_vol = round(res.zombie_working_vol / ZOMBIE_DILUTION, 2)

    return res


# ── Formatting ────────────────────────────────────────────────────────────────

def format_sheet_rows(results: CalculationResults) -> list[list[str]]:
    rows: list[list[str]] = []

    rows.append(["CELL COUNT SUMMARY"])
    rows.append(["Treatment", "Fraction", "Concentration", "Volume (mL)",
                 "Total cells", "All Abs", "IgG", "Unstained"])
    for s in results.samples:
        rows.append([
            s["treatment"], s["fraction"],
            _fmt(s["concentration"]) + "/mL", str(s["volume_ml"]),
            _fmt(s["total_cells"]),
            _fmt(s["all_abs"]) + (" (TUBE)" if s.get("tube") else ""),
            _fmt(s["igg"]), _fmt(s["unstained"]),
        ])
    rows.append([])

    if results.single_stain_total:
        rows.append(["SINGLE STAINS",
                     f"75K × {N_SINGLE_STAINS} = {_fmt(results.single_stain_total)}",
                     f"From {results.treatments[0]} Lin(-)"])
        rows.append([])

    rows.append(["ANTIBODY MASTER MIX (All Ab Pool)"])
    rows.append(["Antibody", "Fluorophore", "µL/1×10⁶", "Total cells", "Volume (µL)"])
    for ab in results.ab_mix:
        note = " (Origin only)" if ab.get("origin_only") else ""
        rows.append([ab["name"] + note, ab.get("fluorophore", ""),
                     str(ab["vol_per_1M"]), _fmt(ab["total_cells"]),
                     str(ab["total_vol"])])
    rows.append(["Staining buffer", "", "", "", str(results.ab_mix_buffer)])
    rows.append(["TOTAL", "", "", "", str(results.ab_mix_total)])
    rows.append([])

    rows.append(["IgG CONTROL POOL"])
    rows.append(["Isotype", "µL/1×10⁶", "Total cells", "Volume (µL)"])
    for ig in results.igg_mix:
        rows.append([ig["name"], str(ig["vol_per_1M"]),
                     _fmt(ig["total_cells"]), str(ig["total_vol"])])
    rows.append(["Staining buffer", "", "", str(results.igg_mix_buffer)])
    rows.append(["TOTAL", "", "", str(results.igg_mix_total)])
    rows.append([])

    rows.append(["ZOMBIE STAINING"])
    rows.append([f"Samples: {results.zombie_samples}",
                 f"Working solution: {results.zombie_working_vol} µL",
                 f"Stock: {results.zombie_stock_vol} µL (1:{ZOMBIE_DILUTION})"])
    rows.append([])

    if results.warnings:
        rows.append(["⚠️ WARNINGS"])
        for w in results.warnings:
            rows.append([w])
        rows.append([])

    return rows


def format_telegram_summary(results: CalculationResults) -> str:
    lines: list[str] = ["📊 FACS Calculator Results\n"]

    for s in results.samples:
        tube = " (TUBE)" if s.get("tube") else ""
        lines.append(f"{s['treatment']} {s['fraction']}: {_fmt(s['total_cells'])} total")
        lines.append(f"  All Abs: {_fmt(s['all_abs'])}{tube}")
        lines.append(f"  IgG: {_fmt(s['igg'])}  |  Unstained: {_fmt(s['unstained'])}")

    if results.single_stain_total:
        lines.append(f"\nSingle stains: {_fmt(results.single_stain_total)}"
                     f" from {results.treatments[0]} Lin(-)")

    lines.append(f"\n🧪 Ab Master Mix ({results.n_ab_wells} samples, {results.ab_mix_total} µL)")
    for ab in results.ab_mix:
        note = " ⚡Origin only" if ab.get("origin_only") else ""
        lines.append(f"  {ab['name']}: {ab['total_vol']} µL{note}")
    lines.append(f"  Buffer: {results.ab_mix_buffer} µL")

    lines.append(f"\n🧪 IgG Pool ({results.n_igg_wells} samples, {results.igg_mix_total} µL)")
    for ig in results.igg_mix:
        lines.append(f"  {ig['name']}: {ig['total_vol']} µL")
    lines.append(f"  Buffer: {results.igg_mix_buffer} µL")

    wv = results.zombie_working_vol
    sv = results.zombie_stock_vol
    lines.append(f"\n☠️ Zombie ({results.zombie_samples} samples)")
    lines.append(f"  Working sol: {wv} µL")
    lines.append(f"  Stock: {sv} µL in {wv - sv:.0f} µL PBS")

    if results.warnings:
        lines.append("")
        lines.extend(results.warnings)

    lines.append("\n✅ Written to experiment sheet")
    return "\n".join(lines)


def _fmt(n: float) -> str:
    if n == 0:
        return "0"
    if n >= 1e6:
        v = n / 1e6
        return f"{int(v)}×10⁶" if v == int(v) else f"{v:.1f}×10⁶"
    if n >= 1e3:
        v = n / 1e3
        return f"{int(v)}K" if v == int(v) else f"{v:.1f}K"
    return str(int(n))
