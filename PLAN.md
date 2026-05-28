# Lab Assistant — Project Plan

A Telegram-based AI lab assistant for molecular biology and cell biology bench work.
Supports text, voice (any language), and image input. Always responds in English.

---

## Architecture Overview

```
Researcher (Telegram)
        │  text / voice / photo
        ▼
  Telegram Bot  (python-telegram-bot v20+, async)
        │
        ├─ Voice? ──► Whisper-1 (OpenAI) ──► transcript text
        │
        ├─ Photo?  ──► Claude Vision (standalone) ──► analysis text ──► main agent
        │
        └─ Text / transcript
                │
                ▼
     Protocol Expert Skill  ◄──── SkillIndex (keyword-based chunk retrieval)
     ┌──────────────────────────────────────┐
     │  System Prompt =                     │
     │    BASE_PROMPT + Protocol .docx text  │
     │  + Per-message relevant chunks from:  │
     │    - Companion method_support Doc     │
     │    - general_methods_assistant        │
     │  + Conversation history               │
     └──────────────────────────────────────┘
                │
                ├─► Google Drive (protocol storage + session reports)
                ├─► Google Sheets (live experiment tabs per session)
                └─► ChromaDB (experiment database — Open Project queries)

          Google Drive — Lab Assistant/
          ├── Bone Marrow FACS/
          ├── Cell_franctionation/   ← cells + tissue protocol variants
          │   └── database/         ← protocol docs per starting material
          ├── HCR FISH/
          ├── stability/
          └── general protocols/    ← cross-method knowledge

          Google Sheets (Lab Assistant.gsheet)
          ├── Lab Journal
          ├── Stock Orders
          └── Received Supplies
```

---

## Technology Decisions

| Concern | Choice | Rationale |
|---|---|---|
| LLM | Anthropic `claude-3-5-sonnet-20241022` | Best reasoning + vision; 200K context handles full protocols |
| Voice transcription | OpenAI Whisper-1 | Native `.ogg Opus` support (Telegram format); multilingual auto-detect |
| Bot framework | `python-telegram-bot` v20+ | Async-native; clean ConversationHandler for multi-state flows |
| Back office | Google Drive + Sheets + Docs | Already in use; no new subscriptions needed |
| Protocol format | `.docx` on Google Drive | Existing workflow; text+tables extracted via `python-docx` |
| Auth (Google) | Service Account | Headless auth — no browser OAuth needed |
| Response language | Always English | Input can be any language; output always English |
| Dev environment | Jupyter notebooks + `nest_asyncio` | Iterative development; `nbconvert` exports to `.py` for production |

---

## Protocol Expert Skill — Core Architecture

The central feature. When a session starts the bot dynamically becomes an expert in the chosen protocol.

**Loading sequence:**
1. User picks a protocol → for Cell Fractionation, an additional "starting material" question (Cells vs Tissue) selects the correct protocol variant
2. `.docx` or Google Doc downloaded from Drive → body + tables extracted
3. Companion `method_support` Google Doc loaded if it exists (replaces legacy `{protocol_name}_context`)
4. `general_methods_assistant` loaded (cross-method bench knowledge)
5. Companion + general methods indexed into **SkillIndex** (keyword-based chunk retrieval, stays under token budget)
6. System prompt assembled: `BASE_PROMPT + Protocol text` (lean base); relevant chunks injected per-message
7. Protocol version (filename + Drive `modifiedTime`) captured and stored in every record

**Session loop:**
- Every message → SkillIndex selects relevant chunks → `system_prompt + chunks + full history + new message` → Claude
- Live experiment sheet tab created per session; events logged in real time

**Specialized behaviors:**
- **Bone Marrow FACS**: auto-generates plate layouts, color-coded treatment groups, FACS cell calculations
- **Cell Fractionation**: starting material question (cells vs tissue) loads the matching protocol variant from `database/`

**Knowledge updates:**
- `/refine` (anytime during session): user flags a finding → Claude drafts a dated knowledge note → appended to companion Google Doc immediately
- `/end` prompt: bot asks "Any findings to save to the knowledge base?" → same flow
- Over time: companion doc accumulates real-world experience across researchers

**Open Project (Experiment Database):**
- Say "open project experiment 547" or "open project [search term]"
- Loads historical experiment data from ChromaDB vector database
- Semantic search across all past experiments
- Can populate data into the active experiment sheet

---

## Google Sheets Structure

### Lab Journal
| Exp Name | Date | Researcher | Protocol | Protocol Version | Objective / Target | Session Doc Link | Status |

### Stock Orders
| Item Name | Catalog # | Qty | Unit | Supplier | Status | Requested By | Date Requested | Date Ordered | Date Arrived |

*Status values: Needed → Ordered → Arrived*

### Received Supplies
| Item | Lot # | Qty | Unit | Expiry Date | Storage Location | Date Received | Received By | Linked Order Row |

---

## Notebooks

Each notebook is a development and documentation artifact. Code lives in `src/`; notebooks import and demonstrate.

| # | File | Phase | Responsibility |
|---|---|---|---|
| 01 | `01_config.ipynb` | 1 | Environment setup, API keys, Drive/Sheets IDs, team map |
| 02 | `02_google_client.ipynb` | 1 | Drive service account auth; download `.docx`; Sheets CRUD; Docs create/read/append |
| 03 | `03_protocol_loader.ipynb` | 2 | python-docx extraction (body + tables); companion Doc loading; combined context string |
| 04 | `04_claude_integration.ipynb` | 1 | ConversationHistory; dynamic system prompt builder; text + image messages |
| 05 | `05_voice_transcription.ipynb` | 1 | Whisper-1 `.ogg → text`, auto language detect |
| 06 | `06_protocol_skill.ipynb` | 2 | Protocol Expert: prompt assembly, session routing, `/refine` handler |
| 07 | `07_stock_management.ipynb` | 4 | Stock order CRUD against Google Sheets |
| 08 | `08_telegram_handlers.ipynb` | 2 | ConversationHandler states, inline keyboards, all handler coroutines |
| 09 | `09_main.ipynb` | 1 | ApplicationBuilder, handler registration, `run_polling()` |
| 10 | `10_deployment.ipynb` | 5 | `nbconvert`, `.env.example`, systemd unit, Docker option |

---

## Bot Conversation Flow

```
/start → main menu (inline keyboard)
│
├── 🧪 Start Experiment
│     ├── Lists protocols from Drive → user picks one
│     ├── Cell Fractionation? → "Starting material?" (🧫 Cells / 🫀 Tissue)
│     ├── User enters session objective
│     ├── [Protocol Expert skill loads — protocol + SkillIndex context]
│     ├── [Live experiment sheet tab created]
│     ├── Any text/voice/photo → Protocol Expert → Claude responds with protocol context
│     ├── 🔬 Buffer            → Claude reads recipe → asks target volume → returns volumes/weights
│     ├── 📋 Deviation         → structured log: what changed vs. protocol step
│     ├── 🧮 Calculate         → dilution / molarity / unit conversion
│     ├── 📝 Note              → explicit note entry (timestamped)
│     ├── 📚 Refine            → Claude drafts knowledge update → appended to companion Doc
│     ├── "open project ..."   → load experiment data from ChromaDB
│     └── 🔚 End Session       → session summary → Lab Journal row + experiment sheet
│
├── 📦 Stock Orders (available always)
│     ├── /order_item   → add row to Stock Orders sheet
│     ├── /view_orders  → show Needed/Ordered items
│     └── /mark_arrived → photo support (Claude extracts lot #) → Received Supplies row
│
├── /settings → name, model preferences
│
└── Outside session: text/voice/photo → general lab AI assistant
```

---

## Implementation Phases

### Phase 1 — Foundation ✅ (current)
Files: `src/config.py`, `src/claude_client.py`, `src/transcription.py`, `src/main.py`
Notebooks: `01_config`, `04_claude_integration`, `05_voice_transcription`, `09_main`

Deliverable: a running Telegram bot that:
- Echoes text through Claude (molecular/cell biology context)
- Transcribes voice messages (any language) and processes with Claude
- Analyzes photos with Claude vision
- Shows placeholder main menu

### Phase 2 — Protocol Expert
Files: `src/google_client.py`, `src/protocol_loader.py`, `src/protocol_skill.py`, `src/handlers.py`
Notebooks: `02_google_client`, `03_protocol_loader`, `06_protocol_skill`, `08_telegram_handlers`

Deliverable:
- Pick protocol from Drive → Protocol Expert loaded
- `/buffer` calculates amounts from recipe in protocol
- `/end` saves session report to Google Doc + Lab Journal row

### Phase 3 — Deviations & Notes
Extension of Phase 2 handlers.

Deliverable:
- `/deviation` structured capture with step reference
- `/refine` appends to companion Doc immediately
- Photo of bench notes → Claude extracts → appended to session

### Phase 4 — Stock & Supply Management
Files: `src/stock.py`
Notebooks: `07_stock_management`

Deliverable:
- Full stock order lifecycle in Google Sheets
- `/mark_arrived` with photo support for lot # extraction via Claude vision

### Phase 5 — Production
Notebook: `10_deployment`

Deliverable:
- `nbconvert --to script` for all source notebooks
- Systemd service unit
- Docker container option
- Team onboarding guide (`TEAM_MEMBERS` map config)

---

## Directory Structure

```
lab_assistant/
├── PLAN.md
├── general_methods_assistant.md  ← cross-method bench knowledge (local fallback)
├── .env.example                  ← copy to .env and fill in your keys
├── .gitignore
├── requirements.txt
├── setup.py
├── service_account.json          ← Google service account key (NOT committed to git)
├── chroma_db/                    ← ChromaDB vector database (experiment records)
│   └── sync_state.json
├── scripts/
│   ├── fill_general_methods.py   ← generate cross-method knowledge via Gemini
│   ├── fill_method_support.py    ← generate per-method companion docs via Gemini
│   └── read_hcr_fish.py
├── src/
│   ├── __init__.py
│   ├── config.py                 ← loads .env, all constants
│   ├── claude_client.py          ← AsyncAnthropic, ConversationHistory, system prompt builder
│   ├── transcription.py          ← Whisper-1 OGG transcription with retry
│   ├── google_client.py          ← Drive / Sheets / Docs (protocol discovery + I/O)
│   ├── protocol_loader.py        ← docx parse + companion Doc loading
│   ├── protocol_skill.py         ← ProtocolSession: skill, sheet logging, FACS specialization
│   ├── skill_retrieval.py        ← SkillIndex: keyword-based chunk retrieval under token budget
│   ├── experiment_db.py          ← ChromaDB experiment database (Open Project)
│   ├── facs_calculator.py        ← Bone Marrow FACS cell count & dilution calculator
│   ├── user_settings.py          ← per-user name/model preferences
│   ├── handlers.py               ← Telegram ConversationHandler: all states & handlers
│   └── main.py                   ← Entry point
└── notebooks/
    ├── 01_config.ipynb
    ├── 02_google_client.ipynb
    ├── 03_protocol_loader.ipynb
    ├── 04_claude_integration.ipynb
    ├── 05_voice_transcription.ipynb
    ├── 06_protocol_skill.ipynb
    ├── 07_stock_management.ipynb
    ├── 08_telegram_handlers.ipynb
    ├── 09_main.ipynb
    └── 10_deployment.ipynb
```

---

## Phase 3: Session Resume (planned)

Multi-day experiments need the ability to pause and resume sessions without losing
the experiment sheet or conversation context.

### What survives a restart

| Data | Currently | Resume strategy |
|------|-----------|-----------------|
| Protocol/folder | In-memory | Re-derive from stored folder_id |
| Spreadsheet ID + tab | In-memory | Persist in JSON |
| Conversation history | In-memory | Rebuild from sheet event rows |
| Objective | In-memory | Persist in JSON |
| Researcher name | In-memory | Persist in JSON |
| Event log | In-memory + sheet | Sheet is source of truth |
| Database files loaded | In-memory skill_index | User re-loads if needed |
| Plate layout state | In-memory flags | Check if sheet already has layout |

### Implementation

**Storage**: `data/sessions/{user_id}.json` per user, tracking paused sessions:

```json
{
  "active_session": {
    "protocol_name": "Bone Marrow FACS",
    "protocol_folder_id": "...",
    "folder_name": "Bone Marrow FACS",
    "spreadsheet_id": "1zTr9AFE...",
    "tab_title": "2026-05-18 — דן",
    "tab_sheet_id": 814043357,
    "objective": "Assess biodistribution...",
    "researcher_name": "דן",
    "session_date": "2026-05-18",
    "plate_layout_written": true,
    "known_treatments": ["Vehicle Control", "NOV340 15mg IV"]
  }
}
```

**Commands**:
- `/resume` — list paused sessions, pick one to continue
- Auto-save on every sheet write (tab creation, deviations, notes, buffers)
- On `end_session` — delete saved state

**Resume flow**:
1. `/resume` → show saved session(s) as inline buttons
2. User picks one → re-create `ProtocolSession` with existing tab (no new tab)
3. Re-index protocol chunks into SkillIndex
4. Read last ~20 rows from sheet → inject as conversation context
5. Bot: "Resuming Bone Marrow FACS from 2026-05-18. Reading back progress..."
6. User continues working — all new events append to the same sheet tab

**Components to build**:
- `src/session_store.py` — JSON read/write for `data/sessions/`
- Save hooks in `ProtocolSession` (after tab creation + each sheet write)
- `ProtocolSession.resume()` classmethod (skip tab creation, load existing state)
- `/resume` command + handler in `handlers.py`
- Read-back logic: fetch last N rows from sheet tab via Sheets API
