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
        ├─ Photo?  ──► Claude Vision (Anthropic) ──► analysis
        │
        └─ Text / transcript
                │
                ▼
     Protocol Expert Skill
     ┌───────────────────────────────────┐
     │  System Prompt =                  │
     │    BASE_PROMPT                    │
     │  + Protocol .docx text            │
     │  + Companion knowledge Doc text   │
     │  + Conversation history           │
     └───────────────────────────────────┘
                │
                ▼
          Google Drive
          ├── Protocols/          (.docx — source of truth)
          ├── Session Reports/    (one Google Doc per experiment)
          └── Exports/

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
1. User picks a protocol → `.docx` downloaded from Drive → body + tables extracted
2. Companion Google Doc loaded if it exists (`{protocol_name}_context`)
   - Contains: rationale, edge cases, known failure points, Gemini development history
3. System prompt assembled: `BASE_PROMPT + Protocol text + Companion knowledge`
4. Protocol version (filename + Drive `modifiedTime`) captured and stored in every record

**Session loop:**
- Every message → `system_prompt + full history + new message` → Claude
- Claude always has full protocol context and full session context simultaneously

**Knowledge updates:**
- `/refine` (anytime during session): user flags a finding → Claude drafts a dated knowledge note → appended to companion Google Doc immediately
- `/end` prompt: bot asks "Any findings to save to the knowledge base?" → same flow
- Over time: companion doc accumulates real-world experience across researchers

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
│     ├── [Protocol Expert skill loads — full protocol + companion in context]
│     ├── Any text/voice/photo → Protocol Expert → Claude responds with protocol context
│     ├── /buffer [name]    → Claude reads recipe → asks target volume → returns volumes/weights
│     ├── /deviation        → structured log: what changed vs. protocol step
│     ├── /calculate        → dilution / molarity / unit conversion
│     ├── /note             → explicit note entry
│     ├── /refine           → Claude drafts knowledge update → appended to companion Doc
│     └── /end              → session summary → Google Doc + Lab Journal row
│
└── 📦 Stock Orders (available always)
      ├── /order_item   → add row to Stock Orders sheet
      ├── /view_orders  → show Needed/Ordered items
      └── /mark_arrived → photo support (Claude extracts lot #) → Received Supplies row
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
├── .env.example          ← copy to .env and fill in your keys
├── .gitignore
├── requirements.txt
├── service_account.json  ← Google service account key (NOT committed to git)
├── src/
│   ├── __init__.py
│   ├── config.py         ← loads .env, all constants
│   ├── claude_client.py  ← AsyncAnthropic, ConversationHistory, system prompt builder
│   ├── transcription.py  ← Whisper-1 OGG transcription
│   ├── google_client.py  ← Drive / Sheets / Docs (Phase 2)
│   ├── protocol_loader.py← docx parse + companion Doc (Phase 2)
│   ├── protocol_skill.py ← Protocol Expert skill (Phase 2)
│   ├── stock.py          ← Stock order management (Phase 4)
│   ├── handlers.py       ← All Telegram handlers (Phase 2)
│   └── main.py           ← Entry point
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
