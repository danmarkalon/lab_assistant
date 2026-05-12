# Lab Assistant 🔬

A Telegram-based AI lab assistant for molecular biology and cell biology bench work. Supports text, voice (any language), and image input. Always responds in English.

## Features

- **Protocol Expert** — load a protocol from Google Drive and get an AI assistant that knows every step, buffer recipe, and edge case
- **Voice & Vision** — send voice messages in any language or photos of gels, plates, bench notes — the bot understands them all
- **Live Experiment Logging** — every session creates a Google Sheets tab with timestamped notes, deviations, buffer preps, and calculations
- **Knowledge Base** — each protocol accumulates lab-tested knowledge over time via `/refine` notes
- **Experiment Database** — say "open project experiment 547" to retrieve historical experiment data from ChromaDB
- **FACS Specialization** — Bone Marrow FACS sessions get auto-generated plate layouts and cell dilution calculations
- **Cell Fractionation Variants** — choose between cultured cells or tissue starting material, each loading the correct protocol
- **Cross-Method Knowledge** — shared bench skills (BCA, Jess Western, buffer math) available in every session

## Architecture

```
Researcher (Telegram)
        │  text / voice / photo
        ▼
  Telegram Bot (python-telegram-bot v20+, async)
        │
        ├─ Voice?  → Whisper-1 (OpenAI) → transcript
        ├─ Photo?  → Claude Vision → extracted text → main agent
        └─ Text    → Protocol Expert + SkillIndex → response
                         │
                         ├─► Google Drive   (protocols, session reports)
                         ├─► Google Sheets  (live experiment tabs)
                         └─► ChromaDB       (experiment database)
```

## Supported Methods

| Method | Folder | Special Features |
|---|---|---|
| Bone Marrow FACS | `Bone Marrow FACS/` | Plate layout, cell calculator |
| Cell Fractionation | `Cell_franctionation/` | Cells vs tissue variant selection |
| HCR FISH | `HCR FISH/` | — |
| Stability | `stability/` | — |

## Bot Commands

### During an experiment session
| Button / Command | Action |
|---|---|
| 🔬 Buffer | Guided buffer preparation from protocol recipe |
| 🧮 Calculate | Dilutions, molarity, unit conversions |
| 📋 Deviation | Log a protocol deviation |
| 📝 Note | Add a timestamped note |
| 📚 Refine | Add a finding to the knowledge base |
| 🔚 End Session | Close session, generate summary, save to Drive |

### General
| Command | Action |
|---|---|
| `/start` | Main menu |
| `/start_experiment` | Pick a protocol and begin a session |
| `/settings` | Name and model preferences |
| `open project experiment [N]` | Load experiment data from database |

## Setup

### Prerequisites
- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Google Cloud Service Account with Drive, Sheets, and Docs API access
- Gemini API key

### Installation

```bash
git clone https://github.com/Bioinfo5/lab_assistant.git
cd lab_assistant
pip install -r requirements.txt
```

### Configuration

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   TELEGRAM_BOT_TOKEN=...
   GEMINI_API_KEY=...
   DRIVE_ROOT_FOLDER_ID=...
   ```

2. Place your `service_account.json` in the project root.

3. Run the one-time setup to create Google Sheets structure:
   ```bash
   python setup.py
   ```

4. Create the expected Drive folder structure under your root folder:
   ```
   Lab Assistant/
   ├── Bone Marrow FACS/
   ├── Cell_franctionation/
   │   └── database/          ← protocol docs per starting material
   ├── HCR FISH/
   ├── stability/
   └── general protocols/
   ```

5. Upload your protocol `.docx` files into each method folder.

### Running

```bash
python -m src.main
```

## Project Structure

```
src/
├── config.py             # Environment variables and constants
├── claude_client.py      # LLM client, conversation history, prompt builder
├── transcription.py      # Voice transcription (Whisper-1)
├── google_client.py      # Drive, Sheets, Docs API
├── protocol_loader.py    # .docx parsing + companion doc loading
├── protocol_skill.py     # ProtocolSession: skill, sheet logging, FACS
├── skill_retrieval.py    # Keyword-based chunk retrieval (token budget)
├── experiment_db.py      # ChromaDB experiment database
├── facs_calculator.py    # FACS cell count & dilution calculator
├── user_settings.py      # Per-user preferences
├── handlers.py           # Telegram conversation states & handlers
└── main.py               # Entry point
```

See [PLAN.md](PLAN.md) for detailed architecture, design decisions, and implementation phases.

## License

Internal use only.
