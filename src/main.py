"""
Phase 1 — Foundation bot entry point.

Capabilities in this phase:
- /start   : greeting + placeholder main menu (inline keyboard)
- /help    : lists all planned commands
- Text     : forwarded to Claude (molecular/cell biology lab context)
- Voice    : transcribed via Whisper-1, then forwarded to Claude
- Photo    : analysed by Claude vision

Run from project root:
    python -m src.main

Or inside a Jupyter notebook (nest_asyncio handles the running event loop):
    from src.main import run
    run()
"""

import io
import logging

import nest_asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .claude_client import (
    BASE_SYSTEM_PROMPT,
    ConversationHistory,
    send_message,
    send_message_with_image,
)
from .config import TELEGRAM_BOT_TOKEN, get_researcher_name
from .transcription import transcribe_ogg

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Per-user conversation histories  { user_id: ConversationHistory }
# Histories are kept in memory for the process lifetime.
# Phase 2+ will persist these (PicklePersistence) to survive restarts.
_histories: dict[int, ConversationHistory] = {}


def _get_history(user_id: int) -> ConversationHistory:
    if user_id not in _histories:
        _histories[user_id] = ConversationHistory()
    return _histories[user_id]


# ── Command handlers ──────────────────────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    name = get_researcher_name(user_id)
    keyboard = [
        [InlineKeyboardButton("🧪 Start Experiment", callback_data="start_experiment")],
        [InlineKeyboardButton("📦 Stock Orders", callback_data="stock_menu")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")],
    ]
    await update.message.reply_text(
        f"Hello, {name}! I'm your lab assistant 🔬\n\n"
        "Full protocol-expert mode is coming in Phase 2.\n"
        "For now, send me text, voice, or a photo and I'll assist you.",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*Lab Assistant*\n\n"
        "*Available now (Phase 1):*\n"
        "• Send any text message → Claude lab assistant\n"
        "• Send a voice message → transcribed then answered\n"
        "• Send a photo → Claude vision analysis\n\n"
        "*Coming in Phase 2:*\n"
        "/start\\_experiment — pick a protocol and begin a session\n"
        "/buffer — AI-assisted buffer preparation from protocol recipe\n"
        "/deviation — log a deviation from the protocol\n"
        "/calculate — dilutions, molarity, unit conversions\n"
        "/refine — add a knowledge note to the protocol knowledge base\n"
        "/note — add an explicit note to the session\n"
        "/end — close the session and save to Google Drive\n\n"
        "*Coming in Phase 4:*\n"
        "/order\\_item — add an item to the stock order list\n"
        "/view\\_orders — see open stock orders\n"
        "/mark\\_arrived — record a supply arrival",
        parse_mode="Markdown",
    )


# ── Message handlers ──────────────────────────────────────────────────────────


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history = _get_history(user_id)
    await update.message.chat.send_action("typing")
    reply = await send_message(history, update.message.text)
    await update.message.reply_text(reply)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await update.message.chat.send_action("typing")

    # Download OGG Opus voice file
    voice_file = await update.message.voice.get_file()
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)

    # Transcribe
    transcript = await transcribe_ogg(buf.getvalue())
    await update.message.reply_text(
        f"🎙️ _{transcript}_", parse_mode="Markdown"
    )

    # Forward transcript to Claude
    history = _get_history(user_id)
    reply = await send_message(history, transcript)
    await update.message.reply_text(reply)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await update.message.chat.send_action("typing")

    # Download highest-resolution version
    photo_file = await update.message.photo[-1].get_file()
    buf = io.BytesIO()
    await photo_file.download_to_memory(buf)

    caption = (
        update.message.caption
        or "Describe what you see in this lab image. "
           "Extract any text, labels, or measurements visible."
    )

    history = _get_history(user_id)
    reply = await send_message_with_image(history, buf.getvalue(), caption)
    await update.message.reply_text(reply)


# ── Application builder ───────────────────────────────────────────────────────


def build_app():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        # concurrent_updates=False is required when using ConversationHandler
        # (Phase 2); keep it here for consistency.
        .concurrent_updates(False)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    return app


def run() -> None:
    """Start the bot.  Works from the command line and from a Jupyter notebook."""
    nest_asyncio.apply()
    app = build_app()
    logger.info("Lab Assistant Phase 1 starting — polling for updates...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run()
