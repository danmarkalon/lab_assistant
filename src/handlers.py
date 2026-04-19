"""
Telegram handlers — Phase 2 (Protocol Expert sessions).

Conversation states:
  PROTOCOL_SELECT   : user is choosing a protocol from the Drive list
  AWAITING_OBJECTIVE: protocol chosen, waiting for session objective text
  EXPERIMENT_ACTIVE : session in progress — all messages go through ProtocolSession
  DEVIATION_ENTRY   : waiting for deviation description text
  REFINE_ENTRY      : waiting for knowledge-refinement finding text
  CONFIRM_END       : waiting for end-of-session findings (or 'no' to skip)

Session state is stored in context.user_data['session'] (a ProtocolSession object).
The protocol list shown to the user is stored in context.user_data['protocols'] so
the CallbackQueryHandler can look up the selection by index.

Outside an experiment session (IDLE), text/voice/photo are routed through the plain
Phase-1-style handlers using per-user ConversationHistory objects.
"""

from __future__ import annotations

import io
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from .claude_client import BASE_SYSTEM_PROMPT, ConversationHistory, send_message, send_message_with_image
from .config import get_researcher_name, is_allowed
from .google_client import list_protocols
from .protocol_skill import ProtocolSession
from .transcription import transcribe_ogg
from .user_settings import (
    AVAILABLE_MODELS,
    get_all_settings,
    get_researcher_name as settings_get_name,
    get_user_model,
    set_setting,
)

logger = logging.getLogger(__name__)

# ── Conversation state constants ──────────────────────────────────────────────

PROTOCOL_SELECT     = 0
AWAITING_OBJECTIVE  = 1
EXPERIMENT_ACTIVE   = 2
DEVIATION_ENTRY     = 3
REFINE_ENTRY        = 4
CONFIRM_END         = 5

# Settings conversation states (offset to avoid collision)
SETTINGS_MENU       = 10
SETTINGS_EDIT_NAME  = 11

# ── Per-user fallback histories (used outside experiment sessions) ─────────────


async def _check_allowed(update: Update) -> bool:
    """Return True if the user is on the allowlist. Silently ignores unauthorized users
    but logs the attempt with their ID so the admin can add them."""
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        logger.warning(
            "Unauthorized access attempt from user_id=%s name=%s",
            user_id,
            update.effective_user.first_name,
        )
        return False
    return True

_histories: dict[int, ConversationHistory] = {}


def _get_history(user_id: int) -> ConversationHistory:
    if user_id not in _histories:
        _histories[user_id] = ConversationHistory()
    return _histories[user_id]


# ── Session reply keyboard ────────────────────────────────────────────────────


def _session_keyboard() -> ReplyKeyboardMarkup:
    """Persistent button bar shown during active experiment sessions."""
    return ReplyKeyboardMarkup(
        keyboard=[
            ["🔬 Buffer", "🧮 Calculate"],
            ["📋 Deviation", "📝 Note"],
            ["📚 Refine", "🔚 End Session"],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


# ── /start and /help (always active, outside ConversationHandler) ─────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_allowed(update):
        return
    user_id = update.effective_user.id
    tg_name = update.effective_user.first_name
    logger.info("/start from user_id=%s name=%s", user_id, tg_name)
    name = settings_get_name(user_id, tg_name)
    keyboard = [
        [InlineKeyboardButton("🧪 Start Experiment", callback_data="menu:start_experiment")],
        [InlineKeyboardButton("📦 Stock Orders", callback_data="menu:stock")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="menu:help")],
    ]
    await update.message.reply_text(
        f"Hello, {name}! I'm your lab assistant 🔬\n\n"
        "Tap <b>Start Experiment</b> to load a protocol and begin a session.\n"
        "Outside a session, just send me a message, voice, or photo for general assistance.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*Lab Assistant — Command Reference*\n\n"
        "*Session commands (use inside an experiment):*\n"
        "/start\\_experiment — pick a protocol and begin a session\n"
        "/buffer \\[name\\] — AI-assisted buffer preparation from protocol recipe\n"
        "/deviation — log a deviation from the protocol\n"
        "/calculate \\[query\\] — dilutions, molarity, unit conversions\n"
        "/refine — add a knowledge note to the protocol's knowledge base\n"
        "/note \\[text\\] — add an explicit timestamped note\n"
        "/end — close the session and save to Google Drive\n"
        "/cancel — abandon session without saving\n\n"
        "*Outside a session:*\n"
        "• Send any text → Claude lab assistant (no protocol context)\n"
        "• Send a voice message → transcribed then answered\n"
        "• Send a photo → Claude vision analysis\n\n"
        "*Stock orders (Phase 4):*\n"
        "/order\\_item · /view\\_orders · /mark\\_arrived",
        parse_mode="Markdown",
    )


# ── /start menu callback handler ─────────────────────────────────────────────


async def handle_menu_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handle taps from the inline keyboard shown by /start."""
    query = update.callback_query
    await query.answer()
    action = query.data.split(":")[1]

    if action == "start_experiment":
        await query.edit_message_text("⏳ Loading protocols from Drive...")
        try:
            protocols = await list_protocols()
        except Exception as exc:
            await query.edit_message_text(f"⚠️ Could not connect to Google Drive: {exc}")
            return ConversationHandler.END
        if not protocols:
            await query.edit_message_text(
                "No protocol .docx files found in the Drive folder.\n"
                "Upload your protocols and try again."
            )
            return ConversationHandler.END
        context.user_data["protocols"] = protocols
        keyboard = [
            [InlineKeyboardButton(p["name"], callback_data=f"proto:{i}")]
            for i, p in enumerate(protocols)
        ]
        await query.edit_message_text(
            "Select a protocol to begin:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return PROTOCOL_SELECT

    elif action == "help":
        await query.edit_message_text(
            "*Lab Assistant — Commands*\n\n"
            "*Session button bar:*\n"
            "🔬 Buffer — guided buffer prep from protocol recipe\n"
            "🧮 Calculate — dilutions, molarity, unit conversions\n"
            "📋 Deviation — log a protocol deviation\n"
            "📝 Note — add a timestamped note\n"
            "📚 Refine — add a finding to the knowledge base\n"
            "🔚 End Session — close session and save to Drive\n\n"
            "*Outside a session:*\n"
            "Send text, voice, or a photo for general lab AI assistance.",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    elif action == "stock":
        await query.edit_message_text("📦 Stock order management is coming in the next phase.")
        return ConversationHandler.END

    return ConversationHandler.END


# ── Entry point: /start_experiment ───────────────────────────────────────────


async def cmd_start_experiment(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """List protocols from Drive and ask the user to pick one."""
    if not await _check_allowed(update):
        return ConversationHandler.END
    await update.message.chat.send_action("typing")
    try:
        protocols = await list_protocols()
    except Exception as exc:
        await update.message.reply_text(
            f"⚠️ Could not connect to Google Drive: {exc}\n\n"
            "Please ensure GOOGLE_SERVICE_ACCOUNT_FILE and "
            "DRIVE_PROTOCOLS_FOLDER_ID are set in .env."
        )
        return ConversationHandler.END

    if not protocols:
        await update.message.reply_text(
            "No protocol .docx files found in the Protocols Drive folder.\n"
            "Upload your protocols and try again."
        )
        return ConversationHandler.END

    # Store the list so the callback handler can look up by index
    context.user_data["protocols"] = protocols
    keyboard = [
        [InlineKeyboardButton(p["name"], callback_data=f"proto:{i}")]
        for i, p in enumerate(protocols)
    ]
    await update.message.reply_text(
        "Select a protocol to begin:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return PROTOCOL_SELECT


# ── PROTOCOL_SELECT state ────────────────────────────────────────────────────


async def select_protocol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle protocol selection from the inline keyboard."""
    query = update.callback_query
    await query.answer()

    idx = int(query.data.split(":")[1])
    protocols: list = context.user_data.get("protocols", [])
    if idx >= len(protocols):
        await query.edit_message_text(
            "Protocol not found. Use /start_experiment to try again."
        )
        return ConversationHandler.END

    context.user_data["selected_protocol"] = protocols[idx]
    await query.edit_message_text(
        f"Protocol: *{protocols[idx]['name']}*\n\n"
        "What is the objective or target for this session?",
        parse_mode="Markdown",
    )
    return AWAITING_OBJECTIVE


# ── AWAITING_OBJECTIVE state ─────────────────────────────────────────────────


async def receive_objective(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Create the ProtocolSession and transition to EXPERIMENT_ACTIVE."""
    protocol = context.user_data["selected_protocol"]
    user_id = update.effective_user.id
    researcher = settings_get_name(user_id, update.effective_user.first_name)
    objective = update.message.text.strip()

    await update.message.reply_text(
        f"⏳ Loading *{protocol['name']}* and knowledge base...",
        parse_mode="Markdown",
    )
    await update.message.chat.send_action("typing")

    try:
        session = await ProtocolSession.create(
            protocol=protocol,
            researcher_name=researcher,
            objective=objective,
        )
    except Exception as exc:
        await update.message.reply_text(
            f"⚠️ Failed to load protocol: {exc}\n\nUse /cancel to abort."
        )
        return AWAITING_OBJECTIVE

    context.user_data["session"] = session

    kb_text = (
        "📚 Knowledge base: loaded ✅" if session.companion_doc_id
        else "📚 Knowledge base: none yet (create a `_context` Google Doc to start one)"
    )
    exp_text = (
        f"📝 Live experiment log: [open sheet]({session.experiments_sheet_url})" if session._exp_spreadsheet_id
        else (
            f"⚠️ No experiments sheet found. Create a Google Sheet named "
            f"`{session.folder_name}_experiments` in the protocol folder to enable live logging."
        )
    )
    await update.message.reply_text(
        f"✅ *Protocol Expert loaded*\n"
        f"Protocol: `{session.protocol_name}`\n"
        f"Version: `{session.protocol_version}`\n"
        f"{kb_text}\n"
        f"{exp_text}\n\n"
        f"Ask questions, send voice/photos, or tap the buttons below.",
        parse_mode="Markdown",
        reply_markup=_session_keyboard(),
    )
    return EXPERIMENT_ACTIVE


# ── EXPERIMENT_ACTIVE state ───────────────────────────────────────────────────


def _get_session(context: ContextTypes.DEFAULT_TYPE) -> ProtocolSession | None:
    return context.user_data.get("session")


async def active_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    if not session:
        return ConversationHandler.END
    await update.message.chat.send_action("typing")
    notify = lambda: update.message.reply_text("⏳ High traffic, hold on...")
    reply = await session.handle_message(update.message.text, notify_retry=notify)
    await update.message.reply_text(reply)
    return EXPERIMENT_ACTIVE


async def active_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    if not session:
        return ConversationHandler.END
    await update.message.chat.send_action("typing")

    voice_file = await update.message.voice.get_file()
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    transcript = await transcribe_ogg(buf.getvalue())

    await update.message.reply_text(f"🎙️ _{transcript}_", parse_mode="Markdown")
    notify = lambda: update.message.reply_text("⏳ High traffic, hold on...")
    reply = await session.handle_message(transcript, notify_retry=notify)
    await update.message.reply_text(reply)
    return EXPERIMENT_ACTIVE


async def active_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    if not session:
        return ConversationHandler.END
    await update.message.chat.send_action("typing")

    photo_file = await update.message.photo[-1].get_file()
    buf = io.BytesIO()
    await photo_file.download_to_memory(buf)

    caption = (
        update.message.caption
        or "Describe this lab image. Extract any text, labels, or measurements visible."
    )
    notify = lambda: update.message.reply_text("⏳ High traffic, hold on...")
    reply = await session.handle_message(caption, image_bytes=buf.getvalue(), notify_retry=notify)
    await update.message.reply_text(reply)
    return EXPERIMENT_ACTIVE


async def _button_buffer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.args = []
    return await cmd_buffer(update, context)


async def _button_calculate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.args = []
    return await cmd_calculate(update, context)


async def _button_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("📝 Send your note as the next message.")
    context.user_data["_note_mode"] = True
    return REFINE_ENTRY


async def cmd_buffer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    if not session:
        return EXPERIMENT_ACTIVE
    buffer_name = " ".join(context.args) if context.args else "the buffer"
    await update.message.chat.send_action("typing")
    session._event_log.append(f"[BUFFER PREP] {buffer_name}")
    prompt = (
        f"Buffer preparation request: {buffer_name}. "
        "Find the recipe in the protocol. "
        "Ask me for the target final volume, then calculate the exact volumes and weights needed."
    )
    reply = await session.handle_message(prompt)
    await session.log_buffer(buffer_name, reply)
    await update.message.reply_text(reply)
    return EXPERIMENT_ACTIVE


async def cmd_calculate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    query = " ".join(context.args) if context.args else ""
    await update.message.chat.send_action("typing")
    prompt = f"Lab calculation: {query}" if query else "I need help with a lab calculation."
    if session:
        reply = await session.handle_message(prompt)
        await session.log_dilution(f"{query}: {reply}" if query else reply)
    else:
        history = _get_history(update.effective_user.id)
        reply = await send_message(history, prompt)
    await update.message.reply_text(reply)
    return EXPERIMENT_ACTIVE


async def cmd_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    note = " ".join(context.args) if context.args else ""
    if not note:
        await update.message.reply_text(
            "Include your note after /note, e.g.:\n"
            "`/note centrifuge set to 9800 rpm instead of 10000`",
            parse_mode="Markdown",
        )
        return EXPERIMENT_ACTIVE
    await update.message.chat.send_action("typing")
    if session:
        session._event_log.append(f"[NOTE] {note}")
        await session.log_note(note)
        reply = await session.handle_message(f"Please record this note: {note}")
        await update.message.reply_text(f"📝 Note recorded:\n{reply}")
    else:
        await update.message.reply_text(f"📝 Note: {note}")
    return EXPERIMENT_ACTIVE


async def cmd_deviation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "📋 Describe the deviation from the standard protocol.\n\n"
        "Include: which step was changed, what was done differently, and why (if known)."
    )
    return DEVIATION_ENTRY


async def cmd_refine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "📝 What finding would you like to add to this protocol's knowledge base?\n\n"
        "(Describe the insight, issue, or tip that would help future runs.)"
    )
    return REFINE_ENTRY


async def cmd_end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🔚 Closing session.\n\n"
        "Any findings to save to this protocol's knowledge base?\n"
        "Reply with your finding, or send *no* to skip.",
        parse_mode="Markdown",
    )
    return CONFIRM_END


# ── DEVIATION_ENTRY state ─────────────────────────────────────────────────────


async def receive_deviation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    if not session:
        return ConversationHandler.END
    await update.message.chat.send_action("typing")
    reply = await session.handle_deviation(update.message.text)
    await update.message.reply_text(reply)
    return EXPERIMENT_ACTIVE


# ── REFINE_ENTRY state ────────────────────────────────────────────────────────


async def receive_refine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    if not session:
        return ConversationHandler.END
    await update.message.chat.send_action("typing")
    if context.user_data.pop("_note_mode", False):
        note_text = update.message.text
        await session.log_note(note_text)
        reply = await session.handle_message(f"Please record this note: {note_text}")
        await update.message.reply_text(f"📝 {reply}")
    else:
        reply = await session.handle_refine(update.message.text)
        await update.message.reply_text(reply)
    return EXPERIMENT_ACTIVE


# ── CONFIRM_END state ─────────────────────────────────────────────────────────


async def receive_end_findings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session = _get_session(context)
    if not session:
        return ConversationHandler.END

    text = update.message.text.strip().lower()
    await update.message.chat.send_action("typing")

    # Save a knowledge note if user provided a finding
    if text not in ("no", "skip", "none", "-"):
        refine_reply = await session.handle_refine(update.message.text)
        await update.message.reply_text(refine_reply)

    await update.message.reply_text("⏳ Generating session summary and saving to Drive...")

    try:
        summary, sheet_url = await session.end_session()
        # Truncate summary for Telegram (max message length)
        display = summary[:1800] + ("..." if len(summary) > 1800 else "")
        sheet_line = f"📄 [Open experiment sheet]({sheet_url})\n" if sheet_url else ""
        await update.message.reply_text(
            f"✅ *Session closed.*\n\n"
            f"{sheet_line}"
            f"📊 Added to Lab Journal sheet\n\n"
            f"*Summary:*\n{display}",
            parse_mode="Markdown",
            reply_markup=ReplyKeyboardRemove(),
        )
    except Exception as exc:
        logger.error("end_session failed: %s", exc)
        await update.message.reply_text(
            f"Session ended.\n"
            f"⚠️ Could not save to Google Drive: {exc}\n\n"
            "Please ensure Google API credentials are configured in .env.",
            reply_markup=ReplyKeyboardRemove(),
        )

    context.user_data.pop("session", None)
    context.user_data.pop("protocols", None)
    context.user_data.pop("selected_protocol", None)
    return ConversationHandler.END


# ── cancel fallback ───────────────────────────────────────────────────────────


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.pop("session", None)
    context.user_data.pop("protocols", None)
    context.user_data.pop("selected_protocol", None)
    context.user_data.pop("_note_mode", None)
    await update.message.reply_text(
        "Session cancelled. Use /start_experiment to begin a new session.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


# ── Fallback plain handlers (outside experiment sessions) ─────────────────────


async def fallback_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_allowed(update):
        return
    history = _get_history(update.effective_user.id)
    await update.message.chat.send_action("typing")
    notify = lambda: update.message.reply_text("⏳ High traffic, hold on...")
    reply = await send_message(history, update.message.text, notify_retry=notify)
    await update.message.reply_text(reply)


async def fallback_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_allowed(update):
        return
    await update.message.chat.send_action("typing")
    voice_file = await update.message.voice.get_file()
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    transcript = await transcribe_ogg(buf.getvalue())
    await update.message.reply_text(f"🎙️ _{transcript}_", parse_mode="Markdown")
    history = _get_history(update.effective_user.id)
    notify = lambda: update.message.reply_text("⏳ High traffic, hold on...")
    reply = await send_message(history, transcript, notify_retry=notify)
    await update.message.reply_text(reply)


async def fallback_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_allowed(update):
        return
    await update.message.chat.send_action("typing")
    photo_file = await update.message.photo[-1].get_file()
    buf = io.BytesIO()
    await photo_file.download_to_memory(buf)
    caption = (
        update.message.caption
        or "Describe this lab image. Extract any text, labels, or measurements visible."
    )
    history = _get_history(update.effective_user.id)
    notify = lambda: update.message.reply_text("⏳ High traffic, hold on...")
    reply = await send_message_with_image(history, buf.getvalue(), caption, notify_retry=notify)
    await update.message.reply_text(reply)


# ── /settings ─────────────────────────────────────────────────────────────────


def _settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    s = get_all_settings(user_id)
    current_name = s.get("name") or "not set (using Telegram name)"
    current_model = s.get("gemini_model") or "default (gemini-2.0-flash)"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"👤 Name: {current_name}",
            callback_data="settings:edit_name",
        )],
        [InlineKeyboardButton(
            f"🤖 Model: {current_model}",
            callback_data="settings:edit_model",
        )],
        [InlineKeyboardButton("✅ Done", callback_data="settings:done")],
    ])


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await update.message.reply_text(
        "⚙️ <b>Settings</b>\n\nTap a setting to change it.",
        reply_markup=_settings_keyboard(user_id),
        parse_mode="HTML",
    )
    return SETTINGS_MENU


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    action = query.data.split(":")[1]

    if action == "edit_name":
        s = get_all_settings(user_id)
        current = s.get("name") or ""
        await query.edit_message_text(
            f"👤 <b>Your display name</b>\n\n"
            f"Current: <code>{current or 'not set'}</code>\n\n"
            "Send your name as a message, or /cancel to go back.",
            parse_mode="HTML",
        )
        return SETTINGS_EDIT_NAME

    elif action == "edit_model":
        buttons = [
            [InlineKeyboardButton(
                f"{'✅ ' if get_all_settings(user_id).get('gemini_model') == mid else ''}{label}",
                callback_data=f"settings:model:{mid}",
            )]
            for mid, label in AVAILABLE_MODELS
        ]
        buttons.append([InlineKeyboardButton(
            f"{'✅ ' if not get_all_settings(user_id).get('gemini_model') else ''}Default (from .env)",
            callback_data="settings:model:default",
        )])
        await query.edit_message_text(
            "🤖 <b>Gemini model</b>\n\nChoose a model:",
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode="HTML",
        )
        return SETTINGS_MENU

    elif action.startswith("model:"):
        model_id = action.split("model:")[1]
        if model_id == "default":
            set_setting(user_id, "gemini_model", None)
            label = "default"
        else:
            set_setting(user_id, "gemini_model", model_id)
            label = model_id
        await query.edit_message_text(
            f"✅ Model set to <code>{label}</code>.\n\n⚙️ <b>Settings</b>",
            reply_markup=_settings_keyboard(user_id),
            parse_mode="HTML",
        )
        return SETTINGS_MENU

    elif action == "done":
        s = get_all_settings(user_id)
        name = s.get("name") or "Telegram name"
        model = s.get("gemini_model") or "default"
        await query.edit_message_text(
            f"✅ <b>Settings saved</b>\n\n"
            f"👤 Name: <code>{name}</code>\n"
            f"🤖 Model: <code>{model}</code>",
            parse_mode="HTML",
        )
        return ConversationHandler.END

    return SETTINGS_MENU


async def settings_edit_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    name = update.message.text.strip()
    if not name:
        await update.message.reply_text("Name can't be empty. Try again or /cancel.")
        return SETTINGS_EDIT_NAME
    set_setting(user_id, "name", name)
    await update.message.reply_text(
        f"✅ Name set to <b>{name}</b>.\n\nUse /settings to change other options.",
        parse_mode="HTML",
    )
    return ConversationHandler.END


async def settings_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Settings closed.")
    return ConversationHandler.END


def build_settings_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[CommandHandler("settings", cmd_settings)],
        states={
            SETTINGS_MENU: [
                CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
            ],
            SETTINGS_EDIT_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, settings_edit_name),
            ],
        },
        fallbacks=[CommandHandler("cancel", settings_cancel)],
        allow_reentry=True,
    )


# ── ConversationHandler factory ───────────────────────────────────────────────


def build_conversation_handler() -> ConversationHandler:
    """Build and return the experiment session ConversationHandler."""
    return ConversationHandler(
        entry_points=[
            CommandHandler("start_experiment", cmd_start_experiment),
            CallbackQueryHandler(handle_menu_callback, pattern=r"^menu:"),
        ],
        states={
            PROTOCOL_SELECT: [
                CallbackQueryHandler(select_protocol, pattern=r"^proto:\d+$"),
            ],
            AWAITING_OBJECTIVE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_objective),
            ],
            EXPERIMENT_ACTIVE: [
                CommandHandler("buffer", cmd_buffer),
                CommandHandler("calculate", cmd_calculate),
                CommandHandler("note", cmd_note),
                CommandHandler("deviation", cmd_deviation),
                CommandHandler("refine", cmd_refine),
                CommandHandler("end", cmd_end),
                # Reply keyboard button shortcuts
                MessageHandler(filters.Regex(r"^🔬 Buffer$"), _button_buffer),
                MessageHandler(filters.Regex(r"^🧮 Calculate$"), _button_calculate),
                MessageHandler(filters.Regex(r"^📋 Deviation$"), cmd_deviation),
                MessageHandler(filters.Regex(r"^📝 Note$"), _button_note),
                MessageHandler(filters.Regex(r"^📚 Refine$"), cmd_refine),
                MessageHandler(filters.Regex(r"^🔚 End Session$"), cmd_end),
                MessageHandler(filters.VOICE, active_voice),
                MessageHandler(filters.PHOTO, active_photo),
                MessageHandler(filters.TEXT & ~filters.COMMAND, active_text),
            ],
            DEVIATION_ENTRY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_deviation),
            ],
            REFINE_ENTRY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_refine),
            ],
            CONFIRM_END: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_end_findings),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        allow_reentry=True,
    )
