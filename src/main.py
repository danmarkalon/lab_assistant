"""
Lab Assistant bot entry point — Phase 2.

Handler registration:
  1. ConversationHandler  — experiment sessions (/start_experiment … /end)
  2. Plain command handlers — /start, /help, /calculate (outside session)
  3. Fallback message handlers — text / voice / photo for casual AI assistance

Run from project root:
    python -m src.main

Or from a Jupyter notebook (nest_asyncio handles the running event loop):
    from src.main import run
    run()
"""

import logging
import os

import nest_asyncio
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from .config import TELEGRAM_BOT_TOKEN


def _patch_ssl() -> None:
    """Point certifi (used by httpx) at the system CA bundle when running behind
    a corporate proxy that injects a self-signed cert into TLS chains."""
    ca_bundle = os.environ.get("HTTPLIB2_CA_CERTS") or os.environ.get("SSL_CERT_FILE")
    if ca_bundle:
        try:
            import certifi
            certifi.where = lambda: ca_bundle  # type: ignore[method-assign]
        except ImportError:
            pass


from .handlers import (
    build_conversation_handler,
    cmd_calculate,
    cmd_help,
    cmd_start,
    fallback_photo,
    fallback_text,
    fallback_voice,
    handle_menu_callback,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_app():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        # Increased timeouts for corporate proxies with high TLS overhead
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        # concurrent_updates=False is required when using ConversationHandler
        .concurrent_updates(False)
        .build()
    )

    # 1. Experiment session conversation handler (highest priority)
    app.add_handler(build_conversation_handler())

    # 2. Always-available command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("calculate", cmd_calculate))

    # 3. Fallback handlers for casual use outside experiment sessions
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_text))
    app.add_handler(MessageHandler(filters.VOICE, fallback_voice))
    app.add_handler(MessageHandler(filters.PHOTO, fallback_photo))

    return app


def run() -> None:
    """Start the bot. Works from the command line and from a Jupyter notebook."""
    _patch_ssl()
    nest_asyncio.apply()
    app = build_app()
    logger.info("Lab Assistant Phase 2 starting — polling for updates...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run()
