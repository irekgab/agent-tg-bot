import logging
import asyncio
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from config import TELEGRAM_BOT_TOKEN
from logging_config import configure_logging
from graph import build_graph_async
import tg_bot_state
import tools
from tg_bot_handlers import (
    clear_command,
    stop_command,
    handle_photo,
    handle_document,
    handle_message,
)
from tg_msg_scheduler import schedule_followup
from tg_file_manager import send_file_callback

logger = logging.getLogger(__name__)

async def main() -> None:
    configure_logging()
    tg_bot_state.MAIN_LOOP = asyncio.get_running_loop()
    tools.SCHEDULE_CALLBACK = schedule_followup
    tools.SEND_FILE_CALLBACK = send_file_callback

    tg_bot_state.application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()

    tg_bot_state.application.add_handler(CommandHandler("clear", clear_command))
    tg_bot_state.application.add_handler(CommandHandler("stop", stop_command))
    tg_bot_state.application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    tg_bot_state.application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    tg_bot_state.application.add_handler(message_handler)

    logger.info("Telegram bot is starting...")

    async for app_instance in build_graph_async():
        tg_bot_state.app = app_instance

        await tg_bot_state.application.initialize()
        await tg_bot_state.application.start()
        await tg_bot_state.application.updater.start_polling()

        logger.info("Bot is polling and ready for messages.")

        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down bot...")
            await tg_bot_state.application.stop()
            await tg_bot_state.application.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped.")
