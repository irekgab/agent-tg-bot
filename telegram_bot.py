import logging
from logging_config import configure_logging

import asyncio
import sqlite3
import time
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from telegramify_markdown import markdownify

from config import TELEGRAM_BOT_TOKEN
from graph import build_graph_async
from streaming import extract_text

logger = logging.getLogger(__name__)

app = None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the user types /start."""
    await update.message.reply_text(markdownify(
        "*Hello! I am your AI Agent.*\n\n"
        "You can chat with me, ask me to read files, "
        "execute commands, or even make web requests!\n\n"
        "*Commands:*\n"
        "/start - Show this message\n"
        "/clear - Reset your conversation history"),
        parse_mode=constants.ParseMode.MARKDOWN_V2
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history for the current user."""
    user_id = str(update.effective_user.id)
    
    try:
        def clear_db():
            conn = sqlite3.connect(".data/history.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (user_id,))
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (user_id,))
            conn.commit()
            conn.close()
        
        await asyncio.to_thread(clear_db)
        
        await update.message.reply_text(
            markdownify("Your conversation history has been cleared! You can start a new chat now."),
            parse_mode=constants.ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        logger.error(f"Error clearing history for user {user_id}: {e}")
        await update.message.reply_text("Failed to clear conversation history. Please try again later.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages from Telegram users with streaming responses and Markdown support."""
    global app
    if app is None:
        logger.error("Agent graph is not initialized.")
        return

    user_id = update.effective_user.id
    user_text = update.message.text

    if not user_text:
        return

    config = {"configurable": {"thread_id": str(user_id)}}

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)

    sent_message = await update.message.reply_text(markdownify("Thinking..."), parse_mode=constants.ParseMode.MARKDOWN_V2)

    try:
        full_response = ""
        last_update_time = time.time()
        update_interval = 0.5 # to avoid rate limits

        async for chunk, metadata in app.astream(
            {"messages": [{"role": "user", "content": user_text}]},
            config=config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") == "agent":
                text = extract_text(chunk.content)
                if text:
                    full_response += text
                    
                    current_time = time.time()
                    if current_time - last_update_time > update_interval:
                        try:
                            formatted_response = markdownify(full_response + " ▌");
                            await sent_message.edit_text(formatted_response, parse_mode=constants.ParseMode.MARKDOWN_V2)
                            last_update_time = current_time
                        except Exception:
                            pass

        if full_response:
            formatted_response = markdownify(full_response)
            try:
                await sent_message.edit_text(formatted_response, parse_mode=constants.ParseMode.MARKDOWN_V2)
            except Exception as e:
                logger.warning(f"Markdown parsing failed for final message: {e}. Falling back to plain text.")
                await sent_message.edit_text(full_response)
        else:
            await sent_message.edit_text("I'm sorry, I couldn't generate a response.")

    except Exception as e:
        logger.error(f"Error processing message from user {user_id}: {e}")
        try:
            await sent_message.edit_text("An error occurred while processing your request. Please try again later.")
        except Exception:
            await update.message.reply_text("An error occurred while processing your request. Please try again later.")

async def main() -> None:
    """Starts the Telegram bot."""
    configure_logging()
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("clear", clear_command))

    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(message_handler)

    logger.info("Telegram bot is starting...")

    async for app_instance in build_graph_async():
        global app
        app = app_instance
        
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        
        logger.info("Bot is polling and ready for messages.")
        
        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down bot...")
            await application.stop()
            await application.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped.")
