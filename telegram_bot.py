import logging
from logging_config import configure_logging

import asyncio
import sqlite3
import time
from telegram import Update, constants
from telegram.error import BadRequest, RetryAfter, TimedOut
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from telegramify_markdown import markdownify

import tools
from config import TELEGRAM_BOT_TOKEN
from graph import build_graph_async
from llm import extract_text

logger = logging.getLogger(__name__)

app = None
application = None
MAIN_LOOP: asyncio.AbstractEventLoop = None

SAFE_CHUNK = 4095

UPDATE_INTERVAL = 1.0


def build_thread_key(user_id: int, message_thread_id: int | None) -> str:
    """Per-user, per-topic conversation identity.

    Using only the Telegram user_id (the old behaviour) means every forum
    topic / thread the same user writes in shares one conversation history.
    Folding the message_thread_id in gives each topic its own independent
    history while still keeping different users fully separate.
    """
    if message_thread_id:
        return f"{user_id}:{message_thread_id}"
    return str(user_id)


def get_message_thread_id(update: Update):
    message = update.message
    if message is not None and getattr(message, "is_topic_message", False):
        return message.message_thread_id
    return None


def split_chunk(text: str, limit: int = SAFE_CHUNK):
    """Split text into (chunk, remainder), preferring a newline boundary."""
    if len(text) <= limit:
        return text, ""
    split_at = text.rfind("\n", 0, limit)
    if split_at == -1 or split_at < limit * 0.4:
        split_at = limit
    return text[:split_at], text[split_at:]


async def safe_edit(message, raw_text: str, with_cursor: bool = False):
    """Edit a message, transparently handling Telegram rate limits (429s)
    and falling back to plain text if Markdown entity parsing fails."""
    display = raw_text + (" ▌" if with_cursor else "")
    try:
        formatted = markdownify(display)
    except Exception:
        formatted = None

    for _ in range(5):
        try:
            if formatted is not None:
                await message.edit_text(formatted, parse_mode=constants.ParseMode.MARKDOWN_V2)
            else:
                await message.edit_text(display)
            return message
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 0.5)
        except BadRequest as e:
            text = str(e).lower()
            if "message is not modified" in text:
                return message
            if formatted is not None:
                formatted = None
                continue
            logger.warning(f"Failed to edit message: {e}")
            return message
        except TimedOut:
            await asyncio.sleep(1)
    return message


async def safe_send(bot, chat_id, raw_text: str, message_thread_id=None, with_cursor: bool = False):
    """Send a new message with the same rate-limit/Markdown-fallback handling as safe_edit."""
    display = raw_text + (" ▌" if with_cursor else "")
    try:
        formatted = markdownify(display)
    except Exception:
        formatted = None

    for _ in range(5):
        try:
            if formatted is not None:
                return await bot.send_message(
                    chat_id=chat_id,
                    text=formatted,
                    parse_mode=constants.ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
            return await bot.send_message(chat_id=chat_id, text=display, message_thread_id=message_thread_id)
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 0.5)
        except BadRequest as e:
            if formatted is not None:
                formatted = None
                continue
            logger.warning(f"Failed to send message: {e}")
            raise
        except TimedOut:
            await asyncio.sleep(1)
    return await bot.send_message(chat_id=chat_id, text=raw_text, message_thread_id=message_thread_id)


async def safe_chat_action(bot, chat_id, action, message_thread_id=None):
    """Perform a chat action, handling rate limits and timeouts."""
    for _ in range(5):
        try:
            return await bot.send_chat_action(
                chat_id=chat_id, action=action, message_thread_id=message_thread_id
            )
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 0.5)
        except TimedOut:
            await asyncio.sleep(1)
        except BadRequest as e:
            logger.warning(f"Failed to send chat action: {e}")
            return None
    return None


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history for the current user/thread."""
    user_id = update.effective_user.id
    thread_key = build_thread_key(user_id, get_message_thread_id(update))

    try:
        def clear_db():
            conn = sqlite3.connect(".data/history.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_key,))
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_key,))
            conn.commit()
            conn.close()

        await asyncio.to_thread(clear_db)

        await update.message.reply_text(
            markdownify("Your conversation history has been cleared! You can start a new chat now."),
            parse_mode=constants.ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        logger.error(f"Error clearing history for thread {thread_key}: {e}")
        await update.message.reply_text("Failed to clear conversation history. Please try again later.")


async def process_agent_turn(
    bot,
    chat_id: int,
    thread_key: str,
    user_content: str,
    message_thread_id=None,
    status_message=None,
) -> None:
    """Run one turn of the agent for (chat_id, thread_key) and stream the
    reply into Telegram, splitting across messages if the reply grows past
    Telegram's length limit and showing live status while tools run.
    """
    global app
    if app is None:
        logger.error("Agent graph is not initialized.")
        return

    config = {"configurable": {"thread_id": thread_key}}
    token = tools.current_chat_context.set(
        {"chat_id": chat_id, "message_thread_id": message_thread_id, "thread_key": thread_key}
    )

    current_message = None
    full_response = ""
    flushed_len = 0
    last_update_time = 0.0
    tool_status = None
    last_shown_tool_status = None


    async def wait_interval():
        nonlocal last_update_time
        now = time.monotonic()
        delay = UPDATE_INTERVAL - (now - last_update_time)
        if delay > 0:
            await asyncio.sleep(delay)
        last_update_time = time.monotonic()


    async def flush(final: bool = False) -> None:
        nonlocal current_message, flushed_len, last_update_time, tool_status
        pending = full_response[flushed_len:]

        while len(pending) > SAFE_CHUNK:
            chunk, pending = split_chunk(pending)

            if current_message is None:
                current_message = await safe_send(bot, chat_id, chunk, message_thread_id=message_thread_id)
            else:
                await wait_interval()
                await safe_edit(current_message, chunk)

            flushed_len += len(chunk)
            current_message = None
        
        body = pending
        if tool_status and not final:
            body = (body + "\n\n" if body else "") + tool_status

        with_cursor = (not final and not tool_status and flushed_len + len(pending) == len(full_response))

        if current_message is None:
            if body is not None:
                current_message = await safe_send(bot, chat_id, body, message_thread_id=message_thread_id, with_cursor=with_cursor)
        else:
            await wait_interval()
            await safe_edit(current_message, body, with_cursor=with_cursor)


    try:
        async for chunk, metadata in app.astream(
            {"messages": [{"role": "user", "content": user_content}]},
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node")
            
            text = extract_text(chunk.content)
            if node == "agent" and text:
                full_response += text
                tool_status = None
                last_shown_tool_status = None
                await flush()
            else:
                tool_status = "_⚙️ Thinking..._"
                if tool_status != last_shown_tool_status:
                    await flush()
                    last_shown_tool_status = tool_status

        tool_status = None
        await flush(final=True)

    except Exception as e:
        logger.error(f"Error processing message for thread {thread_key}: {e}")
        await safe_send(bot, chat_id, "An error occurred while processing your request. Please try again later.", message_thread_id=message_thread_id)
    finally:
        tools.current_chat_context.reset(token)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages from Telegram users with streaming responses and Markdown support."""
    user_id = update.effective_user.id
    user_text = update.message.text
    if not user_text:
        return

    message_thread_id = get_message_thread_id(update)
    thread_key = build_thread_key(user_id, message_thread_id)

    await process_agent_turn(
        context.bot,
        chat_id=update.effective_chat.id,
        thread_key=thread_key,
        user_content=user_text,
        message_thread_id=message_thread_id,
    )


def schedule_followup(chat_id, message_thread_id, thread_key, delay_seconds, instruction) -> None:
    """Registered as tools.SCHEDULE_CALLBACK. Called synchronously from a
    tool-execution worker thread, so it hands off to the bot's event loop
    via call_soon_threadsafe rather than touching the JobQueue directly."""
    def _do_schedule():
        if application is None or application.job_queue is None:
            logger.error("Cannot schedule follow-up: job queue is not available.")
            return
        application.job_queue.run_once(
            on_scheduled_job,
            when=delay_seconds,
            data={
                "chat_id": chat_id,
                "message_thread_id": message_thread_id,
                "thread_key": thread_key,
                "instruction": instruction,
            },
            name=f"followup:{thread_key}:{time.time()}",
        )

    if MAIN_LOOP is None:
        logger.error("Cannot schedule follow-up: bot event loop is not running.")
        return
    MAIN_LOOP.call_soon_threadsafe(_do_schedule)


async def on_scheduled_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fires when a scheduled follow-up's delay has elapsed. Re-invokes the
    agent (with the original instruction as context) and sends whatever it
    produces to the user as a fresh, bot-initiated message."""
    data = context.job.data
    instruction_text = (
        "[Automated trigger: earlier in this conversation you scheduled a follow-up for "
        f"yourself for this exact moment. Your instruction to yourself was: \"{data['instruction']}\". "
        "Send the appropriate message to the user now.]"
    )
    await process_agent_turn(
        context.bot,
        chat_id=data["chat_id"],
        thread_key=data["thread_key"],
        user_content=instruction_text,
        message_thread_id=data.get("message_thread_id"),
    )


async def main() -> None:
    """Starts the Telegram bot."""
    global application, MAIN_LOOP
    configure_logging()
    MAIN_LOOP = asyncio.get_running_loop()
    tools.SCHEDULE_CALLBACK = schedule_followup

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

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
