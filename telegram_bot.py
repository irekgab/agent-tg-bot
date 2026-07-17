import logging
from logging_config import configure_logging

import asyncio
import base64
import os
import sqlite3
import time
from telegram import Update, constants
from telegram.error import BadRequest, RetryAfter, TimedOut
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from telegramify_markdown import markdownify

import tools
import workspace
from config import TELEGRAM_BOT_TOKEN
from graph import build_graph_async
from llm import extract_text

logger = logging.getLogger(__name__)

app = None
application = None
MAIN_LOOP: asyncio.AbstractEventLoop = None

SAFE_CHUNK = 4095
UPDATE_INTERVAL = 1.0
MESSAGE_COMBINE_DELAY = 1.5
MAX_FILE_BYTES = 256 * 1024 * 1024

# Per-thread pending-message buffers, and the debounce task currently
# waiting to flush each one. Guarded by _pending_guard purely for
# safety/clarity - the section it protects never awaits, so it's already
# atomic under asyncio's cooperative scheduling, but a lock costs nothing
# here and keeps this robust if that ever changes.
_pending_messages: dict[str, list[str]] = {}
_pending_tasks: dict[str, asyncio.Task] = {}
_pending_guard = asyncio.Lock()

# Per-thread-key locks. concurrent_updates(True) (set on the Application
# below) lets PTB dispatch updates for different chats/threads fully in
# parallel instead of queuing behind whichever update happens to be
# in-flight - but two overlapping turns for the *same* thread_key (e.g. a
# live message arriving right as a scheduled follow-up fires) would then
# race on the same LangGraph checkpointer row. These locks keep each
# individual thread's turns sequential while leaving different threads
# fully concurrent.
_thread_locks: dict[str, asyncio.Lock] = {}
_thread_locks_guard = asyncio.Lock()


async def _get_thread_lock(thread_key: str) -> asyncio.Lock:
    async with _thread_locks_guard:
        lock = _thread_locks.get(thread_key)
        if lock is None:
            lock = asyncio.Lock()
            _thread_locks[thread_key] = lock
        return lock


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
    """Clears the conversation history and this chat's whole workspace"""
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

        lock = await _get_thread_lock(thread_key)
        async with lock:
            await asyncio.to_thread(clear_db)
            await asyncio.to_thread(workspace.delete_workspace, thread_key)

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
    """Run one turn of the agent for (chat_id, thread_key) and stream the reply into Telegram """
    global app
    if app is None:
        logger.error("Agent graph is not initialized.")
        return

    lock = await _get_thread_lock(thread_key)
    async with lock:
        await _run_agent_turn(bot, chat_id, thread_key, user_content, message_thread_id)


async def _run_agent_turn(
    bot,
    chat_id: int,
    thread_key: str,
    user_content: str,
    message_thread_id=None,
) -> None:
    """The actual turn logic, run while holding this thread_key's lock."""
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
    stage_boundary_pending = False


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
                if stage_boundary_pending and full_response:
                    full_response += "\n\n\n"
                stage_boundary_pending = False
                full_response += text
                tool_status = None
                last_shown_tool_status = None
                await flush()
            else:
                tool_status = "_⚙️ Thinking..._"
                stage_boundary_pending = True
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
    chat_id = update.effective_chat.id
    bot = context.bot

    async with _pending_guard:
        is_first_in_batch = thread_key not in _pending_messages
        _pending_messages.setdefault(thread_key, []).append(user_text)

        existing_task = _pending_tasks.get(thread_key)
        if existing_task and not existing_task.done():
            existing_task.cancel()
        _pending_tasks[thread_key] = asyncio.create_task(
            _flush_pending_after_delay(bot, chat_id, thread_key, message_thread_id)
        )

    if is_first_in_batch:
        # Let the user know we've seen something right away, since actual
        # processing is now delayed by MESSAGE_COMBINE_DELAY (plus however
        # long the agent itself takes).
        await safe_chat_action(bot, chat_id, constants.ChatAction.TYPING, message_thread_id=message_thread_id)


def _save_uploaded_file(thread_key: str, filename: str, raw: bytes) -> None:
    """Save a user-sent file into this chat's workspace/uploads"""
    directory = workspace.uploads_dir(thread_key)
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    with open(os.path.join(directory, candidate), "wb") as f:
        f.write(raw)


async def _download_as_content_blocks(
    bot, file_id: str, mime_type: str, text_note: str | None, thread_key: str, filename: str
):
    """Download a Telegram file, save it into this chat's workspace, and turn it into LangChain multimodal content blocks"""
    try:
        tg_file = await bot.get_file(file_id)
    except Exception as exc:
        logger.warning(f"Failed to fetch Telegram file {file_id}: {exc}")
        return None, "Sorry, I couldn't fetch that file from Telegram. Please try again."

    if tg_file.file_size and tg_file.file_size > MAX_FILE_BYTES:
        limit_mb = MAX_FILE_BYTES // (1024 * 1024)
        return None, f"That file is too large for me to process (the limit is {limit_mb} MB)."

    try:
        raw = await tg_file.download_as_bytearray()
    except Exception as exc:
        logger.warning(f"Failed to download Telegram file {file_id}: {exc}")
        return None, "Sorry, I couldn't download that file from Telegram. Please try again."

    raw_bytes = bytes(raw)
    try:
        await asyncio.to_thread(_save_uploaded_file, thread_key, filename, raw_bytes)
    except Exception as exc:
        logger.warning(f"Failed to save uploaded file for thread {thread_key}: {exc}")

    encoded = base64.b64encode(raw_bytes).decode("ascii")

    blocks = []
    if text_note:
        blocks.append({"type": "text", "text": text_note})
    if mime_type.startswith("image/"):
        blocks.append({"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded}"})
    else:
        blocks.append({"type": "media", "mime_type": mime_type, "data": encoded})
    return blocks, None


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming photos"""
    if not update.message.photo:
        return

    user_id = update.effective_user.id
    message_thread_id = get_message_thread_id(update)
    thread_key = build_thread_key(user_id, message_thread_id)

    photo = update.message.photo[-1]  # last entry = highest resolution
    filename = f"photo_{photo.file_unique_id}.jpg"
    blocks, error = await _download_as_content_blocks(
        context.bot, photo.file_id, "image/jpeg", update.message.caption,
        thread_key=thread_key, filename=filename,
    )
    if error:
        await update.message.reply_text(error)
        return

    await process_agent_turn(
        context.bot,
        chat_id=update.effective_chat.id,
        thread_key=thread_key,
        user_content=blocks,
        message_thread_id=message_thread_id,
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming documents/files: downloads the file and hands it to
    the agent (as an image or generic media block, depending on mime type)
    plus any caption, or a filename note if there's no caption."""
    document = update.message.document
    if document is None:
        return

    user_id = update.effective_user.id
    message_thread_id = get_message_thread_id(update)
    thread_key = build_thread_key(user_id, message_thread_id)

    mime_type = document.mime_type or "application/octet-stream"
    text_note = update.message.caption or (
        f"[Attached file: {document.file_name}]" if document.file_name else None
    )
    filename = document.file_name or f"file_{document.file_unique_id}"
    blocks, error = await _download_as_content_blocks(
        context.bot, document.file_id, mime_type, text_note,
        thread_key=thread_key, filename=filename,
    )
    if error:
        await update.message.reply_text(error)
        return

    await process_agent_turn(
        context.bot,
        chat_id=update.effective_chat.id,
        thread_key=thread_key,
        user_content=blocks,
        message_thread_id=message_thread_id,
    )


async def _flush_pending_after_delay(bot, chat_id, thread_key: str, message_thread_id) -> None:
    """Waits MESSAGE_COMBINE_DELAY; if not superseded by a newer message in
    the same thread arriving first (which cancels this task), combines and
    processes everything buffered for this thread as one turn."""
    try:
        await asyncio.sleep(MESSAGE_COMBINE_DELAY)
    except asyncio.CancelledError:
        return  # A newer message reset the timer; its own task will flush.

    async with _pending_guard:
        texts = _pending_messages.pop(thread_key, [])
        _pending_tasks.pop(thread_key, None)

    if not texts:
        return

    combined_text = "\n\n".join(texts)
    await process_agent_turn(
        bot,
        chat_id=chat_id,
        thread_key=thread_key,
        user_content=combined_text,
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

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()

    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

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
