import asyncio
import logging
import mimetypes
from telegram import Update, constants
from telegram.ext import ContextTypes
from telegramify_markdown import markdownify
import tg_bot_state
from tg_bot_utils import build_thread_key, get_message_thread_id, _get_thread_lock
import sqlite3
import workspace
from tg_agent_interaction import process_agent_turn
from tg_file_manager import _handle_file_upload

logger = logging.getLogger(__name__)

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    message_thread_id = get_message_thread_id(update)
    thread_key = build_thread_key(user_id, message_thread_id)

    async with tg_bot_state._pending_guard:
        pending_task = tg_bot_state._pending_tasks.pop(thread_key, None)
        had_pending_text = bool(tg_bot_state._pending_messages.pop(thread_key, None))
    if pending_task and not pending_task.done():
        pending_task.cancel()

    active_task = tg_bot_state._active_turn_tasks.get(thread_key)
    if active_task and not active_task.done():
        active_task.cancel()
        await update.message.reply_text("Okay, I won't respond to that.")
    elif had_pending_text:
        await update.message.reply_text("Okay, I won't respond to that.")
    else:
        await update.message.reply_text("Nothing is currently running.")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text
    if not user_text:
        return

    message_thread_id = get_message_thread_id(update)
    thread_key = build_thread_key(user_id, message_thread_id)
    chat_id = update.effective_chat.id
    bot = context.bot

    async with tg_bot_state._pending_guard:
        tg_bot_state._pending_messages.setdefault(thread_key, []).append(user_text)

        existing_task = tg_bot_state._pending_tasks.get(thread_key)
        if existing_task and not existing_task.done():
            existing_task.cancel()
        tg_bot_state._pending_tasks[thread_key] = asyncio.create_task(
            _flush_pending_after_delay(bot, chat_id, thread_key, message_thread_id)
        )


async def _flush_pending_after_delay(bot, chat_id, thread_key: str, message_thread_id) -> None:
    try:
        await asyncio.sleep(tg_bot_state.MESSAGE_COMBINE_DELAY)
    except asyncio.CancelledError:
        return 

    async with tg_bot_state._pending_guard:
        texts = tg_bot_state._pending_messages.pop(thread_key, [])
        tg_bot_state._pending_tasks.pop(thread_key, None)

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


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message.photo:
        return
    photo = update.message.photo[-1]
    await _handle_file_upload(
        update, context, photo.file_id, "image/jpeg", 
        f"photo_{photo.file_unique_id}.jpg", update.message.caption
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.message.document
    if document is None:
        return

    text_note = update.message.caption or (
        f"[Attached file: {document.file_name}]" if document.file_name else None
    )
    guessed_mime = mimetypes.guess_type(document.file_name or "")[0]
    await _handle_file_upload(
        update, context, document.file_id, document.mime_type or guessed_mime or "application/octet-stream",
        document.file_name or f"file_{document.file_unique_id}", text_note
    )
