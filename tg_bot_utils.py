import asyncio
import logging
from telegram import Update, constants
from telegram.error import BadRequest, RetryAfter, TimedOut
from telegramify_markdown import markdownify
import tg_bot_state

logger = logging.getLogger(__name__)

async def _get_thread_lock(thread_key: str) -> asyncio.Lock:
    async with tg_bot_state._thread_locks_guard:
        lock = tg_bot_state._thread_locks.get(thread_key)
        if lock is None:
            lock = asyncio.Lock()
            tg_bot_state._thread_locks[thread_key] = lock
        return lock

def build_thread_key(user_id: int, message_thread_id: int | None) -> str:
    if message_thread_id:
        return f"{user_id}:{message_thread_id}"
    return str(user_id)

def get_message_thread_id(update: Update):
    message = update.message
    if message is not None and getattr(message, "is_topic_message", False):
        return message.message_thread_id
    return None

def split_chunk(text: str, limit: int = tg_bot_state.SAFE_CHUNK):
    if len(text) <= limit:
        return text, ""
    split_at = text.rfind("\n", 0, limit)
    if split_at == -1 or split_at < limit * 0.4:
        split_at = limit
    return text[:split_at], text[split_at:]

async def safe_edit(message, raw_text: str, with_cursor: bool = False):
    display = raw_text + (" ▌" if with_cursor else "")
    if not display.strip():
        return message
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
