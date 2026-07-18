import os
import base64
import asyncio
import logging
from telegram import InputFile, constants, Update
from telegram.ext import ContextTypes
from telegram.error import BadRequest, RetryAfter, TimedOut
from telegramify_markdown import markdownify
import tg_bot_state
import workspace
from tg_bot_utils import build_thread_key, get_message_thread_id
from tg_agent_interaction import process_agent_turn

logger = logging.getLogger(__name__)

def _save_uploaded_file(thread_key: str, filename: str, raw: bytes) -> None:
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
    try:
        tg_file = await bot.get_file(file_id)
    except Exception as exc:
        logger.warning(f"Failed to fetch Telegram file {file_id}: {exc}")
        return None, "Sorry, I couldn't fetch that file from Telegram. Please try again."

    if tg_file.file_size and tg_file.file_size > tg_bot_state.MAX_FILE_BYTES:
        limit_mb = tg_bot_state.MAX_FILE_BYTES // (1024 * 1024)
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


async def _handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id, mime_type, filename, text_note) -> None:
    user_id = update.effective_user.id
    message_thread_id = get_message_thread_id(update)
    thread_key = build_thread_key(user_id, message_thread_id)

    blocks, error = await _download_as_content_blocks(
        context.bot, file_id, mime_type, text_note,
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


async def _send_file(chat_id, message_thread_id, path: str, caption: str):
    filename = os.path.basename(path)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    is_image = ext in tg_bot_state.IMAGE_EXTENSIONS

    try:
        formatted_caption = markdownify(caption) if caption else None
        parse_mode = constants.ParseMode.MARKDOWN_V2
    except Exception:
        formatted_caption = caption or None
        parse_mode = None

    for _ in range(5):
        try:
            with open(path, "rb") as f:
                file_obj = InputFile(f, filename=filename)
                if is_image:
                    await tg_bot_state.application.bot.send_photo(
                        chat_id=chat_id,
                        photo=file_obj,
                        caption=formatted_caption,
                        parse_mode=parse_mode,
                        message_thread_id=message_thread_id,
                    )
                else:
                    await tg_bot_state.application.bot.send_document(
                        chat_id=chat_id,
                        document=file_obj,
                        caption=formatted_caption,
                        parse_mode=parse_mode,
                        message_thread_id=message_thread_id,
                    )
            return True, None
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 0.5)
        except BadRequest as e:
            if parse_mode is not None:
                formatted_caption, parse_mode = (caption or None), None
                continue
            logger.warning(f"Failed to send file {path}: {e}")
            return False, str(e)
        except TimedOut:
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to send file {path}: {e}")
            return False, str(e)
    return False, "Failed to send file after multiple retries."


def send_file_callback(chat_id, message_thread_id, thread_key, path, caption):
    if tg_bot_state.application is None or tg_bot_state.MAIN_LOOP is None:
        return False, "Bot is not running."
    future = asyncio.run_coroutine_threadsafe(
        _send_file(chat_id, message_thread_id, path, caption), tg_bot_state.MAIN_LOOP
    )
    try:
        return future.result(timeout=60)
    except Exception as exc:
        return False, str(exc)
