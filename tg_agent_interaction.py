import asyncio
import logging
import time
import tools
from llm import extract_text
import tg_bot_state
from tg_bot_utils import _get_thread_lock, safe_send, split_chunk, safe_edit

logger = logging.getLogger(__name__)

async def process_agent_turn(
    bot,
    chat_id: int,
    thread_key: str,
    user_content: str,
    message_thread_id=None,
    status_message=None,
) -> None:
    if tg_bot_state.app is None:
        logger.error("Agent graph is not initialized.")
        return

    lock = await _get_thread_lock(thread_key)
    async with lock:
        task = asyncio.current_task()
        tg_bot_state._active_turn_tasks[thread_key] = task
        try:
            await _run_agent_turn(bot, chat_id, thread_key, user_content, message_thread_id)
        finally:
            if tg_bot_state._active_turn_tasks.get(thread_key) is task:
                del tg_bot_state._active_turn_tasks[thread_key]


async def _run_agent_turn(
    bot,
    chat_id: int,
    thread_key: str,
    user_content: str,
    message_thread_id=None,
) -> None:
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
        delay = tg_bot_state.UPDATE_INTERVAL - (now - last_update_time)
        if delay > 0:
            await asyncio.sleep(delay)
        last_update_time = time.monotonic()


    async def flush(final: bool = False) -> None:
        nonlocal current_message, flushed_len, last_update_time, tool_status
        pending = full_response[flushed_len:]

        while len(pending) > tg_bot_state.SAFE_CHUNK:
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
        async for chunk, metadata in tg_bot_state.app.astream(
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

    except asyncio.CancelledError:
        tool_status = None
        full_response = (full_response.rstrip() + "\n\n" if full_response else "") + "_⏹ Stopped._"
        try:
            await flush(final=True)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Error processing message for thread {thread_key}: {e}")
        await safe_send(bot, chat_id, "An error occurred while processing your request. Please try again later.", message_thread_id=message_thread_id)
    finally:
        tools.current_chat_context.reset(token)
