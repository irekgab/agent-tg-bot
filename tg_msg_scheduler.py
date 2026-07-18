import logging
import time
from telegram.ext import ContextTypes
import tg_bot_state
from tg_agent_interaction import process_agent_turn

logger = logging.getLogger(__name__)

def schedule_followup(chat_id, message_thread_id, thread_key, delay_seconds, instruction) -> None:
    def _do_schedule():
        if tg_bot_state.application is None or tg_bot_state.application.job_queue is None:
            logger.error("Cannot schedule follow-up: job queue is not available.")
            return
        tg_bot_state.application.job_queue.run_once(
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

    if tg_bot_state.MAIN_LOOP is None:
        logger.error("Cannot schedule follow-up: bot event loop is not running.")
        return
    tg_bot_state.MAIN_LOOP.call_soon_threadsafe(_do_schedule)


async def on_scheduled_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data
    await process_agent_turn(
        context.bot,
        chat_id=data["chat_id"],
        thread_key=data["thread_key"],
        user_content=data["instruction"],
        message_thread_id=data.get("message_thread_id"),
    )
