import asyncio
from telegram.ext import Application

SAFE_CHUNK = 4095
UPDATE_INTERVAL = 1.0
MESSAGE_COMBINE_DELAY = 0.5
MAX_FILE_BYTES = 20 * 1024 * 1024
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}

_pending_messages: dict[str, list[str]] = {}
_pending_tasks: dict[str, asyncio.Task] = {}
_pending_guard = asyncio.Lock()

_thread_locks: dict[str, asyncio.Lock] = {}
_thread_locks_guard = asyncio.Lock()
_active_turn_tasks: dict[str, asyncio.Task] = {}

app = None
application: Application = None
MAIN_LOOP: asyncio.AbstractEventLoop = None
