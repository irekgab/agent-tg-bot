"""Per-chat workspace layout.

Shared by tools.py (file tools operate inside a chat's own workspace) and
graph.py (injects the global instructions + this chat's notes as context).
Centralizing the layout here means both sides agree on where things live
without needing to import each other for it.

Disk layout:

    {WORKSPACE_ROOT}/{sanitized_thread_key}/
        notes.md        <- agent-maintained memory about this chat/user
        uploads/         <- files the user has sent
        (anything else the agent creates via write_file)

    {GLOBAL_INSTRUCTIONS_PATH}
        One file, hand-edited by the operator, applied to every chat.
"""
import os
import re
from config import AGENT_WORKSPACE, AGENT_INSTRUCTIONS_FILE

WORKSPACE_ROOT = AGENT_WORKSPACE
GLOBAL_INSTRUCTIONS_PATH = AGENT_INSTRUCTIONS_FILE

NOTES_FILENAME = "notes.md"
UPLOADS_DIRNAME = "uploads"

# thread_key looks like "<user_id>" or "<user_id>:<message_thread_id>" (see
# telegram_bot.build_thread_key) - both always digits/colon, but sanitize
# defensively so it's always a safe single path component regardless.
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def sanitize_thread_key(thread_key: str) -> str:
    """Turn a thread_key into a safe single directory-name component."""
    return _SANITIZE_RE.sub("_", thread_key)


def workspace_dir(thread_key: str) -> str:
    """This chat's workspace directory, creating it if it doesn't exist yet."""
    path = os.path.join(WORKSPACE_ROOT, sanitize_thread_key(thread_key))
    os.makedirs(path, exist_ok=True)
    return path


def uploads_dir(thread_key: str) -> str:
    """Where files the user sends in this chat get saved."""
    path = os.path.join(workspace_dir(thread_key), UPLOADS_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def notes_path(thread_key: str) -> str:
    return os.path.join(workspace_dir(thread_key), NOTES_FILENAME)


def load_chat_notes(thread_key: str) -> str:
    """Read this chat's notes.md. Returns '' if it doesn't exist yet."""
    try:
        with open(notes_path(thread_key), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def write_chat_notes(thread_key: str, content: str) -> None:
    """Overwrite this chat's notes.md with new content."""
    with open(notes_path(thread_key), "w", encoding="utf-8") as f:
        f.write(content)


def delete_workspace(thread_key: str) -> None:
    """Remove this chat's entire workspace directory - notes.md, uploads/,
    and anything the agent wrote via write_file - e.g. when /clear wipes
    the conversation. The directory is recreated lazily (via workspace_dir)
    the next time this chat needs it."""
    import shutil
    path = os.path.join(WORKSPACE_ROOT, sanitize_thread_key(thread_key))
    shutil.rmtree(path, ignore_errors=True)


def load_global_instructions() -> str:
    """Read the single, hand-edited, applies-to-every-chat instructions
    file. Re-read fresh every call (not cached) so edits to it take effect
    without restarting the bot."""
    try:
        with open(GLOBAL_INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""
