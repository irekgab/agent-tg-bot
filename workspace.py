import os
import re
from config import AGENT_WORKSPACE

WORKSPACE_ROOT = AGENT_WORKSPACE

NOTES_FILENAME = "notes.md"
UPLOADS_DIRNAME = "uploads"

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def sanitize_thread_key(thread_key: str) -> str:
    return _SANITIZE_RE.sub("_", thread_key)


def workspace_dir(thread_key: str) -> str:
    path = os.path.join(WORKSPACE_ROOT, sanitize_thread_key(thread_key))
    os.makedirs(path, exist_ok=True)
    return path


def uploads_dir(thread_key: str) -> str:
    path = os.path.join(workspace_dir(thread_key), UPLOADS_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def notes_path(thread_key: str) -> str:
    return os.path.join(workspace_dir(thread_key), NOTES_FILENAME)


def load_chat_notes(thread_key: str) -> str:
    try:
        with open(notes_path(thread_key), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def write_chat_notes(thread_key: str, content: str) -> None:
    with open(notes_path(thread_key), "w", encoding="utf-8") as f:
        f.write(content)


def delete_workspace(thread_key: str) -> None:
    import shutil
    path = os.path.join(WORKSPACE_ROOT, sanitize_thread_key(thread_key))
    shutil.rmtree(path, ignore_errors=True)
