import os
import subprocess
import contextvars
from datetime import datetime, timezone

from langchain_core.tools import tool

import workspace as ws

current_chat_context: contextvars.ContextVar = contextvars.ContextVar(
    "current_chat_context", default=None
)

SCHEDULE_CALLBACK = None
SEND_FILE_CALLBACK = None
MAX_SEND_FILE_BYTES = 50 * 1024 * 1024


def _current_thread_key() -> str:
    ctx = current_chat_context.get()
    if ctx is None:
        return "_default"
    return ctx["thread_key"]


def _resolve_safe_path(path: str) -> str:
    base = os.path.abspath(ws.workspace_dir(_current_thread_key()))
    full_path = os.path.abspath(os.path.join(base, path))
    if not (full_path == base or full_path.startswith(base + os.path.sep)):
        raise ValueError(f"Access denied: '{path}' is outside the allowed workspace.")
    return full_path


@tool
def get_current_time() -> str:
    """Returns the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@tool
def read_file(path: str) -> str:
    """Reads the content of a file from the workspace."""
    try:
        full_path = _resolve_safe_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Error: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file in the workspace."""
    try:
        full_path = _resolve_safe_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def list_directory(path: str = ".") -> str:
    """Lists the files and directories in a given path."""
    try:
        full_path = _resolve_safe_path(path)
        entries = os.listdir(full_path)
        if not entries:
            return "(empty directory)"
        return "\n".join(sorted(entries))
    except Exception as exc:
        return f"Error: {exc}"    


@tool
def execute_command(command: str) -> str:
    """Executes a shell command in the workspace."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=ws.workspace_dir(_current_thread_key()),
            timeout=120,
        )
        output = []
        if result.stdout:
            output.append(result.stdout)
        if result.stderr:
            output.append(result.stderr)
        if not output:
            return f"Command executed successfully with exit code {result.returncode} (no output)."
        return "".join(output)
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except Exception as exc:
        return f"Error executing command: {exc}"


@tool
def make_web_request(url: str, method: str = "GET", headers: dict = None, data: str = None, params: dict = None) -> str:
    """Performs a HTTP request."""
    import requests
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            data=data,
            params=params,
            timeout=15
        )
        return f"Status Code: {response.status_code}\n\nResponse:\n{response.text}"
    except Exception as exc:
        return f"Error making web request: {exc}"


@tool
def schedule_message(delay_seconds: int, instruction: str) -> str:
    """Schedules a follow-up message."""
    if delay_seconds is None or delay_seconds <= 0:
        return "Error: delay_seconds must be a positive number of seconds."

    ctx = current_chat_context.get()
    if ctx is None or SCHEDULE_CALLBACK is None:
        return "Error: scheduling a follow-up isn't available in this context."

    try:
        SCHEDULE_CALLBACK(
            ctx["chat_id"],
            ctx.get("message_thread_id"),
            ctx["thread_key"],
            delay_seconds,
            instruction,
        )
    except Exception as exc:
        return f"Error scheduling follow-up: {exc}"

    return f"Follow-up scheduled in {delay_seconds} second(s)."


@tool
def remember_about_chat(content: str) -> str:
    """Saves notes about the current chat."""
    try:
        ws.write_chat_notes(_current_thread_key(), content)
        return "Chat notes updated."
    except Exception as exc:
        return f"Error updating chat notes: {exc}"


@tool
def send_file_to_user(path: str, caption: str = "") -> str:
    """Sends a file from the workspace to the user."""
    ctx = current_chat_context.get()
    if ctx is None or SEND_FILE_CALLBACK is None:
        return "Error: sending files isn't available in this context."

    try:
        full_path = _resolve_safe_path(path)
    except Exception as exc:
        return f"Error: {exc}"

    if not os.path.isfile(full_path):
        return f"Error: '{path}' does not exist in the workspace."

    size = os.path.getsize(full_path)
    if size > MAX_SEND_FILE_BYTES:
        return (
            f"Error: '{path}' is {size / (1024 * 1024):.1f} MB, which "
            f"exceeds Telegram's {MAX_SEND_FILE_BYTES // (1024 * 1024)} MB upload limit."
        )

    try:
        success, error = SEND_FILE_CALLBACK(
            ctx["chat_id"], ctx.get("message_thread_id"), ctx["thread_key"], full_path, caption
        )
    except Exception as exc:
        return f"Error sending file: {exc}"

    if not success:
        return f"Error sending file: {error}"
    return f"Sent '{path}' to the user."


GOOGLE_SEARCH_TOOL = {"google_search": {}}

TOOLS = [
    get_current_time,
    read_file,
    write_file,
    list_directory,
    execute_command,
    make_web_request,
    schedule_message,
    remember_about_chat,
    send_file_to_user,
]
