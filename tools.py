"""Tool definitions available to the agent.

Add new tools here as plain functions decorated with @tool, then add
them to the TOOLS list. Keep each tool small and single-purpose.
"""
import re
import os
import subprocess
import contextvars
from datetime import datetime, timezone

from langchain_core.tools import tool


WORKSPACE_DIR = os.path.abspath(os.getenv("AGENT_WORKSPACE", "."))

GOOGLE_SEARCH_TOOL = {"google_search": {}}

@tool
def get_current_time() -> str:
    """Return the current UTC date and time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Proactive / scheduled messaging support.
#
# `schedule_message` lets the agent arrange to "speak first" later, e.g. when
# a user asks to be reminded or followed up with after some delay. The tool
# itself has no idea it's running inside a Telegram bot - it just reads the
# current chat context from a contextvar (set by the caller, e.g.
# telegram_bot.py, right before invoking the graph) and delegates the actual
# scheduling to a callback that the host application registers at startup.
# This keeps tools.py free of any Telegram-specific imports.
# ---------------------------------------------------------------------------

# Set by the host app (e.g. telegram_bot.py) via ContextVar.set() right
# before invoking the graph, so tools running mid-turn know which chat/
# thread to schedule a follow-up for. LangChain's executor-based tool
# invocation (`run_in_executor`) copies the current context into the worker
# thread, so this propagates correctly even though tools run synchronously.
current_chat_context: contextvars.ContextVar = contextvars.ContextVar(
    "current_chat_context", default=None
)

# Set by the host app at startup: a callable with signature
# (chat_id, message_thread_id, thread_key, delay_seconds, instruction) -> None
# that arranges for the agent to be invoked again after delay_seconds with
# the given instruction as context, and for the result to be sent back to
# the same chat/thread.
SCHEDULE_CALLBACK = None


def _resolve_safe_path(path: str) -> str:
    """Resolve a path and ensure it stays inside WORKSPACE_DIR."""
    full_path = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not full_path.startswith(WORKSPACE_DIR):
        raise ValueError(f"Access denied: '{path}' is outside the allowed workspace.")
    return full_path


@tool
def read_file(path: str) -> str:
    """Read and return the text contents of a file inside the workspace."""
    try:
        full_path = _resolve_safe_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Error: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file inside the workspace, creating it if needed."""
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
    """List files and subdirectories at a given path inside the workspace."""
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
    """Execute a shell command inside the workspace and return its stdout and stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=WORKSPACE_DIR,
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
    """Make an HTTP request to a specified URL."""
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
    """Schedule a follow-up message to be sent to this same chat later, without waiting for the user to write again."""
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


TOOLS = [
    get_current_time,
    read_file,
    write_file,
    list_directory,
    execute_command,
    make_web_request,
    schedule_message,
]
