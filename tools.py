"""Tool definitions available to the agent.

Add new tools here as plain functions decorated with @tool, then add
them to the TOOLS list. Keep each tool small and single-purpose.
"""
import re
import os
import subprocess
from datetime import datetime, timezone

from langchain_core.tools import tool


WORKSPACE_DIR = os.path.abspath(os.getenv("AGENT_WORKSPACE", "."))

GOOGLE_SEARCH_TOOL = {"google_search": {}}

@tool
def get_current_time() -> str:
    """Return the current UTC date and time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


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
    """Execute a shell command inside the workspace and return its stdout and stderr.

    Args:
        command: The shell command to execute in the terminal (e.g., 'python script.py', 'ls', 'pip install requirements.txt').

    Returns:
        The combined stdout and stderr output of the executed command, or an error message.
    """
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


TOOLS = [get_current_time, read_file, write_file, list_directory, execute_command]
