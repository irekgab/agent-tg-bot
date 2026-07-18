import contextlib
import os
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler

from config import LLM_API_KEY, MODEL_NAME
import tools
import workspace as ws


class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        thread_key = tools._current_thread_key()
        log_file = os.path.join(ws.workspace_dir(thread_key), "llm_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- NEW TURN ---\nPrompt: {prompts[0]}\n")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        thread_key = tools._current_thread_key()
        log_file = os.path.join(ws.workspace_dir(thread_key), "llm_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Response: {response.generations[0][0].text}\n")


def build_raw_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=LLM_API_KEY,
        temperature=temperature,
        max_retries=7
    )


def build_llm(temperature: float = 0.7) -> Any:
    return build_raw_llm(temperature=temperature)


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def extract_planner_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    parts.append("[Attached image]")
                elif block.get("type") == "media":
                    parts.append(f"[Attached file: {block.get('mime_type', 'unknown')}]")
        return "".join(parts)
    return ""
