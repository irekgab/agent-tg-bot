"""LLM client construction.

Wraps the Gemini/Gemma chat model behind a single factory function so
the rest of the app never has to know which SDK or model is in use.
"""
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_API_KEY, MODEL_NAME


def build_raw_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Create a bare chat model instance, with none of build_llm()'s .bind()-ed extras.

    build_llm() wraps the model in a generic RunnableBinding (via .bind()),
    which doesn't forward model-specific methods like .with_structured_output().
    Use this factory instead whenever you need those methods - e.g. the
    planner/replanner in graph.py, which need structured (JSON) output rather
    than free-form tool-calling.
    """
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=LLM_API_KEY,
        temperature=temperature,
        max_retries=7
    )


def build_llm(temperature: float = 0.7) -> Any:
    """Create a chat model instance configured from environment settings."""
    return build_raw_llm(temperature=temperature)


def extract_text(content) -> str:
    """Pull just the user-facing text out of a message's content."""
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
    """Pull text and descriptions of multimodal content out of a message's content."""
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
