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
    )


def build_llm(temperature: float = 0.7) -> Any:
    """Create a chat model instance configured from environment settings."""
    llm = build_raw_llm(temperature=temperature)

    # We use .bind() to inject custom HTTP retry options. This ensures that 
    # transient 500 errors are retried with a constant delay (exp_base=1.0) 
    # rather than an exponentially increasing one, and for a higher number 
    # of attempts (30).
    return llm.bind(
        http_options={
            "retry_options": {
                "attempts": 30,
                "exp_base": 1.0,
                "initial_delay": 1.0,
            }
        }
    )


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
