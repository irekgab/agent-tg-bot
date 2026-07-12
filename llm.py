"""LLM client construction.

Wraps the Gemini/Gemma chat model behind a single factory function so
the rest of the app never has to know which SDK or model is in use.
"""
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_API_KEY, MODEL_NAME


def build_llm(temperature: float = 0.7) -> Any:
    """Create a chat model instance configured from environment settings."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=LLM_API_KEY,
        temperature=temperature,
    )

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
