"""LLM client construction.

Wraps the Gemini/Gemma chat model behind a single factory function so
the rest of the app never has to know which SDK or model is in use.
"""
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_API_KEY, MODEL_NAME


def build_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Create a chat model instance configured from environment settings."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=LLM_API_KEY,
        temperature=temperature,
    )
