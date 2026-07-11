"""Centralized configuration loader.

Loads the LLM API key and model settings from environment variables,
falling back to a local .env file if present. Every other module reads
config from here instead of touching os.environ directly.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in the working directory, if present


def _require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or in a .env file."
        )
    return value


LLM_API_KEY: str = _require_env("LLM_API_KEY")
MODEL_NAME: str = _require_env("MODEL_NAME", "gemma-4-31b-it")
TELEGRAM_BOT_TOKEN: str = _require_env("TELEGRAM_BOT_TOKEN")
