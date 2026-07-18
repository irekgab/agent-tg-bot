import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or in a .env file."
        )
    return value


LLM_API_KEY: str = _require_env("LLM_API_KEY")
MODEL_NAME: str = _require_env("MODEL_NAME")
TELEGRAM_BOT_TOKEN: str = _require_env("TELEGRAM_BOT_TOKEN")
AGENT_WORKSPACE: str = _require_env("AGENT_WORKSPACE")
MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS"))
