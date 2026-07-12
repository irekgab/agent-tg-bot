# Agent Architecture and Documentation

This document describes the design, architecture, and usage of the LangGraph-based ReAct agent system implemented in this workspace.

---

## 1. System Overview

The system is a fully functional conversational agent powered by Google's Gemma models (specifically, configured for `gemma-4-31b-it` by default) via Google Generative AI APIs. It leverages **LangGraph** to build a robust, state-guided **ReAct (Reasoning and Action) loop**. 

The agent can interact with the external world using:
- **Custom Local Tools**: Strict file-system operations sandboxed to the project directory.
- **Built-in Global Tools**: Gemini's server-side Google Search execution.

The agent can be interacted with via:
- **CLI**: Interactive or single-prompt modes.
- **Telegram Bot**: A real-time messaging interface.

---

## 2. Directory Structure & Components

```
agent-tg-bot/
├── .data/             # Internal storage (history.db, checkpoints)
├── cli.py             # CLI Entrypoint (interactive & single-prompt modes)
├── config.py          # Centralized configuration loader (environment & .env)
├── graph.py           # LangGraph ReAct state machine definition
├── llm.py             # LLM client construction & factory function
├── logging_config.py  # Logger filtering to suppress noisy warnings
├── requirements.txt   # Python dependency list
├── streaming.py       # Custom token stream extractor (handling thinking/reasoning blocks)
├── telegram_bot.py    # Telegram bot implementation
└── tools.py           # Custom Python tool definitions (sandboxed file utilities)
```

---

## 3. LangGraph Flow (The ReAct Loop)

The agent operates in a classic cycle of execution managed by a `StateGraph`:

```
                 +---------+
                 |  START  |
                 +----+----+
                      |
                      v
               +--------------+
               |  agent_node  | <-------------------+
               +------+-------+                     |
                      |                             |
                      v                             |
            /~~~~~~~~~~~~~~~~~~~\                   |
           / Is tool call needed \                  |
           \      by LLM?        /                  |
            \~~~~~~~~~~~~~~~~~~~/                   |
               /             \                      |
         Yes  /               \ No                  |
             v                 v                    |
       +-----------+       +-------+                |
       | tool_node |       |  END  |                |
       +-----+-----+       +-------+                |
             |                                      |
             +--------------------------------------+
```

1. **START**: Input messages are loaded into the `AgentState`.
2. **agent_node**: The LLM is invoked with tools bound. It reviews the conversation and either:
   - Decides to execute a tool (returns tool call schema).
   - Generates a final conversational response (returns a text message).
3. **tools_condition (Conditional Edge)**:
   - If a tool call is requested, it routes execution to the `tools` node.
   - If no tool call is requested, the loop ends (`END`).
4. **tool_node**: Executes the requested tool(s) and appends results back to the conversation history.
5. **Loop**: The graph transitions back to `agent_node` to process tool execution results and formulate the next step.

---

## 4. Component Deep Dive

### `cli.py`
- Handles CLI arguments.
  - Interactive loop: Run `python cli.py` to start an ongoing multi-turn conversation.
  - Single prompt: Run `python cli.py "your query"` to execute a single turn and output the result.
- Calls `configure_logging()` to keep stdout clean.
- Builds the LangGraph application via `build_graph()` and runs the prompt using `stream_response`.

### `config.py`
- Handles dot-env resolution using `python-dotenv`.
- Loads `LLM_API_KEY` (raises `EnvironmentError` if missing).
- Loads `MODEL_NAME` (defaults to `"gemma-4-31b-it"`).
- Loads `TELEGRAM_BOT_TOKEN` (required for Telegram bot mode).

### `llm.py`
- Acts as a factory for `ChatGoogleGenerativeAI`.
- Standardizes temperature configurations (default `0.7`) and passes API keys and model parameters.

### `graph.py`
- Defines the `AgentState` schema, which wraps a list of messages.
- Sets up the `agent` node and `tools` node (`ToolNode(TOOLS)`).
- Binds tools to the LLM. It registers both the custom tools (defined in `tools.py`) and a special `GOOGLE_SEARCH_TOOL` object:
  ```python
  GOOGLE_SEARCH_TOOL = {"google_search": {}}
  ```
- Uses `tool_config={"include_server_side_tool_invocations": True}` to enable Google's native server-side search tool integration.
- **Memory Persistence**: Integrates `SqliteSaver` checkpointer to persist conversation state. The database is stored at `.data/history.db`, allowing the agent to remember previous turns across program restarts and supporting multiple independent conversation threads via `thread_id`.

### `streaming.py`
- Implements custom token-by-token stdout stream parsing.
- Handles standard textual responses and complex content blocks. Specifically, it parses list-based message structures and extracts only the `"type": "text"` blocks (ignoring `"thinking"` or auxiliary blocks) to prevent developer reasoning traces from cluttering user outputs.
- **State Configuration**: Now accepts a `config` dictionary (containing `thread_id`) to enable LangGraph to retrieve and update the correct persistent thread from the SQLite database.

### `logging_config.py`
- Suppresses two redundant warning loggers:
  1. `langchain_google_genai._function_utils`: Silences warnings about stripped JSON schema keys (such as `title`, `$defs`) that the Gemini schema API doesn't support but are natively emitted by Pydantic.
  2. `google_genai`: Silences the message warning that automatic function-calling (AFC) is disabled due to the presence of non-callable dictionary definitions (the `google_search` configuration map) alongside Python tool functions.

### `telegram_bot.py`
- Implements a Telegram bot interface using `python-telegram-bot`.
- **Per-thread memory**: the LangGraph `thread_id` is built from the Telegram `user_id` *plus* the forum `message_thread_id` (when the message is inside a forum topic), via `build_thread_key()`. This means the same user gets independent conversation history in each topic/thread they write in, instead of one history shared across all of them. `/clear` clears only the calling thread's history.
- **Rate-limit safe streaming**: `safe_edit` / `safe_send` wrap every Telegram API call with retry/backoff handling for `RetryAfter` (HTTP 429), `BadRequest` ("message is not modified", bad Markdown entities), and `TimedOut`. Live-edit frequency is throttled to `UPDATE_INTERVAL` (1s) to stay under Telegram's per-chat edit rate limit.
- **Live status during tool use**: while the agent is calling a tool (detected via `tool_call_chunks` on streamed `agent` node chunks, and again when the `tools` node produces a result), the in-progress message is force-updated to show the text generated so far plus a `⚙️ Using <tool>...` / `⚙️ Thinking...` status line, so the message never sits stale during a tool call.
- **Message splitting**: `process_agent_turn()` tracks how much of the running response has been "closed off" into earlier Telegram messages. Once the un-flushed remainder exceeds `SAFE_CHUNK` (3000 chars, safely under Telegram's 4096 hard limit even after Markdown escaping), it finalizes the current message and opens a new one to continue streaming into - so long responses are automatically split across multiple messages instead of erroring.
- **Proactive / scheduled messages**: `process_agent_turn()` is the shared core for both normal replies and bot-initiated follow-ups, used via:
  - `tools.schedule_message(delay_seconds, instruction)` - a tool the agent can call (e.g. when asked "remind me in 10 minutes") which registers a job via `application.job_queue`.
  - `schedule_followup()` - the `tools.SCHEDULE_CALLBACK` implementation; since tools run in a worker thread, it hands off to the bot's asyncio loop via `call_soon_threadsafe` before touching the `JobQueue`.
  - `on_scheduled_job()` - the job callback that fires after the delay, re-invokes the agent with the original instruction as context (on the same `thread_id`, so it has full conversation history), and sends the result to the chat as a new message.
  - Scheduled jobs live in the in-memory `JobQueue` and do **not** persist across bot restarts.
- Integrates with the existing agent graph and toolset.

### `tools.py`
- Defines Python function-based tools using `@tool` from `langchain_core.tools`.
- Implements strict **Project Sandboxing**:
  - `WORKSPACE_DIR` defaults to the current directory (`.`) or can be overridden via the `AGENT_WORKSPACE` environment variable.
  - Path safety is enforced by `_resolve_safe_path(path: str)`, which resolves absolute paths and raises a `ValueError` if any operation attempts to access files outside of `WORKSPACE_DIR`.
- **Custom Tools**:
  - `get_current_time()`: Returns the current UTC date and time.
  - `read_file(path)`: Safely reads the text content of a file within the project.
  - `write_file(path, content)`: Safely writes content to a file within the project (creating parent directories automatically).
  - `list_directory(path=".")`: Safely lists files and subdirectories inside the project directory.
  - `execute_command(command)`: Safely executes a terminal/shell command within the project directory and returns its output (stdout and stderr).
  - `make_web_request(url, method, headers, data, params)`: Makes an HTTP request to a specified URL.
  - `schedule_message(delay_seconds, instruction)`: Lets the agent schedule a follow-up for itself (e.g. "remind me in 10 minutes"). Reads the current chat context from the `current_chat_context` contextvar and delegates to `SCHEDULE_CALLBACK`, both of which the host app (`telegram_bot.py`) sets up - `tools.py` itself has no Telegram-specific code. See the `telegram_bot.py` section above for how this is wired up end to end.

---

## 5. Setup & Usage

### Setup
1. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   LLM_API_KEY=your_google_gemma_api_key
   MODEL_NAME=gemma-4-31b-it
   AGENT_WORKSPACE=.
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ```

### Running the Agent
- **CLI Mode**:
  - **Interactive**: `python cli.py`
  - **Single Prompt**: `python cli.py "your query"`

- **Telegram Bot Mode**:
  ```bash
  python telegram_bot.py
  ```

---

## 6. Development Guide

### Adding a New Custom Tool
To expand the agent's capabilities with a new Python-based tool:

1. **Define the tool in `tools.py`**:
   Write a Python function, decorate it with `@tool`, specify argument types, and provide a detailed docstring explaining *what* the tool does, its arguments, and its return values. This docstring is exposed directly to the LLM for function-calling.
   ```python
   @tool
   def multiply_numbers(a: float, b: float) -> float:
       """Multiply two numbers and return the result."""
       return a * b
   ```
2. **Incorporate Workspace Protection (if applicable)**:
   If your tool interacts with the filesystem, use the `_resolve_safe_path` helper to ensure the agent operates strictly inside the designated workspace.
3. **Register the tool**:
   Append the new tool function to the `TOOLS` list at the bottom of `tools.py`:
   ```python
   TOOLS = [get_current_time, read_file, write_file, list_directory, multiply_numbers]
   ```
   The `graph.py` file dynamically reads this list and binds all registered tools to the LLM model automatically.

### Customizing the Agent State or Logic
- **Modify the State**: Add fields to the `AgentState` TypedDict in `graph.py` (e.g., tracking a user ID, maintaining execution counters, or managing step limits).
- **Adjust System Prompts / Parameters**: You can modify `build_llm()` parameters in `llm.py` to change the temperature or load systemic instructions. If you need to bind a dedicated system prompt, you can prepend a system message block inside `agent_node` in `graph.py` before invoking the LLM model.
