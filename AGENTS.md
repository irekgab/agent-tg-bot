# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this project is

A Telegram bot backed by a LangGraph **plan → execute → review** agent loop,
powered by Google Gemini via `langchain-google-genai`. Each Telegram chat
(or forum topic) gets its own persistent conversation thread, its own
sandboxed on-disk workspace, and its own SQLite-backed checkpoint history.

The agent can browse the web (Gemini's built-in `google_search` tool), read
and write files in its sandboxed workspace, run shell commands there, make
HTTP requests, remember free-form notes about a chat, schedule follow-up
messages, and send files back to the user.

## Runtime architecture

```
tg_bot.py (entrypoint)
  └─ builds the LangGraph app (graph.py) and the python-telegram-bot Application
  └─ registers handlers from tg_bot_handlers.py
  └─ polls Telegram; on shutdown stops the Application cleanly

tg_bot_handlers.py            tg_file_manager.py
  /clear, /stop, text, photo,   downloads Telegram files, converts them to
  document handlers             LLM content blocks (image/media/text-note),
       │                        sends files back to the user
       ▼
tg_agent_interaction.py  ──── streams graph events, incrementally edits
  process_agent_turn()         one Telegram message via safe_send/safe_edit,
                                shows a transient "Processing..." status line
       │
       ▼
graph.py (LangGraph StateGraph)
  planner → executor ⇄ tools → reviewer → (planner | executor | END)
       │
       ▼
llm.py            tools.py            workspace.py
  Gemini client,    agent tools,        per-thread directories,
  logging callback  contextvar-scoped   chat notes, sanitized paths
                     to the current chat
```

Supporting modules: `config.py` (env vars), `logging_config.py` (root logger
setup), `tg_bot_state.py` (shared in-memory/global state), `tg_bot_utils.py`
(Telegram formatting/sending/locking helpers), `tg_msg_scheduler.py`
(follow-up message scheduling via the PTB job queue).

## File map

| File | Responsibility |
|---|---|
| `config.py` | Loads and validates required env vars (`_require_env` raises if missing/empty). |
| `graph.py` | Defines the LangGraph state machine: `AgentState`, `Plan`/`Review` structured-output schemas, planner/executor/reviewer nodes, routing functions, and `build_graph_async()` which yields a compiled graph with an `AsyncSqliteSaver` checkpointer at `.data/history.db`. |
| `llm.py` | `build_llm()` constructs a `ChatGoogleGenerativeAI`; `LoggingCallbackHandler` appends prompts/responses to `<workspace>/llm_log.txt`; `extract_text`/`extract_planner_text` normalize LangChain message content (str or list-of-blocks) into plain text. |
| `logging_config.py` | Configures the root logger (console + midnight-rotating file handler at `.data/agent.log`), quiets noisy third-party loggers, filters `200 OK` httpx lines. |
| `tools.py` | LangChain `@tool` functions exposed to the executor node: `get_current_time`, `read_file`, `write_file`, `list_directory`, `execute_command`, `make_web_request`, `schedule_message`, `remember_about_chat`, `send_file_to_user`. Uses `current_chat_context` (a `contextvars.ContextVar`) to know which chat/thread a tool call belongs to, and `_resolve_safe_path` to keep filesystem access inside the thread's workspace. `SCHEDULE_CALLBACK`/`SEND_FILE_CALLBACK` are injected by `tg_bot.py` at startup to avoid circular imports with the Telegram layer. |
| `workspace.py` | Per-thread workspace directories under `AGENT_WORKSPACE`, sanitizing thread keys into safe directory names; chat notes read/write; workspace deletion for `/clear`. |
| `tg_bot.py` | Process entrypoint. Wires `tools.SCHEDULE_CALLBACK`/`SEND_FILE_CALLBACK`, builds the PTB `Application`, registers handlers, and runs `build_graph_async()` in a loop so a fresh checkpointer/app is available. |
| `tg_bot_handlers.py` | Telegram update handlers: `/clear` (wipes checkpoint rows + workspace dir), `/stop` (cancels pending/active turns), `handle_message` (debounces rapid consecutive messages via `MESSAGE_COMBINE_DELAY` before starting a turn), `handle_photo`/`handle_document` (delegate to `tg_file_manager`). |
| `tg_bot_state.py` | Module-level shared state: tunables (`SAFE_CHUNK`, `UPDATE_INTERVAL`, `MESSAGE_COMBINE_DELAY`, `MAX_FILE_BYTES`, `IMAGE_EXTENSIONS`), pending-message buffers/tasks, per-thread asyncio locks, active-turn task registry, and references to the compiled graph app / PTB application / event loop. |
| `tg_bot_utils.py` | `build_thread_key`/`get_message_thread_id` (forum-topic aware), Markdown-safe chunking/sending/editing of Telegram messages with retry handling for rate limits and "message not modified" errors. |
| `tg_file_manager.py` | Downloads a Telegram file, saves a copy into the thread's `uploads/` dir, and converts it into LLM content blocks (`image_url` for images, `media` for other LLM-supported MIME types, or a plain text note pointing at the saved path otherwise). Also implements `send_file_to_user`'s Telegram-side delivery (`send_file_callback`, thread-safe via `run_coroutine_threadsafe`). |
| `tg_msg_scheduler.py` | Implements `schedule_message`'s Telegram-side delivery using the PTB job queue, called cross-thread via `call_soon_threadsafe`. |
| `tg_agent_interaction.py` | Core turn loop: sets `tools.current_chat_context`, streams `astream_events` from the compiled graph, incrementally builds and flushes the reply to Telegram (chunking long output, showing a transient "Processing..." status between tool/model steps), and handles cancellation (`/stop`) and error cases. |

## The agent loop (`graph.py`)

- **planner_node**: given the objective (fresh, or preserved across a
  replan) and trimmed history, produces an ordered `Plan.steps` list via
  structured output.
- **executor_node**: works on `plan[0]` only, with access to all `TOOLS`
  plus Gemini's server-side `google_search`. Streams its response; a
  step-scoped transcript (`_step_transcript`) is rebuilt from the raw
  message history each time so the executor/reviewer never rely on stale
  summaries.
- **tools node**: a `ToolNode(TOOLS)`; routed to whenever the executor's
  last message has pending tool calls and the per-step tool-call budget
  (`MAX_TOOL_CALLS_PER_STEP`) hasn't been hit.
- **reviewer_node**: judges only the just-attempted step from the raw
  transcript (not the plan or the executor's self-report), and returns one
  of `goal_achieved` / `task_success` / `needs_replan`. Includes safety
  valves: no evidence at all forces `needs_replan`; `MAX_REPLANS` forces
  `goal_achieved`; `MAX_STEPS` forces `goal_achieved`.
- Routing: `needs_replan` → `planner`; steps remaining → `executor`;
  otherwise → `END`.

State (`AgentState`) is checkpointed per `thread_id` via `AsyncSqliteSaver`,
so plans, past steps, and full message history persist across bot restarts
until `/clear` is used.

## Environment variables (`config.py`)

All are required (no silent defaults) — the process raises `EnvironmentError`
at import time if any is missing:

- `LLM_API_KEY` — Google Generative AI API key.
- `HEAVY_MODEL_NAME` — Gemini model used for planning and execution.
- `LIGHT_MODEL_NAME` — Gemini model used for the reviewer.
- `TELEGRAM_BOT_TOKEN` — Telegram bot token.
- `AGENT_WORKSPACE` — root directory for per-thread workspaces.
- `MAX_HISTORY_TOKENS` — integer token budget for `trim_messages`.

Loaded via `python-dotenv`, so a local `.env` file works for development.

## Running locally

```bash
pip install -r requirements.txt
# create a .env with LLM_API_KEY, HEAVY_MODEL_NAME, LIGHT_MODEL_NAME,
# TELEGRAM_BOT_TOKEN, AGENT_WORKSPACE, MAX_HISTORY_TOKENS
python tg_bot.py
```

On startup this creates `.data/history.db` (LangGraph checkpoints) and
`.data/agent.log` (rotating log file). Per-thread workspaces are created
lazily under `AGENT_WORKSPACE/<sanitized-thread-key>/`.

## Conventions and gotchas for future changes

- **Thread identity**: a "thread" is `user_id` alone, or `user_id:message_thread_id`
  for forum topics (`build_thread_key`). This string is used as the LangGraph
  `thread_id`, sanitized for the workspace directory name, and used to key
  every in-memory dict in `tg_bot_state.py`. Keep these three uses in sync
  if the key scheme ever changes.
- **`current_chat_context`**: tools and the logging callback read the active
  chat/thread via this `contextvars.ContextVar`, set once per turn in
  `tg_agent_interaction._run_agent_turn` and reset in its `finally` block.
  Any new tool needing chat context should read it from here rather than
  taking chat_id as a parameter.
- **Sandboxed filesystem access**: `tools._resolve_safe_path` is the only
  sanctioned way tools touch disk; it resolves against
  `workspace.workspace_dir(thread_key)` and rejects paths that escape it.
  Don't bypass it with raw `os.path.join` in new tools.
- **Callback injection to avoid circular imports**: `tools.SCHEDULE_CALLBACK`
  and `tools.SEND_FILE_CALLBACK` are plain module attributes set once in
  `tg_bot.py:main()`. `tools.py` has no import-time dependency on the
  Telegram layer; preserve that direction if adding new cross-cutting
  callbacks.
- **Cross-thread scheduling**: `tg_msg_scheduler.schedule_followup` and
  `tg_file_manager.send_file_callback` are called from tool code that may be
  running on a different thread than the asyncio event loop, hence the use
  of `call_soon_threadsafe` / `run_coroutine_threadsafe` against
  `tg_bot_state.MAIN_LOOP`.
- **Message debouncing**: `handle_message` buffers rapid consecutive text
  messages for `MESSAGE_COMBINE_DELAY` seconds and merges them into a single
  turn; cancelling the pending flush task is how `/stop` interrupts a
  not-yet-started turn, while cancelling `_active_turn_tasks[thread_key]`
  interrupts an in-flight one.
- **Streaming output**: `tg_agent_interaction.flush()` incrementally edits a
  single Telegram message, splitting into new messages only when the
  Markdown-formatted length would exceed `tg_bot_state.SAFE_CHUNK`. Only
  `on_chat_model_stream` events from the `executor` node are treated as
  user-visible text; other node activity only updates the transient
  "Processing..." status line.
- **Step transcripts over self-reports**: both the executor and reviewer
  reconstruct what actually happened from raw messages
  (`graph._step_transcript`) rather than trusting a model's summary of its
  own actions — preserve this pattern if the loop is extended, since it's
  what keeps the reviewer's verdicts grounded.
- **Structured output models** (`Plan`, `Review`) are Pydantic models bound
  via `.with_structured_output`; field descriptions double as the prompt
  the model sees, so keep them precise when editing.
- **History trimming**: `trimmed_history()` uses a crude 4-chars-per-token
  estimate (`token_counter`), not a real tokenizer — fine for budgeting but
  don't rely on it for exactness.
- **`/clear` semantics**: deletes LangGraph checkpoint rows for the thread
  *and* the on-disk workspace (including uploads and notes). There's no
  soft-delete/undo.