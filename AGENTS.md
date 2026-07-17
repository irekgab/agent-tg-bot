# AGENTS.md

Guidance for AI coding agents (Claude Code, etc.) working on this repo.
Read this before making changes - several invariants here aren't obvious
from any single file in isolation.

## What this is

A Telegram bot backed by a LangGraph plan-and-execute agent, talking to a
Gemini/Gemma model via `langchain-google-genai`. The user chats with it in
Telegram; under the hood every message is broken into a short plan, each
step is executed by a tool-calling agent, and the plan is revised until
the request is fully handled.

## File map

| File | Responsibility |
|---|---|
| `telegram_bot.py` | Entry point. Telegram handlers, message streaming/chunking, per-thread locking, scheduled follow-ups, sending files back to the user. |
| `graph.py` | The LangGraph state machine itself: state schema, planner/replanner, the agent<->tools loop, history compaction. |
| `tools.py` | Tool definitions available to the agent. Deliberately Telegram-agnostic. |
| `workspace.py` | Per-chat workspace layout on disk (files, notes, global instructions file). Shared by `tools.py` and `graph.py`. |
| `llm.py` | Model client factory + helpers for pulling text out of multimodal message content. |
| `config.py` | Env var loading. Every module reads config from here, never `os.environ` directly. |
| `logging_config.py` | Silences known-noisy third-party loggers. |
| `AGENT_INSTRUCTIONS.md` | Not code - the operator-editable file that shapes the *bot's* behavior. Don't confuse this with this file. |

## Architecture: the graph

```
START -> start_turn -> compact_history -> planner -> agent <-> tools -> finish_step
                                              -> [replan -> agent <-> tools -> finish_step]* -> END
```

- **start_turn**: resets per-turn scratch state when a new user message
  arrives.
- **compact_history**: keeps `messages` bounded - see "History compaction"
  below.
- **planner**: LLM call with structured output (`Plan`), breaks the
  request into ordered steps.
- **agent / tools**: the ReAct-style loop that executes the *current*
  step only. Runs repeatedly until the step's LLM call comes back with no
  tool calls.
- **finish_step**: records the step's result, appends a summary
  `AIMessage` to the durable `messages`.
- **replan**: after a step finishes, decides whether the original request
  is now fully satisfied or more steps are needed (skipped entirely for
  single-step plans - see `route_after_finish`).

### State: durable vs. scratch

`AgentState` in `graph.py` has two categories of field, and mixing them up
is the easiest way to introduce a subtle bug:

- **Durable** (persisted by the checkpointer, survives across turns):
  `messages`, `conversation_summary`. Never reset by `start_turn`.
- **Scratch** (reset every turn by `start_turn`): `plan`, `past_steps`,
  `current_task`, `step_messages`.

`step_messages` is deliberately narrow: the tool-executing agent only ever
sees the current step's task instruction, never the other steps or the
raw conversation. Context it *does* need (global instructions, this
chat's notes, the conversation summary) is injected fresh on every LLM
call via `_context_blocks()` rather than being persisted into any message
list - see `agent_node`, `plan_step`, `replan_step`.

### History compaction

`messages` would otherwise grow forever and get fully re-sent to the
planner on every turn. `compact_history` uses `trim_messages(...,
token_counter="approximate")` to keep a *dynamically sized* recent window
(many small messages, or just one or two huge ones) and folds anything
older into `conversation_summary` via one extra LLM call, then physically
removes those messages from `messages` with `RemoveMessage`. This means
the persisted history itself stays bounded, not just the planner's view
of it.

**If you add a new durable field**: don't reset it in `start_turn`, and
consider whether `compact_history` or the summarizer prompt needs to know
about it.

## Adding things

**A new tool**: add a plain `@tool`-decorated function in `tools.py` and
append it to `TOOLS`. Keep tools Telegram-agnostic - if a tool needs to
talk back to the chat directly (like `schedule_message` or
`send_file_to_user`), don't import Telegram stuff into `tools.py`.
Instead: read `current_chat_context.get()` for `chat_id` /
`message_thread_id` / `thread_key`, and call a host-registered callback
(a module-level `None` that `telegram_bot.py` sets in `main()`). Look at
`SCHEDULE_CALLBACK`/`schedule_followup` (fire-and-forget) and
`SEND_FILE_CALLBACK`/`send_file_callback` (blocks for a result via
`run_coroutine_threadsafe(...).result()`) as the two existing patterns -
pick whichever matches whether the agent needs to know if it succeeded.

**A new graph node**: define the async function, then **both**
`graph.add_node("name", fn)` **and** wire it in with `add_edge` /
`add_conditional_edges`. It's easy to write a node, get everything else
right, and forget the second half - that exact bug shipped once in this
repo (a fully-implemented `compact_history` node that was never added to
the graph, so it silently never ran). If a node's behavior seems to have
no effect, check it's actually reachable in `get_graph_definition()`
before debugging the logic itself.

**Needing `thread_id` inside a node**: add `config: RunnableConfig` as a
second parameter (LangGraph injects it automatically) and read
`config["configurable"]["thread_id"]` - see `_thread_key_from_config`.
This is also exactly the value used as the SQLite checkpoint thread and
as the workspace directory name via `workspace.py`.

## Workspace & per-chat isolation

Every chat gets its own directory under `workspace.WORKSPACE_ROOT`
(`.data/workspaces/<sanitized thread_key>/` by default), containing
`notes.md`, an `uploads/` folder, and anything the agent writes. All file
tools resolve paths against *that specific chat's* directory
(`tools._resolve_safe_path`) - there is no shared workspace across chats,
and no tool should ever be given a way to escape it. If you change
anything here, re-run the isolation check (two different `thread_key`s
must not be able to see each other's files) before shipping.

`/clear` deletes both the checkpointed conversation (the SQLite rows) and
the entire workspace directory (`workspace.delete_workspace`) - not just
`notes.md`. Keep those two in sync if you change what `/clear` does.

## Config / environment

Everything is read once in `config.py` via `_require_env`; other modules
import the resulting constants, never `os.getenv` directly (`workspace.py`
is the one exception, since its env vars are optional with defaults).

- `LLM_API_KEY`, `TELEGRAM_BOT_TOKEN` - required, no defaults.
- `MODEL_NAME` - defaults to `gemma-4-26b-a4b-it`.
- `AGENT_WORKSPACE` - root directory for *all* chats' workspaces (not a
  single shared workspace - each chat gets a subfolder under it). Defaults
  to `.data/workspaces`.
- `AGENT_INSTRUCTIONS_FILE` - path to the global instructions file.
  Defaults to `AGENT_INSTRUCTIONS.md` at the project root.

## Testing / verifying changes

There's no formal test suite yet. Before considering a change done:

1. `python3 -m py_compile` every file you touched.
2. Actually build and compile the graph:
   ```python
   import graph
   g = graph.get_graph_definition().compile()
   list(g.get_graph().nodes.keys())  # sanity-check your new node is there
   ```
3. For tool or workspace changes, exercise them directly with a fake
   `current_chat_context` rather than spinning up the whole bot - e.g.
   set two different `thread_key`s and confirm they can't see each
   other's files (this has caught real bugs before).
4. Set dummy `LLM_API_KEY`/`TELEGRAM_BOT_TOKEN` env vars for any of the
   above - `config.py` will otherwise refuse to import.

## Conventions

- Every tool and non-obvious function has a docstring explaining *why*,
  not just what - keep that up when adding to `tools.py` especially,
  since tool docstrings are also what the LLM sees to decide when to use
  them.
- Prefer small, focused comments over rewriting existing ones - several
  comments in this codebase exist specifically to prevent a bug from
  being reintroduced (e.g. the `step_messages` narrowing, the
  `RemoveMessage` reset pattern). Don't delete them just because the code
  looks self-explanatory.