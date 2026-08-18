# Telegram Agent Bot

A Telegram bot backed by an autonomous **plan → execute → review** AI agent,
built with [LangGraph](https://github.com/langchain-ai/langgraph) and
Google Gemini. Talk to it like a normal chat — it breaks your request into
steps, works through them with real tools (web search, a private sandboxed
filesystem, shell commands, HTTP requests), checks its own work, and streams
the answer back to you live as it types.

## Features

- **Plan/execute/review loop** — a planner drafts a short step list, an
  executor carries out one step at a time with tool access, and a reviewer
  grades each step, revising the plan if something goes wrong.
- **Live streaming replies** — responses are edited into your chat in real
  time, with a transient "Processing..." status while the agent is thinking
  or using a tool.
- **Per-chat sandboxed workspace** — every chat (or forum topic) gets its
  own private directory the agent can read from, write to, and run shell
  commands in. Uploaded files land there too.
- **Web search** — the agent can search the web for current information
  when it needs to.
- **File in, file out** — send photos or documents and the agent can see
  and use them; it can also send files back to you from its workspace.
- **Scheduled follow-ups** — the agent can schedule itself to message you
  again later (e.g. "remind me in an hour").
- **Per-chat memory notes** — the agent can jot down durable notes about a
  conversation to recall later.
- **Persistent history** — conversations survive bot restarts, stored in a
  local SQLite database.
- **Message debouncing** — if you send several messages in a row, they're
  combined into a single turn instead of triggering multiple replies.

## Commands

| Command | Effect |
|---|---|
| `/clear` | Wipes the conversation history and workspace for the current chat, starting fresh. |
| `/stop` | Cancels a reply that's currently being generated (or a pending one that hasn't started yet). |

Just send a normal message to start a conversation — no command needed.

## Requirements

- Python 3.11+
- A Telegram bot token ([create one via @BotFather](https://t.me/BotFather))
- A Google Generative AI (Gemini) API key

## Setup

1. **Clone the repo and install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the project root with:

   ```env
   LLM_API_KEY=your_google_generative_ai_api_key
   HEAVY_MODEL_NAME=gemini-2.5-pro
   LIGHT_MODEL_NAME=gemini-2.5-flash
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   AGENT_WORKSPACE=./workspace
   MAX_HISTORY_TOKENS=100000
   ```

   | Variable | Description |
   |---|---|
   | `LLM_API_KEY` | API key for Google's Generative AI (Gemini). |
   | `HEAVY_MODEL_NAME` | Gemini model used for planning and step execution. |
   | `LIGHT_MODEL_NAME` | Gemini model used for reviewing each step (a cheaper/faster model works well here). |
   | `TELEGRAM_BOT_TOKEN` | Your Telegram bot's token from BotFather. |
   | `AGENT_WORKSPACE` | Local directory where per-chat workspaces are created. |
   | `MAX_HISTORY_TOKENS` | Rough token budget for how much conversation history is kept in context. |

3. **Run the bot:**

   ```bash
   python tg_bot.py
   ```

   On first run this creates a `.data/` directory for the conversation
   database and log file, and creates per-chat folders under
   `AGENT_WORKSPACE` as people start chatting.

## How it works

1. You send a message. If you send several in quick succession, they're
   combined into one turn.
2. A **planner** breaks your request into a short list of concrete steps.
3. An **executor** works through the steps one at a time, calling tools
   (file access, shell commands, web search, HTTP requests, scheduling,
   etc.) as needed.
4. A **reviewer** checks each completed step. If something went wrong, it
   sends the agent back to re-plan; otherwise it moves to the next step.
5. Once everything is done, the final reply is streamed back into your
   Telegram chat.

Each chat has its own isolated workspace on disk and its own conversation
history, so nothing leaks between different chats or users.

## Project structure

See [AGENTS.md](./AGENTS.md) for a detailed breakdown of the codebase,
architecture, and development conventions.

## Tech stack

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) — Telegram integration
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) + `langchain-google-genai` — Gemini integration and tool calling
- SQLite (via `langgraph-checkpoint-sqlite`) — persistent conversation state
- [telegramify-markdown](https://pypi.org/project/telegramify-markdown/) — Markdown → Telegram-safe formatting