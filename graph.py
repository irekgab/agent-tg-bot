"""Agent graph definition.
Nodes:
    start_turn -> compact_history -> planner -> agent <-> tools -> finish_step
        -> [replan -> agent <-> tools -> finish_step]* -> END

`messages` is the durable, full conversation (persisted across turns by
the checkpointer). `plan` / `past_steps` / `current_task` / `step_messages`
are scratch state for the turn currently in progress, reset by
`start_turn` every time a new user message comes in.

`conversation_summary` is durable too (not reset by start_turn):
`compact_history` keeps `messages` bounded to a recent, dynamically-sized
window and folds anything older into this summary, so a long-running
conversation doesn't grow the checkpoint - or the planner's input - without
bound.

Every LLM call in this graph (planner, replanner, and the tool-executing
agent) is additionally primed with: the operator's global instructions
file (applies to every chat), this chat's own persistent notes file (the
agent maintains it via the remember_about_chat tool), and the rolling
conversation_summary above - see `_context_blocks`.
"""
from typing import Annotated, List, Literal, Tuple, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from llm import build_llm, build_raw_llm, extract_planner_text, extract_text
from tools import TOOLS
import workspace as ws


GOOGLE_SEARCH_TOOL = {"google_search": {}}

# Recent messages are kept verbatim in `messages` up to this approximate
# token budget (a dynamically-sized window: a run of short messages fits
# more of them, a couple of huge ones may be all that fits). Anything
# older gets folded into `conversation_summary` and dropped - see
# `compact_history`.
HISTORY_TOKEN_BUDGET = 3000

SUMMARIZER_SYSTEM_PROMPT = """You maintain a running summary of an ongoing \
conversation between a user and an AI assistant, so that older turns can \
be dropped from the assistant's context without losing the thread. Given \
the existing summary (if any) and a batch of older messages that are \
about to be dropped, write an updated summary that folds the new \
messages into it. Keep it compact - a few sentences to a short paragraph \
- and focus on facts, decisions, preferences, and unresolved threads the \
assistant will need to stay consistent later. Output only the summary \
text itself, with no preamble."""


def _thread_key_from_config(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def _context_blocks(thread_key: str, conversation_summary: str | None) -> list[tuple[str, str]]:
    """(role, text) system blocks shared by every LLM call in this graph:
    the operator's global instructions (all chats), this chat's own
    persistent notes (the agent maintains these via remember_about_chat),
    and the rolling conversation summary. Read fresh every call - global
    instructions and notes can change mid-conversation (an operator
    editing the file, or the agent calling remember_about_chat)."""
    blocks = []

    global_instructions = ws.load_global_instructions()
    if global_instructions:
        blocks.append((
            "system",
            f"Operator-configured instructions that apply to every chat:\n{global_instructions}",
        ))

    notes = ws.load_chat_notes(thread_key)
    if notes:
        blocks.append((
            "system",
            f"Notes you previously saved about this chat/user (see remember_about_chat):\n{notes}",
        ))

    if conversation_summary:
        blocks.append((
            "system",
            f"Summary of earlier parts of this conversation (older messages "
            f"have already been compacted out of context):\n{conversation_summary}",
        ))

    return blocks


class AgentState(TypedDict):
    # The durable, full conversation - persisted across turns via the
    # checkpointer, and what the planner reads for context.
    messages: Annotated[list, add_messages]

    # --- Scratch state for the turn currently in progress. ---
    # Reset to empty/None by start_turn on every new user message.
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    current_task: str | None
    # The narrow, per-step working conversation that a step's inner
    # agent<->tools loop reads and writes. Deliberately never shown the
    # other steps, so it can't "helpfully" redo them. Uses the
    # add_messages reducer so ToolNode can just append ToolMessages;
    # it's reset between steps with RemoveMessage (see _replace_step_messages).
    step_messages: Annotated[list, add_messages]

    # --- Durable, cross-turn state (NOT reset by start_turn). ---
    # A running summary of whatever has been dropped out of `messages`
    # by compact_history, so context from far back in a long conversation
    # isn't simply lost. Plain str - overwritten wholesale each time it's
    # updated, no reducer needed.
    conversation_summary: str


class Plan(BaseModel):
    """An ordered list of steps to fully satisfy the user's request."""

    steps: List[str] = Field(
        description="Individual, self-contained steps, in the order they "
        "should be executed. Split unrelated parts of the request into "
        "separate steps so they can be executed independently - e.g. "
        "'summarize X, then wait 15 seconds, then summarize Y' is three "
        "steps: summarize X, the wait, summarize Y. A single simple "
        "request can just be one step."
    )


class Replan(BaseModel):
    """Decision on whether the plan is finished or needs revising."""

    is_complete: bool = Field(
        description="True if every part of the user's original request "
        "has now been handled by the executed steps and nothing further "
        "is needed."
    )
    remaining_steps: List[str] = Field(
        default_factory=list,
        description="If not complete, the ordered list of steps that "
        "still need to be executed. Do not include steps that were "
        "already executed, and do not duplicate their work.",
    )


PLANNER_SYSTEM_PROMPT = """You are the planning component of an assistant \
that can browse the web, run shell commands, read/write files, send \
documents/images to the user, and schedule follow-up messages. Given the \
conversation so far, break the user's newest message into a short \
ordered list of concrete steps. Only split out a step if it needs its \
own tool call or is a distinct piece of work; keep unrelated parts of \
the request as separate steps so they can be executed independently. If \
the request is a single simple ask, the plan may just be one step."""

REPLANNER_SYSTEM_PROMPT = """You are the planning component of an \
assistant. A plan was made for the user's request; some steps have \
already been executed and their results are shown below. Decide what \
happens next: if every step needed to satisfy the user's original \
request has now been executed, set is_complete to true. Otherwise set \
is_complete to false and list ONLY the steps that still remain - never \
repeat a step whose result already appears below."""


def _step_task_message(plan: List[str], latest_human_message: HumanMessage) -> HumanMessage:
    """Build the scoped instruction for executing plan[0] - and only plan[0]."""
    plan_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))

    instruction = (
        f"\n\nFull plan (for context only, do not act on the other steps):\n{plan_str}\n\n"
        f"Execute ONLY this step now:\n1. {plan[0]}\n\n"
        "Give your final answer for this step alone; do not mention or "
        "address any other step."
    )
    
    if isinstance(latest_human_message.content, list):
        # We need to make sure we don't modify the original content object if it's a list (it's mutable)
        new_content = list(latest_human_message.content)
        new_content.append({"type": "text", "text": instruction})
    else:
        new_content = f"{latest_human_message.content}{instruction}"
        
    return HumanMessage(content=new_content)


def _replace_step_messages(old_messages: list, new_message: HumanMessage) -> list:
    """Reset the add_messages-backed step_messages list to just [new_message].

    add_messages only ever appends unless told to delete, so clearing it
    for a fresh step means explicitly removing every prior message by id.
    """
    return [RemoveMessage(id=m.id) for m in old_messages] + [new_message]


def get_graph_definition() -> StateGraph:
    """Returns the uncompiled StateGraph definition."""
    planner = build_raw_llm(temperature=0).with_structured_output(Plan)
    replanner = build_raw_llm(temperature=0).with_structured_output(Replan)
    llm_with_tools = build_llm().bind_tools(
        [*TOOLS, GOOGLE_SEARCH_TOOL],
        tool_config={"include_server_side_tool_invocations": True},
    )

    async def start_turn(state: AgentState) -> AgentState:
        """Resets per-turn scratch state for a freshly arrived user message."""
        return {
            "plan": [],
            "past_steps": [],
            "current_task": None,
            "step_messages": [RemoveMessage(id=m.id) for m in state.get("step_messages", [])],
        }

    async def compact_history(state: AgentState) -> AgentState:
        """Keeps `messages` bounded going forward. Recent messages are kept
        verbatim up to HISTORY_TOKEN_BUDGET (a dynamic window - lots of
        short messages fit, only one or two huge ones might), and anything
        older than that is folded into `conversation_summary` via a single
        extra LLM call and then permanently dropped with RemoveMessage, so
        neither the checkpointed history nor the planner's input to the
        model grows without bound over a long-running conversation.
        """
        messages = state["messages"]
        kept = trim_messages(
            messages,
            max_tokens=HISTORY_TOKEN_BUDGET,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            include_system=False,
        )
        kept_ids = {m.id for m in kept}
        dropped = [m for m in messages if m.id not in kept_ids]

        if not dropped:
            return {}

        dropped_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {extract_planner_text(m.content)}"
            for m in dropped
        )
        prior_summary = state.get("conversation_summary") or "(none yet)"
        prompt = (
            f"Existing summary:\n{prior_summary}\n\n"
            f"Older messages to fold in:\n{dropped_text}"
        )
        summarizer = build_raw_llm(temperature=0)
        result = await summarizer.ainvoke([("system", SUMMARIZER_SYSTEM_PROMPT), ("user", prompt)])
        new_summary = extract_text(result.content).strip()

        return {
            "conversation_summary": new_summary or prior_summary,
            "messages": [RemoveMessage(id=m.id) for m in dropped],
        }

    async def plan_step(state: AgentState, config: RunnableConfig) -> AgentState:
        # Prepare the messages for the planner
        thread_key = _thread_key_from_config(config)
        summary = state.get("conversation_summary")
        planner_messages = [("system", PLANNER_SYSTEM_PROMPT), *_context_blocks(thread_key, summary)]
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                planner_messages.append(("human", extract_planner_text(m.content)))
            elif isinstance(m, AIMessage):
                planner_messages.append(("ai", extract_planner_text(m.content)))
        
        plan = await planner.ainvoke(planner_messages)
        steps = plan.steps or ["Respond to the user's request."]
        
        latest_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        
        return {
            "plan": steps,
            "current_task": steps[0],
            "step_messages": _replace_step_messages(state["step_messages"], _step_task_message(steps, latest_human)),
        }

    async def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
        thread_key = _thread_key_from_config(config)
        context = _context_blocks(thread_key, state.get("conversation_summary"))
        response = await llm_with_tools.ainvoke([*context, *state["step_messages"]])
        return {"step_messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["tools", "finish_step"]:
        last = state["step_messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finish_step"

    async def finish_step(state: AgentState) -> AgentState:
        last = state["step_messages"][-1]
        result_text = last.content if isinstance(last.content, str) else str(last.content)
        return {
            "past_steps": state["past_steps"] + [(state["current_task"], result_text)],
            "messages": [AIMessage(content=result_text)],
        }

    def route_after_finish(state: AgentState) -> str:
        # (returns "replan" or END - END is a runtime constant, not a
        # Literal-compatible type, hence the plain `str` annotation)
        # Common-case optimization: a single-step plan needs no replanning
        # LLM call at all - just finish, same latency as the old single
        # agent call.
        if len(state["plan"]) <= 1:
            return END
        return "replan"

    async def replan_step(state: AgentState, config: RunnableConfig) -> AgentState:
        past_steps_str = "\n\n".join(
            f"Step: {task}\nResult: {result}" for task, result in state["past_steps"]
        )
        latest_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        original_request = extract_planner_text(latest_human.content)
        thread_key = _thread_key_from_config(config)
        summary = state.get("conversation_summary")

        context_text = "\n\n".join(text for _, text in _context_blocks(thread_key, summary))
        prompt = (
            (f"{context_text}\n\n" if context_text else "")
            + f"Original request:\n{original_request}\n\n"
            f"Original plan:\n" + "\n".join(f"- {s}" for s in state["plan"]) + "\n\n"
            f"Steps executed so far:\n{past_steps_str}"
        )
        result = await replanner.ainvoke([("system", REPLANNER_SYSTEM_PROMPT), ("user", prompt)])
        if result.is_complete or not result.remaining_steps:
            return {"plan": []}
        new_plan = result.remaining_steps
        return {
            "plan": new_plan,
            "current_task": new_plan[0],
            "step_messages": _replace_step_messages(state["step_messages"], _step_task_message(new_plan, latest_human)),
        }

    def route_after_replan(state: AgentState) -> str:
        # (returns "agent" or END - see note on route_after_finish above)
        if state["plan"]:
            return "agent"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("start_turn", start_turn)
    graph.add_node("compact_history", compact_history)
    graph.add_node("planner", plan_step)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS, messages_key="step_messages"))
    graph.add_node("finish_step", finish_step)
    graph.add_node("replan", replan_step)

    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "compact_history")
    graph.add_edge("compact_history", "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finish_step": "finish_step"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("finish_step", route_after_finish, {"replan": "replan", END: END})
    graph.add_conditional_edges("replan", route_after_replan, {"agent": "agent", END: END})

    return graph


async def build_graph_async():
    """
    A generator that yields a compiled graph with an AsyncSqliteSaver.
    Usage:
        async for app in build_graph_async():
            # use app
    """
    async with AsyncSqliteSaver.from_conn_string(".data/history.db") as memory:
        graph = get_graph_definition()
        yield graph.compile(checkpointer=memory)
