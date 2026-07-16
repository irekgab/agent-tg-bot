"""Agent graph definition.
Nodes:
    start_turn -> planner -> agent <-> tools -> finish_step
        -> [replan -> agent <-> tools -> finish_step]* -> END

`messages` is the durable, full conversation (persisted across turns by
the checkpointer, same as before). `plan` / `past_steps` / `current_task`
/ `step_messages` are scratch state for the turn currently in progress,
reset by `start_turn` every time a new user message comes in.
"""
from typing import Annotated, List, Literal, Tuple, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from llm import build_llm, build_raw_llm
from tools import TOOLS


GOOGLE_SEARCH_TOOL = {"google_search": {}}


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
that can browse the web, run shell commands, read/write files, and \
schedule follow-up messages. Given the conversation so far, break the \
user's newest message into a short ordered list of concrete steps. Only \
split out a step if it needs its own tool call or is a distinct piece of \
work; keep unrelated parts of the request as separate steps so they can \
be executed independently. If the request is a single simple ask, the \
plan may just be one step."""

REPLANNER_SYSTEM_PROMPT = """You are the planning component of an \
assistant. A plan was made for the user's request; some steps have \
already been executed and their results are shown below. Decide what \
happens next: if every step needed to satisfy the user's original \
request has now been executed, set is_complete to true. Otherwise set \
is_complete to false and list ONLY the steps that still remain - never \
repeat a step whose result already appears below."""


def _step_task_message(plan: List[str]) -> HumanMessage:
    """Build the scoped instruction for executing plan[0] - and only plan[0]."""
    plan_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
    return HumanMessage(
        content=(
            f"Full plan (for context only, do not act on the other steps):\n{plan_str}\n\n"
            f"Execute ONLY this step now:\n1. {plan[0]}\n\n"
            "Give your final answer for this step alone; do not mention or "
            "address any other step."
        )
    )


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

    async def plan_step(state: AgentState) -> AgentState:
        plan = await planner.ainvoke([("system", PLANNER_SYSTEM_PROMPT), *state["messages"]])
        steps = plan.steps or ["Respond to the user's request."]
        return {
            "plan": steps,
            "current_task": steps[0],
            "step_messages": _replace_step_messages(state["step_messages"], _step_task_message(steps)),
        }

    async def agent_node(state: AgentState) -> AgentState:
        response = await llm_with_tools.ainvoke(state["step_messages"])
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

    async def replan_step(state: AgentState) -> AgentState:
        past_steps_str = "\n\n".join(
            f"Step: {task}\nResult: {result}" for task, result in state["past_steps"]
        )
        original_request = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        prompt = (
            f"Original request:\n{original_request}\n\n"
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
            "step_messages": _replace_step_messages(state["step_messages"], _step_task_message(new_plan)),
        }

    def route_after_replan(state: AgentState) -> str:
        # (returns "agent" or END - see note on route_after_finish above)
        if state["plan"]:
            return "agent"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("start_turn", start_turn)
    graph.add_node("planner", plan_step)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS, messages_key="step_messages"))
    graph.add_node("finish_step", finish_step)
    graph.add_node("replan", replan_step)

    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "planner")
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
