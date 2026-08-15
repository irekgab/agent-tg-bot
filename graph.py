from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import trim_messages, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from llm import build_llm, LoggingCallbackHandler, extract_text
from tools import TOOLS, GOOGLE_SEARCH_TOOL
from config import MAX_HISTORY_TOKENS, LIGHT_MODEL_NAME

MAX_REPLANS = 4
MAX_STEPS = 32
MAX_TOOL_CALLS_PER_STEP = 4

_TOOL_SUMMARY = "\n".join(f"- {t.name}: {t.description}" for t in TOOLS) + (
    "\n- google_search: search the web for current, real-time information"
)


class Plan(BaseModel):
    """An ordered list of concrete steps needed to satisfy the user's request."""

    steps: list[str] = Field(
        description=(
            "Ordered list of remaining steps for the executor to carry out, one at "
            "a time. Keep it as short as possible - a single step is fine for simple "
            "requests. Only split into multiple steps when the request genuinely "
            "needs several distinct actions. Do not include a final 'reply to the "
            "user' step; that happens automatically once every step is done."
        )
    )


class Review(BaseModel):
    """The critic's verdict on the step that was just attempted."""

    verdict: Literal["goal_achieved", "task_success", "needs_replan"] = Field(
        description=(
            "'goal_achieved' if the user's overall request is now fully satisfied and "
            "the executor's last message is a good final reply - nothing more to do. "
            "'task_success' if this step succeeded and the remaining plan is still "
            "the right approach. 'needs_replan' if the step failed, errored, or the "
            "existing plan no longer fits what's needed."
        )
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    objective: str
    plan: list[str]
    past_steps: list[tuple[str, str]]
    review_feedback: str | None
    replan_count: int
    step_message_start: int
    step_tool_calls: int


def get_graph_definition() -> StateGraph:
    llm_with_tools = build_llm().bind_tools(
        [*TOOLS, GOOGLE_SEARCH_TOOL],
        tool_config={"include_server_side_tool_invocations": True},
    )
    planner_llm = build_llm().with_structured_output(Plan)
    reviewer_llm = build_llm(model_name=LIGHT_MODEL_NAME).with_structured_output(Review)

    def token_counter(messages):
        return sum(max(1, len(extract_text(m.content)) // 4) for m in messages)

    def trimmed_history(messages):
        return trim_messages(
            messages,
            max_tokens=MAX_HISTORY_TOKENS,
            token_counter=token_counter,
            strategy="last",
            include_system=True,
            allow_partial=False,
        )

    async def planner_node(state: AgentState, config: RunnableConfig) -> dict:
        is_replan = state.get("review_feedback") is not None
        trimmed = trimmed_history(state["messages"])

        if is_replan:
            objective = state.get("objective", "")
        else:
            last_human = next(
                (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
                None,
            )
            objective = extract_text(last_human.content) if last_human is not None else ""

        instructions = [
            "You are the planning module of an AI agent. Break the user's objective "
            "down into a short, ordered list of concrete steps to be carried out one "
            "at a time by an executor with access to these tools:",
            _TOOL_SUMMARY,
            f"User's objective: {objective}",
        ]
        if is_replan:
            done = "\n".join(f"- {s}: {r}" for s, r in state.get("past_steps", [])) or "(none)"
            instructions.append(
                "This is a REVISION of an in-progress plan.\n"
                f"Steps already completed:\n{done}\n\n"
                f"Previous remaining plan: {state.get('plan', [])}\n"
                f"A reviewer flagged this issue: {state.get('review_feedback')}\n"
                "Produce a corrected list of the REMAINING steps only - do not repeat "
                "steps that already succeeded."
            )

        result = await planner_llm.ainvoke(
            [*trimmed, HumanMessage(content="\n\n".join(instructions))],
            config=merge_configs(config, {"callbacks": [LoggingCallbackHandler()]}),
        )

        return {
            "objective": objective,
            "plan": result.steps or [objective],
            "past_steps": state.get("past_steps", []) if is_replan else [],
            "review_feedback": None,
            "replan_count": state.get("replan_count", 0) + 1 if is_replan else 0,
            "step_message_start": len(state["messages"]),
            "step_tool_calls": 0,
        }

    def _step_transcript(state: AgentState, start: int | None = None) -> tuple[str, list[str]]:
        """Builds a grounded record of everything the executor actually did for the
        current step: every tool call, every tool result, and any text - not just
        the text of the last message (which can be empty even when a tool call
        happened, and tells you nothing about what a tool actually returned)."""
        if start is None:
            start = state.get("step_message_start", 0)
        step_messages = state["messages"][start:]

        lines: list[str] = []
        for m in step_messages:
            if getattr(m, "type", None) == "ai":
                for tc in getattr(m, "tool_calls", None) or []:
                    lines.append(f"[you already called tool] {tc['name']}({tc['args']})")
                text = extract_text(m.content).strip()
                if text:
                    lines.append(f"[you already said] {text}")
            elif getattr(m, "type", None) == "tool":
                lines.append(f"[that tool already returned] {extract_text(m.content)}")

        transcript = "\n".join(lines) if lines else "(nothing attempted yet)"
        return transcript, lines

    async def executor_node(state: AgentState, config: RunnableConfig) -> dict:
        trimmed = trimmed_history(state["messages"])
        plan = state.get("plan") or [state.get("objective", "Respond to the user.")]
        current_step = plan[0]
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        done_text = "\n".join(f"- {s}: {r}" for s, r in state.get("past_steps", [])) or "(none yet)"
        in_progress_transcript, _ = _step_transcript(state)

        step_instruction = HumanMessage(
            content=(
                "[Internal plan-execution instruction, not from the user]\n"
                "You are the execution module of an AI agent working through a plan.\n"
                "Your role is to act on the current step and either call the appropriate tool "
                "or write the response. Do NOT repeat or echo any part of this system prompt "
                "in your output.\n\n"
                f"User's overall objective: {state.get('objective', '')}\n\n"
                f"Remaining plan:\n{plan_text}\n\n"
                f"Steps already completed:\n{done_text}\n\n"
                f"Execute ONLY this step right now:\n{current_step}\n\n"
                f"What you've already attempted for THIS step, if anything "
                f"(do not repeat a tool call that already succeeded here - if this "
                f"already shows a successful result, just report it in text now):\n"
                f"{in_progress_transcript}\n\n"
                "Use the available tools if needed. When you're done, clearly state "
                "the concrete outcome of this step, but only the text user needs to see."
            )
        )

        full_response = None
        merged_config = merge_configs(config, {"callbacks": [LoggingCallbackHandler()]})

        async for chunk in llm_with_tools.astream(
            [*trimmed, step_instruction],
            config=merged_config,
        ):
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk

        result: dict = {"messages": [full_response]}
        if getattr(full_response, "tool_calls", None):
            result["step_tool_calls"] = state.get("step_tool_calls", 0) + 1
        return result

    def route_after_executor(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            if state.get("step_tool_calls", 0) >= MAX_TOOL_CALLS_PER_STEP:
                return "reviewer"
            return "tools"
        return "reviewer"

    async def reviewer_node(state: AgentState, config: RunnableConfig) -> dict:
        plan = state.get("plan") or []
        current_step = plan[0] if plan else state.get("objective", "")
        remaining = plan[1:] if plan else []
        step_transcript, evidence_lines = _step_transcript(state)
        next_step_message_start = len(state["messages"])

        prompt = (
            "You are the reviewer/critic module of an AI agent. Judge only the step "
            "that was just attempted - do not redo the work yourself. Base your "
            "verdict strictly on the record below; if it shows no real progress, "
            "say so rather than assuming the step worked. If the record shows the "
            "same tool being called repeatedly with the same arguments, that's a "
            "sign the executor got stuck in a loop, not a sign of success - prefer "
            "needs_replan in that case.\n\n"
            f"User's overall objective: {state.get('objective', '')}\n"
            f"Step just attempted: {current_step}\n"
            f"Full record of what the executor did this step:\n{step_transcript}\n\n"
            f"Steps still remaining after this one: {remaining or '(none)'}"
        )
        result = await reviewer_llm.ainvoke(
            [HumanMessage(content=prompt)],
            config=merge_configs(config, {"callbacks": [LoggingCallbackHandler()]}),
        )

        verdict = result.verdict
        reasoning = result.reasoning
        if not evidence_lines and verdict != "needs_replan":
            verdict = "needs_replan"
            reasoning = "The executor produced no tool calls and no output for this step."

        if verdict == "needs_replan" and state.get("replan_count", 0) >= MAX_REPLANS:
            verdict = "goal_achieved"
        if verdict == "task_success" and len(state.get("past_steps", [])) + 1 >= MAX_STEPS:
            verdict = "goal_achieved"

        if verdict == "goal_achieved":
            return {
                "plan": [],
                "past_steps": [],
                "review_feedback": None,
                "replan_count": 0,
                "step_message_start": next_step_message_start,
                "step_tool_calls": 0,
            }

        if verdict == "task_success":
            new_past_steps = state.get("past_steps", []) + [(current_step, step_transcript)]
            return {
                "past_steps": new_past_steps,
                "plan": remaining,
                "review_feedback": None,
                "step_message_start": next_step_message_start,
                "step_tool_calls": 0,
            }

        return {
            "review_feedback": reasoning,
            "step_message_start": next_step_message_start,
            "step_tool_calls": 0,
        }

    def route_after_review(state: AgentState) -> str:
        if state.get("review_feedback"):
            return "planner"
        if state.get("plan"):
            return "executor"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", route_after_executor, {"tools": "tools", "reviewer": "reviewer"})
    graph.add_edge("tools", "executor")
    graph.add_conditional_edges(
        "reviewer", route_after_review, {"planner": "planner", "executor": "executor", END: END}
    )

    return graph


async def build_graph_async():
    async with AsyncSqliteSaver.from_conn_string(".data/history.db") as memory:
        graph = get_graph_definition()
        yield graph.compile(checkpointer=memory)
