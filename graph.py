"""Agent graph definition.

Builds a small ReAct-style LangGraph: the agent node calls the LLM
(with tools bound), and a conditional edge routes to the tool node
whenever the LLM requests a tool call, looping until a final answer
is produced.
"""
import sqlite3
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm import build_llm
from tools import TOOLS
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


GOOGLE_SEARCH_TOOL = {"google_search": {}}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def get_graph_definition() -> StateGraph:
    """Returns the uncompiled StateGraph definition."""
    llm_with_tools = build_llm().bind_tools(
        [*TOOLS, GOOGLE_SEARCH_TOOL],
        tool_config={"include_server_side_tool_invocations": True},
    )

    def agent_node(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # agent -> tools or END
    graph.add_edge("tools", "agent")
    
    return graph


def build_graph() -> any:
    """Builds the graph with a synchronous SqliteSaver (for CLI)."""
    graph = get_graph_definition()
    conn = sqlite3.connect(".data/history.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(checkpointer=memory)


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
