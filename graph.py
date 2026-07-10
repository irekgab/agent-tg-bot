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
from langgraph.checkpoint.sqlite import SqliteSaver

from llm import build_llm
from tools import TOOLS


GOOGLE_SEARCH_TOOL = {"google_search": {}}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
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

    conn = sqlite3.connect(".data/history.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    return graph.compile(checkpointer=memory)
