from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import trim_messages

from llm import build_llm, LoggingCallbackHandler, extract_text
from tools import TOOLS, GOOGLE_SEARCH_TOOL
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from config import MAX_HISTORY_TOKENS


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def get_graph_definition() -> StateGraph:
    llm_with_tools = build_llm().bind_tools(
        [*TOOLS, GOOGLE_SEARCH_TOOL],
        tool_config={"include_server_side_tool_invocations": True},
    )

    async def agent_node(state: AgentState) -> AgentState:
        def token_counter(messages):
            return sum(llm_with_tools.get_num_tokens(extract_text(m.content)) for m in messages)

        trimmed_messages = trim_messages(
            state["messages"],
            max_tokens=MAX_HISTORY_TOKENS,
            token_counter=token_counter,
            strategy="last",
            include_system=True,
            allow_partial=False,
        )
        response = await llm_with_tools.ainvoke(
            trimmed_messages,
            config={"callbacks": [LoggingCallbackHandler()]}
        )
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    
    return graph

async def build_graph_async():
    async with AsyncSqliteSaver.from_conn_string(".data/history.db") as memory:
        graph = get_graph_definition()
        yield graph.compile(checkpointer=memory)