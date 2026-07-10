"""Streaming helpers for printing agent responses token-by-token.

The agent's LLM can return content as a plain string, or (when the
model emits reasoning) as a list of content blocks like
[{"type": "thinking", ...}, {"type": "text", "text": "..."}]. Only the
"text" blocks are the actual answer meant for the user, so streaming
output filters out everything else.
"""


def extract_text(content) -> str:
    """Pull just the user-facing text out of a message's content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def stream_response(app, messages: list, config: dict = None) -> str:
    """Stream one turn of the graph to stdout and return the full text."""
    full_text = ""
    for chunk, metadata in app.stream(
        {"messages": messages},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "agent":
            continue  # skip tool-execution chunks, only print the agent's own tokens
        text = extract_text(chunk.content)
        if text:
            print(text, end="", flush=True)
            full_text += text
    print()  # newline once the stream finishes
    return full_text