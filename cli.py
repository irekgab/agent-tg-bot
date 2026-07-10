"""Entry point: run the agent on a single prompt or interactively.

Usage:
    python main.py "your_prompt"        # single prompt
    python main.py                      # interactive loop
"""
import sys

from graph import build_graph
from streaming import stream_response
from logging_config import configure_logging


def run(prompt: str) -> str:
    app = build_graph()
    config = {"configurable": {"thread_id": "single-prompt-session"}}
    return stream_response(app, [{"role": "user", "content": prompt}], config=config)


def main() -> None:
    if len(sys.argv) > 1:
        print(run(" ".join(sys.argv[1:])))
        return

    configure_logging()
    app = build_graph()
    print("Agent ready. Type 'exit' to quit.")
    
    config = {"configurable": {"thread_id": "interactive-session"}}
    
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        
        stream_response(app, [{"role": "user", "content": user_input}], config=config)


if __name__ == "__main__":
    main()
