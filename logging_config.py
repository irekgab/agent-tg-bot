"""Quiets known-noisy, harmless third-party loggers.

Combining Gemini's built-in google_search tool with our custom function
tools disables Gemini's automatic function-calling (AFC) fast path.
Two separate libraries log about this, and each needs its own logger
silenced since Python's logging hierarchy is per-package:

- langchain_google_genai._function_utils logs a WARNING for every
  JSON-schema key (title, default, $defs, anyOf, etc.) it strips when
  building FunctionDeclarations manually, because Gemini's schema
  format doesn't support them.
- The raw google-genai SDK itself logs that AFC is disabled once a
  non-callable tool (like our google_search dict) appears alongside
  function tools. This specific check lives in google_genai.models, so
  it's targeted directly rather than relying on inheriting the level
  set on the parent "google_genai" logger, in case that submodule sets
  its own level explicitly (which would override an inherited one).

Both are expected and harmless — they never affect tool-calling
behavior — so we raise their log levels to hide the noise without
hiding real errors from other loggers.
"""
import logging


def configure_logging() -> None:
    logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
    logging.getLogger("google_genai").setLevel(logging.ERROR)