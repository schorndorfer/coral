"""Public interfaces for the frozen CORAL paper replication protocol."""

from .grid import build_request_grid, load_source
from .protocol import (
    MAX_OUTPUT_TOKENS,
    PAPER_GPT4,
    PAPER_PREAMBLE,
    PAPER_SOURCE_COMMIT,
    REASONING_EFFORT,
    TASK_ORDER,
    TASK_PROMPTS,
    get_task_prompt,
)

__all__ = [
    "MAX_OUTPUT_TOKENS",
    "PAPER_GPT4",
    "PAPER_PREAMBLE",
    "PAPER_SOURCE_COMMIT",
    "REASONING_EFFORT",
    "TASK_ORDER",
    "TASK_PROMPTS",
    "build_request_grid",
    "get_task_prompt",
    "load_source",
]
