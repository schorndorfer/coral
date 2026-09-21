"""Public interfaces for the frozen CORAL paper replication protocol."""

from .grid import build_request_grid, load_source
from .parsing import (
    parse_annotation_set,
    parse_namedtuple_expression,
    parse_paper_output,
    serialize_parsed_tuples,
)
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
from .scoring import (
    ArtifactPaths,
    ReplicationScores,
    format_relations,
    score_completed_records,
    write_artifacts,
)

__all__ = [
    "MAX_OUTPUT_TOKENS",
    "ArtifactPaths",
    "PAPER_GPT4",
    "PAPER_PREAMBLE",
    "PAPER_SOURCE_COMMIT",
    "REASONING_EFFORT",
    "ReplicationScores",
    "TASK_ORDER",
    "TASK_PROMPTS",
    "build_request_grid",
    "get_task_prompt",
    "format_relations",
    "load_source",
    "parse_annotation_set",
    "parse_namedtuple_expression",
    "parse_paper_output",
    "serialize_parsed_tuples",
    "score_completed_records",
    "write_artifacts",
]
