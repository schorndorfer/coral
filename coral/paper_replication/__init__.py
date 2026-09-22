"""Public interfaces for the frozen CORAL paper replication protocol."""

from .grid import build_request_grid, load_source
from .parsing import (
    parse_annotation_set,
    parse_namedtuple_expression,
    parse_paper_output,
    parse_paper_source_output,
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
from .runner import (
    PaperCheckpointRecord,
    checkpoint_spend,
    completed_keys,
    create_azure_client,
    load_azure_settings,
    project_grid_cost,
    read_checkpoint,
    run_replication,
    smoke_is_complete,
    smoke_is_scored,
    write_smoke_marker,
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
    "PAPER_GPT4",
    "PAPER_PREAMBLE",
    "PAPER_SOURCE_COMMIT",
    "REASONING_EFFORT",
    "TASK_ORDER",
    "TASK_PROMPTS",
    "ArtifactPaths",
    "PaperCheckpointRecord",
    "ReplicationScores",
    "build_request_grid",
    "checkpoint_spend",
    "completed_keys",
    "create_azure_client",
    "format_relations",
    "get_task_prompt",
    "load_azure_settings",
    "load_source",
    "parse_annotation_set",
    "parse_namedtuple_expression",
    "parse_paper_output",
    "parse_paper_source_output",
    "project_grid_cost",
    "read_checkpoint",
    "run_replication",
    "score_completed_records",
    "serialize_parsed_tuples",
    "smoke_is_complete",
    "smoke_is_scored",
    "write_artifacts",
    "write_smoke_marker",
]
