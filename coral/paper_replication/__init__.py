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
    "PaperCheckpointRecord",
    "REASONING_EFFORT",
    "ReplicationScores",
    "TASK_ORDER",
    "TASK_PROMPTS",
    "build_request_grid",
    "checkpoint_spend",
    "completed_keys",
    "create_azure_client",
    "get_task_prompt",
    "format_relations",
    "load_azure_settings",
    "load_source",
    "parse_annotation_set",
    "parse_namedtuple_expression",
    "parse_paper_output",
    "project_grid_cost",
    "read_checkpoint",
    "run_replication",
    "serialize_parsed_tuples",
    "score_completed_records",
    "smoke_is_complete",
    "write_artifacts",
]
