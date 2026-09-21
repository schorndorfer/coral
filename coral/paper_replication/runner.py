"""Cost-capped Responses API execution for the frozen paper protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping, cast
from urllib.parse import urlsplit

import pandas as pd

from coral.azure_evaluation import (
    AzureSettings,
    ResponseClient,
    Usage,
    estimate_cost,
    load_azure_settings,
)
from .protocol import MAX_OUTPUT_TOKENS, REASONING_EFFORT


@dataclass(frozen=True)
class PaperCheckpointRecord:
    doc_idx: str
    section_name: str
    task: str
    model: str
    status: str
    output_text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    elapsed_seconds: float = 0.0
    error: str | None = None


def create_azure_client(
    settings: AzureSettings,
    openai_class: Callable[..., object] | None = None,
    azure_class: Callable[..., object] | None = None,
) -> ResponseClient:
    """Select by hostname, preserving a Foundry base URL byte-for-byte."""
    hostname = (urlsplit(settings.endpoint).hostname or "").lower()
    if hostname.endswith(".services.ai.azure.com"):
        if openai_class is None:
            from openai import OpenAI

            openai_class = OpenAI
        client = openai_class(base_url=settings.endpoint, api_key=settings.api_key)
    else:
        if azure_class is None:
            from openai import AzureOpenAI

            azure_class = AzureOpenAI
        client = azure_class(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
        )
    # SDK retries otherwise multiply the runner's attempts and bypass its cap check.
    with_options = getattr(client, "with_options", None)
    if callable(with_options):
        client = with_options(max_retries=0)
    return cast(ResponseClient, client)


def read_checkpoint(path: Path) -> list[dict[str, object]]:
    """Read valid JSON objects, tolerating interrupted or malformed JSONL lines."""
    if not path.exists():
        return []
    records = []
    with path.open("rb") as stream:
        for line in stream:
            try:
                record = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def completed_keys(
    records: Iterable[Mapping[str, object]],
) -> set[tuple[str, str, str, str]]:
    """Only real completed responses prevent another request on resume."""
    keys = set()
    for record in records:
        if record.get("status") != "completed":
            continue
        key = tuple(record.get(name) for name in ("doc_idx", "section_name", "task", "model"))
        if all(isinstance(value, str) for value in key):
            keys.add(cast(tuple[str, str, str, str], key))
    return keys


def checkpoint_spend(records: Iterable[Mapping[str, object]]) -> float:
    """Sum all known actual costs, including earlier attempts and deployments."""
    spent = 0.0
    for record in records:
        cost = record.get("cost")
        if (
            isinstance(cost, (int, float)) and not isinstance(cost, bool)
            and math.isfinite(cost) and cost >= 0
        ):
            spent += cost
    return spent


def _projected_cost(instructions: str, request_input: str) -> float:
    input_tokens = max(1, math.ceil((len(instructions) + len(request_input)) / 4)) + 256
    return estimate_cost(Usage(input_tokens, MAX_OUTPUT_TOKENS))


def project_grid_cost(grid: pd.DataFrame) -> float:
    """Project every request using its exact prompt and the fixed output ceiling."""
    return sum(
        (_projected_cost(row.instructions, row.request_input) for row in grid.itertuples(index=False)),
        0.0,
    )


def smoke_is_complete(
    grid: pd.DataFrame, records: Iterable[Mapping[str, object]], model: str,
) -> bool:
    required = {
        (str(row.doc_idx), str(row.section_name), str(row.task), model)
        for row in grid.head(8).itertuples(index=False)
    }
    return required.issubset(completed_keys(records))


def _append_checkpoint(path: Path, record: PaperCheckpointRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":"))
    with path.open("ab+") as stream:
        # Separate a prior interrupted tail (or a valid line without a newline)
        # from the new record, keeping resumed completions readable thereafter.
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(payload.encode("utf-8") + b"\n")


def _response_result(response: object) -> tuple[str, Usage]:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str):
        raise ValueError("response did not include output_text")
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (input_tokens, output_tokens)):
        raise ValueError("response usage must include integer input_tokens and output_tokens")
    return output_text, Usage(input_tokens, output_tokens)


def _sanitized_error(error: Exception) -> str:
    """Keep diagnostic metadata only; exception messages may echo clinical text."""
    details = type(error).__name__
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        details += f" (status_code={int(status_code)})"
    return details[:500]


def run_replication(
    grid: pd.DataFrame,
    client: ResponseClient,
    settings: AzureSettings,
    checkpoint_path: Path,
    spend_cap: float,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Run serially, retrying identical transient requests and retaining resumability."""
    if not math.isfinite(spend_cap) or spend_cap < 0:
        raise ValueError("spend_cap must be finite and nonnegative")
    records = read_checkpoint(checkpoint_path)
    completed = completed_keys(records)
    spent = checkpoint_spend(records)
    results = []
    delays = (1.0, 2.0)
    for ordinal, row in enumerate(grid.to_dict("records"), start=1):
        identity = {
            "doc_idx": str(row["doc_idx"]), "section_name": str(row["section_name"]),
            "task": str(row["task"]), "model": settings.deployment,
        }
        key = (identity["doc_idx"], identity["section_name"], identity["task"], identity["model"])
        if key in completed:
            record = PaperCheckpointRecord(**identity, status="skipped_on_resume")
        else:
            request = {
                "model": settings.deployment,
                "instructions": row["instructions"],
                "input": row["request_input"],
                "reasoning": {"effort": REASONING_EFFORT},
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
            started = time.perf_counter()
            for attempt in range(3):
                projected = _projected_cost(request["instructions"], request["input"])
                if spent + projected > spend_cap:
                    record = PaperCheckpointRecord(
                        **identity, status="spend_cap_reached",
                        elapsed_seconds=time.perf_counter() - started,
                    )
                    break
                try:
                    response = client.responses.create(**request)
                    output_text, usage = _response_result(response)
                except Exception as error:
                    status_code = getattr(error, "status_code", None)
                    transient = isinstance(status_code, int) and (
                        status_code in (408, 409, 429) or 500 <= status_code <= 599
                    )
                    if transient and attempt < 2:
                        sleep(delays[attempt])
                        continue
                    record = PaperCheckpointRecord(
                        **identity, status="api_failed",
                        elapsed_seconds=time.perf_counter() - started,
                        error=_sanitized_error(error),
                    )
                    break
                cost = estimate_cost(usage)
                spent += cost
                record = PaperCheckpointRecord(
                    **identity, status="completed", output_text=output_text,
                    input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                    cost=cost, elapsed_seconds=time.perf_counter() - started,
                )
                completed.add(key)
                break
            _append_checkpoint(checkpoint_path, record)
        results.append(asdict(record))
        if progress is not None:
            progress(
                f"{ordinal}/{len(grid)}: doc {record.doc_idx}, {record.section_name}, "
                f"{record.task}, {record.status}, spend ${spent:.6f}, {checkpoint_path}"
            )
        if record.status == "spend_cap_reached":
            break
    return pd.DataFrame(results, columns=PaperCheckpointRecord.__dataclass_fields__)
