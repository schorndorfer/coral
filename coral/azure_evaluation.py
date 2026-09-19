"""Pure helpers for the Azure GPT-5.6 Sol evaluation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from coral import task_to_default_tuple_dict

SOL_INPUT_PER_MILLION = 4.0
SOL_OUTPUT_PER_MILLION = 20.0
API_ENVELOPE_TOKEN_MARGIN = 256
TERMINAL_STATUSES = {
    "valid",
    "valid_after_retry",
    "validation_failed",
    "api_failed",
    "spend_cap_reached",
}

# These names deliberately follow the JSON response contract rather than the
# legacy namedtuple spelling.  The ordering is the legacy tuple field order.
TASK_FIELDS: dict[str, tuple[str, ...]] = {
    "symptoms": ("symptom", "datetimes"),
    "symptoms_at_diagnosis": ("symptom", "datetimes"),
    "symptoms_due_to_cancer": ("symptom", "datetimes"),
    "radtest_datetime_site_reason_result": (
        "radiology_test", "datetimes", "sites", "reasons", "results",
    ),
    "procedure_datetime_site_reason_result": (
        "procedure_name", "datetimes", "sites", "reasons", "results",
    ),
    "biomarker_datetime": ("biomarker", "datetimes"),
    "histology_datetime": ("histology", "datetimes"),
    "metastasis_site_procedure_datetime": (
        "metastasis", "sites", "procedures", "datetimes",
    ),
    "stage_datetime_addtest": ("stage", "datetimes", "additional_testing"),
    "tnm_datetime_addtest": ("tnm", "datetimes", "additional_testing"),
    "grade_datetime_addtest": ("grade", "datetimes", "additional_testing"),
    "prescribed_med_begin_end_reason_continuity_ae": (
        "medication_name", "begins", "ends", "reasons", "continuity",
        "confirmed_adverse_events", "potential_adverse_events",
    ),
    "future_med_consideration_ae": (
        "medication_name", "consideration", "potential_adverse_events",
    ),
    "genomictest_datetime_result": (
        "genomic_test_name", "datetimes", "results",
    ),
}

_SCALAR_FIELDS = {"symptom", "radiology_test", "procedure_name", "biomarker", "histology", "metastasis", "stage", "tnm", "grade", "medication_name", "continuity", "consideration", "genomic_test_name"}


def _require_task(task: str) -> tuple[str, ...]:
    if task not in TASK_FIELDS or task not in task_to_default_tuple_dict:
        raise ValueError(f"unknown task: {task}")
    return TASK_FIELDS[task]


def _field_schema(field: str) -> dict[str, object]:
    if field in _SCALAR_FIELDS:
        return {"type": "string"}
    return {"type": "array", "items": {"type": "string"}}


def build_request(row: Mapping[str, str]) -> tuple[str, dict[str, object]]:
    """Build the task-specific instruction and strict Responses API JSON format."""
    task = row.get("task", "")
    fields = _require_task(task)
    record_properties = {field: _field_schema(field) for field in fields}
    record_properties["evidence_quotes"] = {
        "type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1,
    }
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["task", "records"],
        "properties": {
            "task": {"type": "string", "const": task},
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [*fields, "evidence_quotes"],
                    "properties": record_properties,
                },
            },
        },
    }
    prompt = (
        f"Extract {task} records from the supplied clinical section. "
        "Return exactly one JSON object matching the requested schema. "
        "Every record must include one or more evidence_quotes copied verbatim from the section.\n\n"
        f"Clinical section:\n{row.get('section_text', '')}"
    )
    return prompt, {
        "type": "json_schema",
        "name": f"coral_{task}",
        "strict": True,
        "schema": schema,
    }


def _normalized(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _validate_record(fields: tuple[str, ...], record: object, section_text: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    expected = {*fields, "evidence_quotes"}
    if set(record) != expected:
        raise ValueError("record has missing or unexpected fields")
    validated: dict[str, object] = {}
    for field in fields:
        value = record[field]
        if field in _SCALAR_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
        elif not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be a list of strings")
        validated[field] = value
    quotes = record["evidence_quotes"]
    if not isinstance(quotes, list) or not quotes or not all(isinstance(quote, str) and quote for quote in quotes):
        raise ValueError("evidence_quotes must be a non-empty list of strings")
    normalized_section = _normalized(section_text)
    if any(_normalized(quote) not in normalized_section for quote in quotes):
        raise ValueError("evidence_quotes must appear verbatim in section text")
    validated["evidence_quotes"] = quotes
    return validated


def validate_response(task: str, section_text: str, raw: str) -> list[dict[str, object]]:
    """Safely parse and validate a model response against its task contract."""
    fields = _require_task(task)
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("response must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"task", "records"}:
        raise ValueError("response must contain only task and records")
    if payload["task"] != task:
        raise ValueError("response task does not match requested task")
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    return [_validate_record(fields, record, section_text) for record in records]


def _serialize_set(values: object) -> str:
    if not isinstance(values, (list, tuple, set)) or not all(isinstance(value, str) for value in values):
        raise ValueError("legacy set fields must be collections of strings")
    rendered = sorted({repr(value) for value in values})
    # parse_output's legacy regex needs non-empty brace content; unpacking an
    # empty list is an evaluable empty-set expression that satisfies it.
    return "{*[]}" if not rendered else "{" + ", ".join(rendered) + "}"


def to_legacy_output(task: str, records: list[dict[str, object]]) -> str:
    """Render validated JSON records as deterministic legacy namedtuple calls."""
    fields = _require_task(task)
    default = task_to_default_tuple_dict[task]
    if not records:
        return repr(default)
    tuple_fields = default._fields
    rendered_records: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not set(fields).issubset(record):
            raise ValueError("record is missing required task fields")
        values = []
        for json_field, tuple_field in zip(fields, tuple_fields, strict=True):
            value = record[json_field]
            serialized = repr(value) if json_field in _SCALAR_FIELDS else _serialize_set(value)
            values.append(f"{tuple_field}={serialized}")
        rendered_records.append(f"{type(default).__name__}({', '.join(values)})")
    return ", ".join(rendered_records)


@dataclass(frozen=True, repr=False)
class AzureSettings:
    api_key: str = field(repr=False)
    endpoint: str = ""
    deployment: str = "gpt-5.6-sol"
    api_version: str = "2025-04-01-preview"

    def __repr__(self) -> str:
        return (
            "AzureSettings(api_key='[REDACTED]', "
            f"endpoint={self.endpoint!r}, deployment={self.deployment!r}, "
            f"api_version={self.api_version!r})"
        )


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")


@dataclass(frozen=True)
class CheckpointRecord:
    doc_idx: str
    section_name: str
    task: str
    model: str
    validation_status: str
    output_text: str | None = None
    parsed_records: list[dict[str, Any]] | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    elapsed_seconds: float | None = None
    raw_attempts: list[str] | None = None


class ResponsesAPI(Protocol):
    def create(self, **kwargs: object) -> object: ...


class ResponseClient(Protocol):
    """The small Responses API surface the evaluation runner needs."""

    responses: ResponsesAPI


def _response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(response, Mapping):
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
    raise ValueError("response did not include output_text")


def _response_usage(response: object) -> Usage:
    usage = getattr(response, "usage", None)
    if isinstance(response, Mapping):
        usage = response.get("usage", usage)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens", input_tokens)
        output_tokens = usage.get("output_tokens", output_tokens)
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise ValueError("response usage must include integer input_tokens and output_tokens")
    return Usage(input_tokens, output_tokens)


def _projected_cost(request_input: str, max_output_tokens: int) -> float:
    if max_output_tokens < 0:
        raise ValueError("max_output_tokens cannot be negative")
    # Input includes the exact prompt. This fixed margin covers Responses API
    # framing plus the structured-output schema that is sent outside input.
    estimated_input_tokens = max(1, math.ceil(len(request_input) / 4)) + API_ENVELOPE_TOKEN_MARGIN
    return estimate_cost(Usage(estimated_input_tokens, max_output_tokens))


def _checkpoint_spend(path: Path) -> float:
    if not path.exists():
        return 0.0
    spent = 0.0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cost = record.get("cost") if isinstance(record, dict) else None
            if isinstance(cost, (int, float)) and cost >= 0:
                spent += cost
    return spent


def _row_value(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name, "")
    return value if isinstance(value, str) else str(value)


def run_evaluation(
    rows: pd.DataFrame,
    client: ResponseClient,
    settings: AzureSettings,
    checkpoint_path: Path,
    spend_cap: float,
    max_output_tokens: int,
    reasoning_effort: str,
) -> pd.DataFrame:
    """Run rows serially, checkpointing every terminal outcome for safe resume."""
    if spend_cap < 0:
        raise ValueError("spend_cap cannot be negative")
    completed = terminal_keys(checkpoint_path)
    spent = _checkpoint_spend(checkpoint_path)
    results: list[dict[str, Any]] = []

    for row in rows.to_dict("records"):
        doc_idx = _row_value(row, "doc_idx")
        section_name = _row_value(row, "section_name")
        task = _row_value(row, "task")
        section_text = _row_value(row, "section_text")
        key = (doc_idx, section_name, task, settings.deployment)
        base = {
            "doc_idx": doc_idx,
            "section_name": section_name,
            "task": task,
            "model": settings.deployment,
        }
        if key in completed:
            results.append({**base, "validation_status": "skipped_on_resume"})
            continue

        prompt, response_format = build_request(row)
        attempts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        attempt_cost = 0.0
        output_text: str | None = None
        parsed_records: list[dict[str, object]] | None = None
        started = time.perf_counter()
        error: str | None = None
        stop_due_cap = False
        for attempt_number in range(2):
            request_input = prompt
            if attempt_number:
                request_input = (
                    f"{prompt}\n\nYour previous response was invalid: {error}. "
                    "Correct it and return only schema-compliant JSON."
                )
            if not can_afford(
                spent, _projected_cost(request_input, max_output_tokens), spend_cap
            ):
                record = CheckpointRecord(
                    **base, validation_status="spend_cap_reached", output_text=output_text,
                    input_tokens=input_tokens, output_tokens=output_tokens, cost=attempt_cost,
                    elapsed_seconds=time.perf_counter() - started, raw_attempts=attempts,
                )
                append_checkpoint(checkpoint_path, record)
                completed.add(key)
                results.append(asdict(record))
                stop_due_cap = True
                break
            try:
                response = client.responses.create(
                    model=settings.deployment,
                    input=request_input,
                    reasoning={"effort": reasoning_effort},
                    max_output_tokens=max_output_tokens,
                    text={"format": response_format},
                )
            except Exception as exception:  # External client failures are terminal and safe to resume.
                elapsed = time.perf_counter() - started
                record = CheckpointRecord(
                    **base, validation_status="api_failed",
                    error=f"{type(exception).__name__}: {exception}",
                    input_tokens=input_tokens, output_tokens=output_tokens, cost=attempt_cost,
                    elapsed_seconds=elapsed, raw_attempts=attempts,
                )
                append_checkpoint(checkpoint_path, record)
                completed.add(key)
                results.append(asdict(record))
                break
            try:
                raw = _response_text(response)
                attempts.append(raw)
                output_text = raw
                usage = _response_usage(response)
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
                cost = estimate_cost(usage)
                attempt_cost += cost
                spent += cost
                parsed_records = validate_response(task, section_text, raw)
            except ValueError as validation_error:
                error = str(validation_error)
                if attempt_number == 0:
                    continue
                elapsed = time.perf_counter() - started
                record = CheckpointRecord(
                    **base, validation_status="validation_failed", error=error,
                    output_text=output_text, input_tokens=input_tokens,
                    output_tokens=output_tokens, cost=attempt_cost,
                    elapsed_seconds=elapsed, raw_attempts=attempts,
                )
                append_checkpoint(checkpoint_path, record)
                completed.add(key)
                results.append(asdict(record))
                break
            else:
                record = CheckpointRecord(
                    **base,
                    validation_status="valid" if attempt_number == 0 else "valid_after_retry",
                    output_text=output_text, parsed_records=parsed_records,
                    input_tokens=input_tokens, output_tokens=output_tokens, cost=attempt_cost,
                    elapsed_seconds=time.perf_counter() - started, raw_attempts=attempts,
                )
                append_checkpoint(checkpoint_path, record)
                completed.add(key)
                results.append(asdict(record))
                break
        if stop_due_cap:
            break
    return pd.DataFrame(results)


def load_azure_settings(env: Mapping[str, str]) -> AzureSettings:
    api_key = env.get("AZURE_OPENAI_API_KEY", "")
    endpoint = env.get("AZURE_OPENAI_ENDPOINT", "")
    if not api_key.strip():
        raise ValueError("AZURE_OPENAI_API_KEY is required")
    if not endpoint.strip():
        raise ValueError("AZURE_OPENAI_ENDPOINT is required")

    parsed = urlsplit(endpoint.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("AZURE_OPENAI_ENDPOINT must be an absolute URL")
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    deployment = env.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-sol").strip()
    api_version = env.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview").strip()
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT cannot be empty")
    if not api_version:
        raise ValueError("AZURE_OPENAI_API_VERSION cannot be empty")
    return AzureSettings(api_key, endpoint, deployment, api_version)


def estimate_cost(usage: Usage) -> float:
    return usage.input_tokens / 1_000_000 * SOL_INPUT_PER_MILLION + usage.output_tokens / 1_000_000 * SOL_OUTPUT_PER_MILLION


def can_afford(spent: float, projected: float, cap: float) -> bool:
    if spent < 0 or projected < 0:
        raise ValueError("spent and projected costs cannot be negative")
    if cap < 0:
        raise ValueError("cap cannot be negative")
    return spent + projected <= cap


def append_checkpoint(path: Path, record: CheckpointRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def terminal_keys(path: Path) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict) or record.get("validation_status") not in TERMINAL_STATUSES:
                continue
            values = (record.get("doc_idx"), record.get("section_name"), record.get("task"), record.get("model"))
            if all(isinstance(value, str) for value in values):
                keys.add(values)  # type: ignore[arg-type]
    return keys
