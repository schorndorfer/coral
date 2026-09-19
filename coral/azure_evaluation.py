"""Pure helpers for the Azure GPT-5.6 Sol evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

SOL_INPUT_PER_MILLION = 4.0
SOL_OUTPUT_PER_MILLION = 20.0
TERMINAL_STATUSES = {
    "valid",
    "valid_after_retry",
    "validation_failed",
    "api_failed",
    "spend_cap_reached",
}


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
