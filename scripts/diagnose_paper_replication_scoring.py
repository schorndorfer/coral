"""Safely identify a paper-replication record that cannot be scored.

This diagnostic reads local source and checkpoint files only. It does not
create an Azure client, make API calls, or write evaluation artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

import pandas as pd

from coral.paper_replication import (
    load_source,
    read_checkpoint,
    score_completed_records,
)
from coral.paper_replication.scoring import MetricProtocol
from coral.utils.metrics import Metrics

MODEL = "gpt-5.6-sol"
SOURCE_PATH = Path("data/coral_inference.csv")
CHECKPOINT_PATH = Path("output/gpt56_sol_paper_replication.jsonl")


def diagnose_records(
    source: pd.DataFrame,
    records: list[dict[str, object]],
    *,
    metrics: MetricProtocol,
    model: str,
    output: TextIO = sys.stdout,
) -> int:
    """Score records separately and report only safe failure metadata."""
    print(f"Testing {len(records)} completed records", file=output)
    for number, record in enumerate(records, start=1):
        try:
            score_completed_records(source, [record], metrics=metrics, model=model)
        # The notebook catches every scoring/artifact exception. Mirror that
        # boundary here while revealing only its class, never its message.
        except Exception as error:  # noqa: BLE001
            print(
                "FAILED:",
                number,
                record.get("doc_idx", "<missing>"),
                record.get("section_name", "<missing>"),
                record.get("task", "<missing>"),
                type(error).__name__,
                file=output,
            )
            return 1

        if number % 100 == 0:
            print(f"Checked {number}/{len(records)}", file=output)

    print("All individual records scored successfully", file=output)
    return 0


def main() -> int:
    """Load the default replication files and run the safe diagnostic."""
    source = load_source(SOURCE_PATH)
    records = [
        record
        for record in read_checkpoint(CHECKPOINT_PATH)
        if record.get("status") == "completed" and record.get("model") == MODEL
    ]
    metrics = Metrics(tokenizer="default")
    return diagnose_records(
        source,
        records,
        metrics=metrics,
        model=MODEL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
