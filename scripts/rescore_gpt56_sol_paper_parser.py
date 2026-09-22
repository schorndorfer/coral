"""Rescore checkpointed GPT-5.6 Sol outputs with the paper-style safe parser."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import pandas as pd

from coral.paper_replication import (
    ArtifactPaths,
    load_source,
    parse_paper_output,
    parse_paper_source_output,
    read_checkpoint,
    score_completed_records,
    serialize_parsed_tuples,
    write_artifacts,
)
from coral.paper_replication.scoring import MetricProtocol
from coral.utils.metrics import Metrics

MODEL = "gpt-5.6-sol"
SOURCE_PATH = Path("data/coral_inference.csv")
CHECKPOINT_PATH = Path("output/gpt56_sol_paper_replication.jsonl")
OUTPUT_DIR = Path("output")
CURRENT_PREFIX = "gpt56_sol_paper_replication"
PAPER_PARSER_PREFIX = "gpt56_sol_paper_replication_paper_parser"
COMPARISON_NAME = "gpt56_sol_paper_replication_parser_comparison.csv"


@dataclass(frozen=True)
class ParserRescoreResult:
    """Artifacts and aggregate diagnostics from one offline rescore."""

    artifacts: ArtifactPaths
    comparison: pd.DataFrame
    comparison_path: Path
    changed_records: int


def _changed_parse_count(records: list[dict[str, object]]) -> int:
    changed = 0
    for record in records:
        task = str(record.get("task", ""))
        current = serialize_parsed_tuples(
            parse_paper_output(record.get("output_text"), task)
        )
        paper_source = serialize_parsed_tuples(
            parse_paper_source_output(record.get("output_text"), task)
        )
        changed += current != paper_source
    return changed


def _build_comparison(
    current_topline: pd.DataFrame, paper_parser_topline: pd.DataFrame
) -> pd.DataFrame:
    current = current_topline.loc[
        :, ["metric", "gpt56_sol", "difference_vs_paper"]
    ].rename(columns={
        "gpt56_sol": "current_parser",
        "difference_vs_paper": "current_difference_vs_paper",
    })
    paper_parser = paper_parser_topline.loc[
        :, ["metric", "gpt56_sol", "paper_gpt4", "difference_vs_paper"]
    ].rename(columns={
        "gpt56_sol": "paper_source_parser",
        "difference_vs_paper": "paper_source_difference_vs_paper",
    })
    comparison = current.merge(paper_parser, on="metric", validate="one_to_one")
    comparison["parser_difference"] = (
        comparison["paper_source_parser"] - comparison["current_parser"]
    ).round(2)
    return comparison.loc[:, [
        "metric",
        "current_parser",
        "paper_source_parser",
        "parser_difference",
        "paper_gpt4",
        "current_difference_vs_paper",
        "paper_source_difference_vs_paper",
    ]]


def rescore_records(
    source: pd.DataFrame,
    records: list[dict[str, object]],
    *,
    output_dir: Path,
    metrics: MetricProtocol,
    model: str = MODEL,
    output: TextIO = sys.stdout,
) -> ParserRescoreResult:
    """Rescore completed records without changing current artifacts or using Azure."""
    current_topline_path = output_dir / f"{CURRENT_PREFIX}_topline.csv"
    current_topline = pd.read_csv(current_topline_path)
    selected_records = [
        record
        for record in records
        if record.get("status") == "completed" and record.get("model") == model
    ]
    scores = score_completed_records(
        source,
        selected_records,
        metrics=metrics,
        model=model,
        output_parser=parse_paper_source_output,
    )
    artifacts = write_artifacts(scores, output_dir, prefix=PAPER_PARSER_PREFIX)
    comparison = _build_comparison(current_topline, scores.topline)
    comparison_path = output_dir / COMPARISON_NAME
    comparison.to_csv(comparison_path, index=False)
    changed_records = _changed_parse_count(selected_records)

    print(
        f"Paper-parser rescore completed for {len(selected_records)} records. "
        f"Changed parses: {changed_records}/{len(selected_records)}",
        file=output,
    )
    print(comparison.to_string(index=False), file=output)
    print(f"Saved comparison: {comparison_path}", file=output)
    print(f"Saved paper-parser top line: {artifacts.topline}", file=output)
    return ParserRescoreResult(
        artifacts=artifacts,
        comparison=comparison,
        comparison_path=comparison_path,
        changed_records=changed_records,
    )


def main() -> int:
    """Load completed local artifacts and run the offline parser rescore."""
    source = load_source(SOURCE_PATH)
    records = read_checkpoint(CHECKPOINT_PATH)
    metrics = Metrics(tokenizer="default")
    rescore_records(
        source,
        records,
        output_dir=OUTPUT_DIR,
        metrics=metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
