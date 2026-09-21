"""Paper-compatible relation scoring for completed replication outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from coral import CancerDiagnosis, task_to_default_tuple_dict

from .parsing import (
    parse_annotation_set,
    parse_paper_output,
    serialize_parsed_tuples,
)
from .protocol import PAPER_GPT4


class MetricProtocol(Protocol):
    """Metric methods used by the original paper's relation scorer."""

    def compute_bleu_score(
        self,
        preds: list[str],
        references: list[list[str]],
        max_n: int,
        smooth: bool,
    ) -> dict[str, float]: ...

    def compute_rouge_score(
        self,
        preds: list[str],
        references: list[str],
        rouge_types: list[str],
    ) -> dict[str, float]: ...

    def compute_em_over_multiset_prec_recall_f1(
        self, outputs: list[str], annotations: list[str]
    ) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class ReplicationScores:
    """Completed outputs and their paper-compatible score tables."""

    outputs: pd.DataFrame
    instances: pd.DataFrame
    relations: pd.DataFrame
    topline: pd.DataFrame


@dataclass(frozen=True)
class ArtifactPaths:
    """Isolated paths used for paper-replication artifacts."""

    checkpoint: Path
    outputs: Path
    instances: Path
    relations: Path
    topline: Path


_RECORD_KEY_FIELDS = ("doc_idx", "section_name", "task", "model")
_INSTANCE_COLUMNS = [
    "doc_idx",
    "section_name",
    "task",
    "subrelation",
    "model",
    "bleu4",
    "rouge1",
    "em_precision",
    "em_recall",
    "em_f1",
]
_RELATION_COLUMNS = [
    "task",
    "model",
    "subrelation",
    "bleu4",
    "rouge1",
    "em_precision",
    "em_recall",
    "em_f1",
]
_TOPLINE_COLUMNS = ["metric", "gpt56_sol", "paper_gpt4", "difference_vs_paper"]


def format_relations(values: list[tuple]) -> dict[str, set[str]]:
    """Format named tuples into the relation values compared by the paper."""
    relations: dict[str, set[str]] = {}
    for value in values:
        if isinstance(value, CancerDiagnosis):
            continue
        if not hasattr(value, "_fields"):
            raise ValueError("relations must be named tuples")

        primary_type = value._fields[0]
        primary_value = value[0]
        for index, (relation_type, relation_value) in enumerate(value._asdict().items()):
            if index == 0 or relation_type == "AdditionalTesting":
                continue
            if len(relation_value) == 0:
                relation_value = "unknown"
            if isinstance(relation_value, dict):
                relation_value = set(relation_value.values())

            subrelation = f"{primary_type} {relation_type}"
            relation_values = relations.setdefault(subrelation, set())
            if isinstance(relation_value, (set, list)):
                for item in relation_value:
                    relation_values.add(f"{primary_value} {item}")
            else:
                relation_values.add(f"{primary_value} {relation_value}")

    if not relations:
        raise ValueError("no relations produced from named tuples")
    return relations


def _completed_record_key(record: dict[str, object]) -> tuple[str, str, str, str]:
    try:
        return tuple(str(record[field]) for field in _RECORD_KEY_FIELDS)  # type: ignore[return-value]
    except KeyError as error:
        raise ValueError(f"completed record is missing {error.args[0]}") from error


def _completed_unique_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    unique_records = []
    seen_keys = set()
    for record in records:
        if record.get("status") != "completed":
            continue
        key = _completed_record_key(record)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized = dict(record)
        normalized.update(dict(zip(_RECORD_KEY_FIELDS, key, strict=True)))
        normalized.pop("api_key", None)
        unique_records.append(normalized)
    return unique_records


def _source_annotation_values(source: pd.DataFrame, key: tuple[str, str, str, str]) -> list[tuple]:
    doc_idx, section_name, task, _model = key
    matching = source[
        (source["doc_idx"].astype(str) == doc_idx)
        & (source["section_name"].astype(str) == section_name)
        & (source["task"].astype(str) == task)
    ]
    if len(matching) > 1:
        raise ValueError(
            "duplicate source annotations for "
            f"doc_idx={doc_idx}, section_name={section_name}, task={task}"
        )
    if matching.empty:
        try:
            return [task_to_default_tuple_dict[task]]
        except KeyError as error:
            raise ValueError(f"unknown paper task: {task}") from error
    return parse_annotation_set(matching.iloc[0]["annotation_set"], task)


def _paper_metric_values(
    prediction_values: list[str], annotation_values: list[str], metrics: MetricProtocol
) -> tuple[float, float, float, float, float]:
    bleu4 = float(np.mean([
        metrics.compute_bleu_score(
            preds=[prediction], references=[annotation_values], max_n=4, smooth=True,
        )["bleu"]
        for prediction in prediction_values
    ]))
    rouge1 = float(np.mean([
        max(
            metrics.compute_rouge_score(
                preds=[prediction], references=[annotation], rouge_types=["rouge1"],
            )["rouge1"]
            for prediction in prediction_values
        )
        for annotation in annotation_values
    ]))
    em_precision, em_recall, em_f1 = metrics.compute_em_over_multiset_prec_recall_f1(
        prediction_values, annotation_values
    )
    return bleu4, rouge1, em_precision, em_recall, em_f1


def _relation_scores(instances: pd.DataFrame) -> pd.DataFrame:
    if instances.empty:
        return pd.DataFrame(columns=_RELATION_COLUMNS)
    return (
        instances.groupby(["task", "model", "subrelation"], as_index=False)[
            ["bleu4", "rouge1", "em_precision", "em_recall", "em_f1"]
        ]
        .mean()
        .loc[:, _RELATION_COLUMNS]
    )


def _topline_scores(relations: pd.DataFrame) -> pd.DataFrame:
    target = relations[relations["model"] == "gpt-5.6-sol"]
    metric_columns = (("BLEU-4", "bleu4"), ("ROUGE-1", "rouge1"), ("EM F1", "em_f1"))
    records = []
    for label, column in metric_columns:
        score = float(target[column].mean()) if not target.empty else float("nan")
        paper_score = PAPER_GPT4[label]
        records.append({
            "metric": label,
            "gpt56_sol": round(score, 2),
            "paper_gpt4": paper_score,
            "difference_vs_paper": round(score - paper_score, 2),
        })
    return pd.DataFrame(records, columns=_TOPLINE_COLUMNS)


def score_completed_records(
    source: pd.DataFrame,
    records: list[dict[str, object]],
    metrics: MetricProtocol | None = None,
) -> ReplicationScores:
    """Score only actual completed API outputs using the paper's relation loop."""
    completed = _completed_unique_records(records)
    output_records = []
    instance_records = []
    active_metrics = metrics

    for record in completed:
        key = _completed_record_key(record)
        doc_idx, section_name, task, model = key
        parsed_output = parse_paper_output(record.get("output_text"), task)
        output_records.append({
            **record,
            "parsed_output_json": serialize_parsed_tuples(parsed_output),
        })
        annotation_values = _source_annotation_values(source, key)
        predicted_relations = format_relations(parsed_output)
        annotated_relations = format_relations(annotation_values)

        if active_metrics is None:
            from coral.utils.metrics import Metrics

            active_metrics = Metrics(tokenizer="default")

        for subrelation, predictions in predicted_relations.items():
            prediction_values = sorted(value.lower() for value in predictions)
            gold_values = sorted(
                value.lower() for value in annotated_relations[subrelation]
            )
            bleu4, rouge1, em_precision, em_recall, em_f1 = _paper_metric_values(
                prediction_values, gold_values, active_metrics
            )
            instance_records.append({
                "doc_idx": doc_idx,
                "section_name": section_name,
                "task": task,
                "subrelation": subrelation,
                "model": model,
                "bleu4": bleu4,
                "rouge1": rouge1,
                "em_precision": em_precision,
                "em_recall": em_recall,
                "em_f1": em_f1,
            })

    outputs = pd.DataFrame(output_records)
    instances = pd.DataFrame(instance_records, columns=_INSTANCE_COLUMNS)
    relations = _relation_scores(instances)
    return ReplicationScores(
        outputs=outputs,
        instances=instances,
        relations=relations,
        topline=_topline_scores(relations),
    )


def write_artifacts(
    scores: ReplicationScores,
    output_dir: Path,
    prefix: str = "gpt56_sol_paper_replication",
) -> ArtifactPaths:
    """Write score tables without altering the checkpoint JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(
        checkpoint=output_dir / f"{prefix}.jsonl",
        outputs=output_dir / f"{prefix}_outputs.csv",
        instances=output_dir / f"{prefix}_instance_scores.csv",
        relations=output_dir / f"{prefix}_relation_scores.csv",
        topline=output_dir / f"{prefix}_topline.csv",
    )
    scores.outputs.to_csv(paths.outputs, index=False)
    scores.instances.to_csv(paths.instances, index=False)
    scores.relations.to_csv(paths.relations, index=False)
    scores.topline.to_csv(paths.topline, index=False)
    return paths
