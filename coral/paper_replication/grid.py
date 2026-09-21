"""Source-data validation and deterministic request-grid construction."""

from pathlib import Path

import pandas as pd

from .protocol import PAPER_PREAMBLE, TASK_ORDER, TASK_PROMPTS


REQUIRED_COLUMNS = {
    "doc_idx", "section_name", "section_text", "task", "annotation_set",
}
EXPECTED_SECTIONS = ("hpi", "a&p")


def load_source(path: Path) -> pd.DataFrame:
    """Load paper inference data with its task and document identifiers normalized."""
    frame = pd.read_csv(path).rename(columns={"inference_subtype": "task"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"source data is missing columns: {', '.join(sorted(missing))}"
        )
    frame = frame.copy()
    frame["doc_idx"] = frame["doc_idx"].astype(str)
    return frame


def _sorted_document_ids(document_ids: set[str]) -> list[str]:
    if all(document_id.isdecimal() for document_id in document_ids):
        return sorted(document_ids, key=int)
    return sorted(document_ids)


def build_request_grid(source: pd.DataFrame) -> pd.DataFrame:
    """Validate paper source data and expand every document section over all tasks."""
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(
            f"source data is missing columns: {', '.join(sorted(missing))}"
        )

    frame = source.copy()
    frame["doc_idx"] = frame["doc_idx"].astype(str)
    section_text = frame["section_text"]
    if section_text.isna().any() or section_text.astype(str).str.strip().eq("").any():
        raise ValueError("source data contains null or blank section text")

    section_text_counts = frame.groupby(
        ["doc_idx", "section_name"], dropna=False
    )["section_text"].nunique(dropna=False)
    if (section_text_counts > 1).any():
        raise ValueError("source data contains conflicting section text")

    document_ids = set(frame["doc_idx"])
    if len(document_ids) != 40:
        raise ValueError("source data must contain exactly 40 documents")

    sections_by_document = frame.groupby("doc_idx", dropna=False)["section_name"].agg(set)
    expected_section_set = set(EXPECTED_SECTIONS)
    if not sections_by_document.map(lambda sections: sections == expected_section_set).all():
        raise ValueError("each document must contain exactly hpi and a&p sections")

    if set(frame["task"]) != set(TASK_ORDER):
        raise ValueError("source task set does not match the paper task set")

    unique_sections = frame.drop_duplicates(
        ["doc_idx", "section_name"], keep="first"
    ).set_index(["doc_idx", "section_name"])["section_text"]
    records = []
    for doc_idx in _sorted_document_ids(document_ids):
        for section_name in EXPECTED_SECTIONS:
            section = unique_sections.loc[(doc_idx, section_name)]
            for task in TASK_ORDER:
                task_prompt = TASK_PROMPTS[task]
                records.append(
                    {
                        "doc_idx": doc_idx,
                        "section_name": section_name,
                        "section_text": section,
                        "task": task,
                        "task_prompt": task_prompt,
                        "instructions": PAPER_PREAMBLE,
                        "request_input": section + task_prompt,
                    }
                )
    return pd.DataFrame(
        records,
        columns=[
            "doc_idx",
            "section_name",
            "section_text",
            "task",
            "task_prompt",
            "instructions",
            "request_input",
        ],
    )
