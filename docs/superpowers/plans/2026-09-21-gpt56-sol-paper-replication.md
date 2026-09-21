# GPT-5.6 Sol Paper-Protocol Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, resumable GPT-5.6 Sol evaluation that sends all 1,120 CORAL paper-protocol requests and reports paper-compatible BLEU-4, ROUGE-1, and EM F1 against the published GPT-4 top line.

**Architecture:** Add a focused `coral.paper_replication` package containing the frozen paper protocol, complete-grid construction, safe named-tuple parsing, paper-compatible scoring, and checkpointed Azure execution. A separate marimo notebook composes those tested units, exposes explicit smoke/full-run gates, and renders only concise run metadata plus the top-line comparison table.

**Tech Stack:** Python 3.12, pandas, OpenAI Python SDK Responses API, Hugging Face `evaluate`, NumPy, marimo, unittest, AST parsing.

**Spec:** `docs/superpowers/specs/2026-09-21-gpt56-sol-paper-replication-design.md`

## Global Constraints

- Reproduce `OncLLMExtraction` commit `ddf1792ee5e1433be9ee4150d67d97b604a8daec` and never import that sibling repository at runtime.
- Use the exact 14 advanced-inference prompts, exact advanced-inference system preamble, named-tuple response contract, defaults, relation formatting, and metric aggregation from that commit.
- Build exactly 40 documents × 2 sections × 14 tasks = 1,120 unique requests from `data/coral_inference.csv`.
- Send `section_text + task_prompt` unchanged as `input` and the exact paper preamble as `instructions`; never add evidence-schema or corrective text.
- Use `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_DEPLOYMENT` (default `gpt-5.6-sol`); never persist or display the API key.
- Preserve a `*.services.ai.azure.com/openai/v1` endpoint verbatim and use it as `OpenAI(base_url=endpoint, api_key=api_key)`.
- Fix model settings at `max_output_tokens=4096` and `reasoning={"effort": "low"}`.
- Retry only transient API failures, and resend byte-for-byte identical API arguments on every retry. Malformed model text is not retried and maps to the paper default tuple.
- Score only actual completed Responses API outputs. Missing model outputs and API failures must never become synthetic predictions.
- Supply the task default reference only when the source CSV has no annotation row for a completed document-section-task.
- Enforce the spend cap before every Azure request attempt and make API failures and cap stops eligible for a later resume.
- Use only the `gpt56_sol_paper_replication` artifact prefix; do not read or overwrite existing `gpt56_sol_azure` artifacts.
- Full execution requires the literal confirmation `RUN 1120`; no test or ungated script invocation may make a paid request.
- The final notebook result contains only BLEU-4, ROUGE-1, and EM F1 for GPT-5.6 Sol, published GPT-4, and their differences. Do not render raw outputs or instance-level tables.
- Preserve unrelated worktree changes, especially `coral/benchmarking/evaluate_model.py`, `notebooks/evaluate_gpt56_sol_azure.py`, and `tests/test_evaluate_model.py`.
- Use `apply_patch` for edits and `UV_CACHE_DIR=/tmp/coral-uv-cache uv run` for Python commands.
- Before editing the notebook, read and follow the `marimo-notebook` skill. Before declaring completion, read and follow `superpowers:verification-before-completion`.

---

## File Structure

- `coral/paper_replication/__init__.py`: narrow public API for the notebook and tests.
- `coral/paper_replication/protocol.py`: immutable provenance, exact prompt text, task order, constructor mapping, and protocol constants.
- `coral/paper_replication/grid.py`: source-data validation and deterministic 1,120-row request-grid construction.
- `coral/paper_replication/parsing.py`: allowlisted AST named-tuple parser plus JSON-safe tuple serialization.
- `coral/paper_replication/scoring.py`: reference indexing, relation formatting, paper metric loop, aggregation, top-line comparison, and CSV artifact writing.
- `coral/paper_replication/runner.py`: Azure settings/client selection, cost projection, JSONL checkpointing, retry/resume semantics, and serial execution.
- `tests/test_paper_replication_protocol.py`: frozen prompt provenance and grid cardinality/error tests.
- `tests/test_paper_replication_parsing.py`: valid/default/malicious parser tests.
- `tests/test_paper_replication_scoring.py`: paper-loop parity, actual-output-only, aggregation, and artifact tests.
- `tests/test_paper_replication_runner.py`: fake-client settings, request, cap, retry, checkpoint, and resume tests.
- `tests/test_paper_replication_notebook.py`: static notebook safety, presentation, and artifact-path tests.
- `notebooks/evaluate_gpt56_sol_paper_replication.py`: clean marimo UI and inert-by-default script entry point.

### Task 1: Freeze the paper protocol and build the complete request grid

**Files:**
- Create: `coral/paper_replication/__init__.py`
- Create: `coral/paper_replication/protocol.py`
- Create: `coral/paper_replication/grid.py`
- Create: `tests/test_paper_replication_protocol.py`

**Interfaces:**
- Produces `PAPER_SOURCE_COMMIT: str`, `PAPER_PREAMBLE: str`, `TASK_ORDER: tuple[str, ...]`, `TASK_PROMPTS: Mapping[str, str]`, `PAPER_GPT4: Mapping[str, float]`, `MAX_OUTPUT_TOKENS: int`, and `REASONING_EFFORT: str`.
- Produces `get_task_prompt(task: str) -> str`.
- Produces `build_request_grid(source: pd.DataFrame) -> pd.DataFrame` with columns `doc_idx`, `section_name`, `section_text`, `task`, `task_prompt`, `instructions`, and `request_input`.
- Produces `load_source(path: Path) -> pd.DataFrame` with `inference_subtype` normalized to `task` and `doc_idx` normalized to string.

- [ ] **Step 1: Write failing frozen-protocol tests**

Create `tests/test_paper_replication_protocol.py` with the exact order and hashes captured from commit `ddf1792`:

```python
import hashlib
import unittest

from coral.paper_replication.protocol import (
    MAX_OUTPUT_TOKENS,
    PAPER_PREAMBLE,
    PAPER_SOURCE_COMMIT,
    REASONING_EFFORT,
    TASK_ORDER,
    TASK_PROMPTS,
)


EXPECTED_PROMPT_HASHES = {
    "symptoms": "8a8a8b45bd1317bfb810412329bd09874370ba6caf64e8cd9d42af6b7ef91fb6",
    "symptoms_at_diagnosis": "2cc8c90dad86ebd4ae17b15816e7c7e03e026aaf2434155683ca17afbba11d0c",
    "symptoms_due_to_cancer": "b295dc2feca3d5efee332d49876943b7bb5b88390da09de4731d638d51ef7e7e",
    "radtest_datetime_site_reason_result": "61bf4578d25c42f435387ddb103a4ffa34ecb7ed64d9e511aec931c40cc7c472",
    "procedure_datetime_site_reason_result": "234a7d06f10190cb3415c493bbf44fa1504cdbf6b555d9d1b7f7f1b47cf5c886",
    "genomictest_datetime_result": "ef01001655a00f11c7ea733c53093a6fd1e597a06e8cf3805db2f5301eca161b",
    "biomarker_datetime": "1d167fac5fbbba18f383e439eead7c89ea55f7128aca137d86cc953881ac8dbb",
    "histology_datetime": "34b8a648ca7743cbf77a5992ef8267da0427e1ac483f2675269cbd81ce259862",
    "metastasis_site_procedure_datetime": "8b9e66a0641a0aab787ba53ccba3f8eaab73b3655b4b447927608d66176b0434",
    "stage_datetime_addtest": "a3e0d85dc34316dab086edd8116c01dc5832224d0c842087c204d292d7f10a8d",
    "tnm_datetime_addtest": "99da426bc5c3d4e5ec548d0d655dee37c33b45ff560b8e93f970163f4a79a95f",
    "grade_datetime_addtest": "7fd5c63d53c7542d0f5481a8d8fa206ee4f4fa544cc57828ff73a0f375f391c0",
    "prescribed_med_begin_end_reason_continuity_ae": "01a1cdf5b8863f9b83d761b28b91e67780ed1245c8d8296efd3a18a93a55059e",
    "future_med_consideration_ae": "59b7058620ddb667879765eab5c36175b6c127034b27925c8ba83b0a2c203e0b",
}


class PaperProtocolTests(unittest.TestCase):
    def test_protocol_is_frozen_to_paper_commit(self):
        self.assertEqual(
            PAPER_SOURCE_COMMIT,
            "ddf1792ee5e1433be9ee4150d67d97b604a8daec",
        )
        self.assertEqual(
            hashlib.sha256(PAPER_PREAMBLE.encode()).hexdigest(),
            "53258763d20b6de42633a87d6c5a260184e2d06de44d3453f5f6010d0407cfde",
        )
        self.assertEqual(tuple(EXPECTED_PROMPT_HASHES), TASK_ORDER)
        self.assertEqual(
            {name: hashlib.sha256(prompt.encode()).hexdigest()
             for name, prompt in TASK_PROMPTS.items()},
            EXPECTED_PROMPT_HASHES,
        )

    def test_model_controls_are_fixed(self):
        self.assertEqual(MAX_OUTPUT_TOKENS, 4096)
        self.assertEqual(REASONING_EFFORT, "low")
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_protocol -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'coral.paper_replication'`.

- [ ] **Step 3: Port the exact paper protocol into `protocol.py`**

Read the source from the frozen commit, not from the sibling working tree:

```bash
git -C /Users/wkt406/projects/feinberg/OncLLMExtraction show ddf1792:gptextract/prompts/prompt_design.py
git -C /Users/wkt406/projects/feinberg/OncLLMExtraction show ddf1792:gptextract/__init__.py
```

Copy `Prompt.adv_chat_preamble`, `Prompt.adv_prompt_template`, and all 14 entries of `Prompt.adv_criteria_to_prompt_mapping` exactly. Materialize `TASK_PROMPTS` as the formatted `adv_prompt_template.format(subprompt)` values in the paper generator order. Define the fixed controls and published comparison values:

```python
PAPER_SOURCE_COMMIT = "ddf1792ee5e1433be9ee4150d67d97b604a8daec"
PAPER_PREAMBLE = "Pretend you are an oncologist. Answer based on the given clinical note for a patient."
MAX_OUTPUT_TOKENS = 4096
REASONING_EFFORT = "low"
PAPER_GPT4 = {"BLEU-4": 0.73, "ROUGE-1": 0.72, "EM F1": 0.51}

TASK_ORDER = (
    "symptoms",
    "symptoms_at_diagnosis",
    "symptoms_due_to_cancer",
    "radtest_datetime_site_reason_result",
    "procedure_datetime_site_reason_result",
    "genomictest_datetime_result",
    "biomarker_datetime",
    "histology_datetime",
    "metastasis_site_procedure_datetime",
    "stage_datetime_addtest",
    "tnm_datetime_addtest",
    "grade_datetime_addtest",
    "prescribed_med_begin_end_reason_continuity_ae",
    "future_med_consideration_ae",
)


def get_task_prompt(task: str) -> str:
    try:
        return TASK_PROMPTS[task]
    except KeyError as error:
        raise ValueError(f"unknown paper task: {task}") from error
```

Keep `TASK_PROMPTS` read-only with `types.MappingProxyType`. Export only the public constants and functions from `coral/paper_replication/__init__.py`.

- [ ] **Step 4: Write failing grid construction and validation tests**

Extend `tests/test_paper_replication_protocol.py`:

```python
from pathlib import Path

import pandas as pd

from coral.paper_replication.grid import build_request_grid, load_source


class PaperGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_source(Path("data/coral_inference.csv"))

    def test_real_data_builds_complete_unique_grid(self):
        grid = build_request_grid(self.source)
        self.assertEqual(len(grid), 1120)
        self.assertEqual(grid.doc_idx.nunique(), 40)
        self.assertEqual(set(grid.section_name), {"hpi", "a&p"})
        self.assertEqual(tuple(grid.task.drop_duplicates()), TASK_ORDER)
        self.assertFalse(grid.duplicated(["doc_idx", "section_name", "task"]).any())

    def test_request_is_exact_section_text_plus_paper_prompt(self):
        grid = build_request_grid(self.source)
        row = grid.iloc[0]
        self.assertEqual(row.request_input, row.section_text + TASK_PROMPTS[row.task])
        self.assertEqual(row.instructions, PAPER_PREAMBLE)
        self.assertNotIn("evidence_quotes", row.request_input)

    def test_conflicting_section_text_is_rejected(self):
        bad = pd.concat([
            self.source,
            pd.DataFrame([{**self.source.iloc[0].to_dict(), "section_text": "conflict"}]),
        ], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "conflicting section text"):
            build_request_grid(bad)

    def test_incomplete_document_section_set_is_rejected(self):
        bad = self.source[self.source.doc_idx != self.source.doc_idx.iloc[0]]
        with self.assertRaisesRegex(ValueError, "40 documents"):
            build_request_grid(bad)

    def test_unknown_or_missing_task_definition_is_rejected(self):
        bad = self.source.copy()
        bad.loc[bad.index[0], "task"] = "not_a_paper_task"
        with self.assertRaisesRegex(ValueError, "task set"):
            build_request_grid(bad)
```

- [ ] **Step 5: Run the focused tests to verify RED**

Run the Task 1 test module again. Expected: prompt tests PASS and grid tests FAIL because `grid.py` is absent.

- [ ] **Step 6: Implement deterministic grid validation**

Implement `load_source` and `build_request_grid` with these exact checks:

```python
REQUIRED_COLUMNS = {
    "doc_idx", "section_name", "section_text", "task", "annotation_set",
}
EXPECTED_SECTIONS = ("hpi", "a&p")


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"inference_subtype": "task"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"source data is missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["doc_idx"] = frame["doc_idx"].astype(str)
    return frame
```

In `build_request_grid`, reject null/blank section text; reject any document-section pair with more than one distinct text; require 40 distinct document IDs; require every document to have exactly `hpi` and `a&p`; require the source task set to equal `set(TASK_ORDER)`; sort document IDs numerically when all are decimal strings, otherwise lexically; order sections as `hpi`, then `a&p`; and cross-join each section with `TASK_ORDER`. Construct `request_input` only as `section_text + TASK_PROMPTS[task]`.

- [ ] **Step 7: Verify GREEN and commit Task 1**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_protocol -v
```

Expected: all Task 1 tests PASS.

Commit only Task 1 files:

```bash
git add coral/paper_replication/__init__.py coral/paper_replication/protocol.py coral/paper_replication/grid.py tests/test_paper_replication_protocol.py
git commit -m "feat: freeze CORAL paper replication protocol"
```

### Task 2: Add the allowlisted paper-compatible named-tuple parser

**Files:**
- Create: `coral/paper_replication/parsing.py`
- Create: `tests/test_paper_replication_parsing.py`
- Modify: `coral/paper_replication/__init__.py`

**Interfaces:**
- Consumes task names and defaults from `coral.task_to_default_tuple_dict` and `TASK_ORDER` from Task 1.
- Produces `parse_namedtuple_expression(source: str, task: str) -> tuple` for one safe expression.
- Produces `parse_paper_output(output: object, task: str) -> list[tuple]` with paper default behavior.
- Produces `parse_annotation_set(annotation_set: object, task: str) -> list[tuple]` for trusted-format CSV annotations without `eval()`.
- Produces `serialize_parsed_tuples(values: list[tuple]) -> str`, a deterministic JSON representation for CSV artifacts.

- [ ] **Step 1: Write failing valid-constructor and default tests**

Create `tests/test_paper_replication_parsing.py`:

```python
import json
import unittest

from coral import RadTest, SymptomEnt, task_to_default_tuple_dict
from coral.paper_replication.parsing import (
    parse_annotation_set,
    parse_namedtuple_expression,
    parse_paper_output,
    serialize_parsed_tuples,
)


class PaperParserTests(unittest.TestCase):
    def test_parses_keyword_and_positional_named_tuples(self):
        keyword = parse_namedtuple_expression(
            "SymptomEnt(Symptom='fatigue', Datetime={'today'})", "symptoms"
        )
        positional = parse_namedtuple_expression(
            "SymptomEnt('fatigue', {'today'})", "symptoms"
        )
        self.assertEqual(keyword, SymptomEnt("fatigue", {"today"}))
        self.assertEqual(positional, keyword)

    def test_splits_paper_style_space_separated_and_newline_outputs(self):
        output = (
            "SymptomEnt(Symptom='fatigue', Datetime={'today'}) "
            "SymptomEnt(Symptom='pain', Datetime={'yesterday'})\nN/A"
        )
        self.assertEqual(
            parse_paper_output(output, "symptoms"),
            [
                SymptomEnt("fatigue", {"today"}),
                SymptomEnt("pain", {"yesterday"}),
                task_to_default_tuple_dict["symptoms"],
            ],
        )

    def test_empty_nonanswer_and_malformed_text_default(self):
        for output in ("", "N/A", "no symptoms", "none reported", "unknown", "bad("):
            with self.subTest(output=output):
                self.assertEqual(
                    parse_paper_output(output, "symptoms"),
                    [task_to_default_tuple_dict["symptoms"]],
                )

    def test_each_malformed_line_gets_a_default(self):
        parsed = parse_paper_output("bad(\nN/A", "symptoms")
        self.assertEqual(parsed, [task_to_default_tuple_dict["symptoms"]] * 2)

    def test_annotation_lines_use_the_same_safe_constructor_parser(self):
        parsed = parse_annotation_set(
            "RadTest(RadiologyTest='CT', Datetime={'today'}, Site={'chest'}, "
            "Reason={'staging'}, Result={'stable'})\n",
            "radtest_datetime_site_reason_result",
        )
        self.assertEqual(
            parsed,
            [RadTest("CT", {"today"}, {"chest"}, {"staging"}, {"stable"})],
        )

    def test_every_paper_task_accepts_its_default_constructor(self):
        for task, default in task_to_default_tuple_dict.items():
            with self.subTest(task=task):
                self.assertEqual(parse_paper_output(repr(default), task), [default])

    def test_serialization_is_json_and_stable_for_sets(self):
        rendered = serialize_parsed_tuples([SymptomEnt("fatigue", {"today", "yesterday"})])
        self.assertEqual(
            json.loads(rendered),
            [{"constructor": "SymptomEnt", "fields": {
                "Symptom": "fatigue", "Datetime": ["today", "yesterday"]
            }}],
        )
```

- [ ] **Step 2: Write failing code-execution rejection tests**

Add these tests to the same class:

```python
    def test_rejects_wrong_constructor_for_task(self):
        with self.assertRaisesRegex(ValueError, "constructor"):
            parse_namedtuple_expression(
                "RadTest('CT', {'today'}, {'chest'}, {'staging'}, {'stable'})",
                "symptoms",
            )

    def test_arbitrary_python_is_never_executed(self):
        payloads = (
            "__import__('os').system('false')",
            "SymptomEnt(Symptom=open('/tmp/x'), Datetime={'today'})",
            "SymptomEnt(**{'Symptom': 'fatigue', 'Datetime': {'today'}})",
            "(lambda: 1)()",
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(
                    parse_paper_output(payload, "symptoms"),
                    [task_to_default_tuple_dict["symptoms"]],
                )
```

- [ ] **Step 3: Run the tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_parsing -v
```

Expected: FAIL because `coral.paper_replication.parsing` is absent.

- [ ] **Step 4: Implement the AST allowlist and paper fallback behavior**

Create one task-specific constructor map; do not accept any constructor other than the one belonging to the requested task:

```python
TASK_CONSTRUCTORS = {
    task: type(default) for task, default in task_to_default_tuple_dict.items()
    if task in TASK_ORDER
}
NON_ANSWERS = ("no ", "none ")
```

`parse_namedtuple_expression` must:

1. parse with `ast.parse(source, mode="eval")`;
2. require the body to be exactly `ast.Call`;
3. require `call.func` to be an `ast.Name` equal to the expected constructor name;
4. reject starred positional arguments and keyword entries with `arg is None`;
5. decode every argument value only with `ast.literal_eval`;
6. invoke the expected named-tuple class so wrong fields/arity raise `ValueError`;
7. never call Python `eval`, `exec`, `compile`, or imported functions from model text.

`parse_paper_output` must normalize non-string/NaN to an empty string, apply the paper's `o'clock` → `o clock` repair, insert newlines using `re.sub(r"(\))\s([A-Z]+)", r"\1\n\2", output)`, strip the whole output, and parse every resulting line. Empty whole output, empty interior lines, `N/A`, `unknown`, strings starting with `no ` or `none `, and any rejected expression produce one task default for that line. Return a fresh list while reusing immutable named tuples.

`parse_annotation_set` must parse every non-empty annotation line with `parse_namedtuple_expression`; blank/NaN annotations return the task default. Annotation parse errors must raise a contextual `ValueError` rather than silently changing gold data.

`serialize_parsed_tuples` must emit JSON with constructor name and named fields; sets become sorted lists, tuples/lists become JSON lists, dictionaries are key-sorted, and scalars remain scalars.

- [ ] **Step 5: Verify GREEN and commit Task 2**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_parsing -v
```

Expected: all parser tests PASS and no file is created at `/tmp/x`.

Commit:

```bash
git add coral/paper_replication/__init__.py coral/paper_replication/parsing.py tests/test_paper_replication_parsing.py
git commit -m "feat: safely parse CORAL paper named tuples"
```

### Task 3: Reproduce paper scoring and write the five isolated artifacts

**Files:**
- Create: `coral/paper_replication/scoring.py`
- Create: `tests/test_paper_replication_scoring.py`
- Modify: `coral/paper_replication/__init__.py`

**Interfaces:**
- Consumes `parse_annotation_set`, `parse_paper_output`, and `serialize_parsed_tuples` from Task 2.
- Produces `format_relations(values: list[tuple]) -> dict[str, set[str]]`.
- Produces `score_completed_records(source: pd.DataFrame, records: list[dict[str, object]], metrics: MetricProtocol | None = None) -> ReplicationScores`.
- Produces frozen `ReplicationScores(outputs, instances, relations, topline)` whose fields are pandas DataFrames.
- Produces `write_artifacts(scores: ReplicationScores, output_dir: Path, prefix: str = "gpt56_sol_paper_replication") -> ArtifactPaths`.
- Produces frozen `ArtifactPaths(checkpoint, outputs, instances, relations, topline)` with the exact filenames from the spec.

- [ ] **Step 1: Write failing relation-format parity tests**

Create `tests/test_paper_replication_scoring.py`:

```python
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coral import PrescribedMedEnt, StageEnt, SymptomEnt
from coral.paper_replication.scoring import (
    format_relations,
    score_completed_records,
    write_artifacts,
)


class PaperRelationFormattingTests(unittest.TestCase):
    def test_formats_relations_like_paper_and_omits_additional_testing(self):
        relations = format_relations([
            StageEnt("II", {"today"}, {"PET"}),
            SymptomEnt("Fatigue", {"Today"}),
        ])
        self.assertEqual(relations["Stage Datetime"], {"II today"})
        self.assertEqual(relations["Symptom Datetime"], {"Fatigue Today"})
        self.assertNotIn("Stage AdditionalTesting", relations)

    def test_formats_scalar_and_set_medication_fields(self):
        relations = format_relations([
            PrescribedMedEnt(
                "Drug", {"start"}, {"end"}, {"cancer"}, "ongoing",
                {"rash"}, {"swelling"},
            )
        ])
        self.assertEqual(relations["MedicationName Continuity"], {"Drug ongoing"})
        self.assertEqual(relations["MedicationName Reason"], {"Drug cancer"})
```

- [ ] **Step 2: Write failing actual-output-only and aggregation tests**

Use a deterministic fake with the same method surface as `coral.utils.metrics.Metrics`:

```python
class FakeMetrics:
    def compute_bleu_score(self, preds, references, max_n, smooth):
        return {"bleu": float(preds[0] in references[0])}

    def compute_rouge_score(self, preds, references, rouge_types):
        return {"rouge1": float(preds[0] == references[0])}

    def compute_em_over_multiset_prec_recall_f1(self, outputs, annotations):
        output_set, annotation_set = set(outputs), set(annotations)
        true_positive = len(output_set & annotation_set)
        precision = true_positive / len(output_set)
        recall = true_positive / len(annotation_set)
        f1 = 0.0 if precision == recall == 0.0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1


SOURCE = pd.DataFrame([{
    "doc_idx": "1",
    "section_name": "hpi",
    "section_text": "Fatigue today.",
    "task": "symptoms",
    "annotation_set": "SymptomEnt(Symptom='fatigue', Datetime={'today'})\n",
}])

COMPLETED = {
    "doc_idx": "1", "section_name": "hpi", "task": "symptoms",
    "model": "gpt-5.6-sol", "status": "completed",
    "output_text": "SymptomEnt(Symptom='fatigue', Datetime={'today'})",
    "input_tokens": 100, "output_tokens": 20, "cost": 0.0008,
    "elapsed_seconds": 1.0,
}


class PaperScoringTests(unittest.TestCase):
    def test_scores_only_completed_api_outputs(self):
        failed = {**COMPLETED, "doc_idx": "2", "status": "api_failed", "output_text": None}
        scores = score_completed_records(SOURCE, [COMPLETED, failed], FakeMetrics())
        self.assertEqual(len(scores.outputs), 1)
        self.assertEqual(len(scores.instances), 1)
        self.assertEqual(scores.instances.iloc[0].em_f1, 1.0)

    def test_malformed_completed_output_is_scored_as_paper_default(self):
        malformed = {**COMPLETED, "output_text": "not a named tuple"}
        scores = score_completed_records(SOURCE, [malformed], FakeMetrics())
        self.assertEqual(scores.instances.iloc[0].em_f1, 0.0)

    def test_missing_annotation_uses_default_reference(self):
        no_annotation_source = SOURCE.iloc[0:0].copy()
        unknown = {
            **COMPLETED,
            "output_text": "SymptomEnt(Symptom='unknown', Datetime={'unknown'})",
        }
        scores = score_completed_records(no_annotation_source, [unknown], FakeMetrics())
        self.assertEqual(scores.instances.iloc[0].em_f1, 1.0)

    def test_topline_is_unweighted_mean_of_relation_aggregates(self):
        second = {
            **COMPLETED,
            "doc_idx": "2",
            "output_text": "SymptomEnt(Symptom='pain', Datetime={'today'})",
        }
        source = pd.concat([
            SOURCE,
            pd.DataFrame([{**SOURCE.iloc[0].to_dict(), "doc_idx": "2"}]),
        ], ignore_index=True)
        scores = score_completed_records(source, [COMPLETED, second], FakeMetrics())
        self.assertEqual(scores.topline.metric.tolist(), ["BLEU-4", "ROUGE-1", "EM F1"])
        self.assertEqual(scores.topline.paper_gpt4.tolist(), [0.73, 0.72, 0.51])
        self.assertEqual(scores.topline.gpt56_sol.tolist(), [0.5, 0.5, 0.5])
```

- [ ] **Step 3: Run scoring tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_scoring -v
```

Expected: FAIL because `scoring.py` is absent.

- [ ] **Step 4: Implement the paper relation and metric loops**

Define `MetricProtocol` with the three methods used by the paper. When no metric object is injected, lazily instantiate `coral.utils.metrics.Metrics(tokenizer="default")` so importing the package does not load metric resources.

`format_relations` must match `OncInfoExtr._format_tuple_annots_for_eval` at commit `ddf1792`: skip `CancerDiagnosis`; use the first tuple field as primary type/value; skip the primary field and `AdditionalTesting`; map empty relation values to `unknown`; expand sets one value at a time; combine as `f"{primary_value} {relation_value}"`; and raise if no relation is produced.

For each unique completed `(doc_idx, section_name, task, model)` record, `score_completed_records` must:

1. parse its raw output with `parse_paper_output`;
2. obtain exactly one matching source annotation row or use `[task_to_default_tuple_dict[task]]` when absent;
3. reject duplicate matching annotation rows;
4. format predicted and gold relations;
5. iterate only predicted relation keys, lowercase both value sets, and calculate paper metrics exactly as follows:

```python
bleu4 = np.mean([
    metrics.compute_bleu_score(
        preds=[prediction], references=[annotation_values], max_n=4, smooth=True,
    )["bleu"]
    for prediction in prediction_values
])

rouge1 = np.mean([
    max(
        metrics.compute_rouge_score(
            preds=[prediction], references=[annotation], rouge_types=["rouge1"],
        )["rouge1"]
        for prediction in prediction_values
    )
    for annotation in annotation_values
])

em_precision, em_recall, em_f1 = (
    metrics.compute_em_over_multiset_prec_recall_f1(
        prediction_values, annotation_values,
    )
)
```

Write one instance row per predicted relation. Aggregate by `task`, `model`, and `subrelation` with means for `bleu4`, `rouge1`, `em_precision`, `em_recall`, and `em_f1`. Compute each GPT-5.6 top-line score as the unweighted mean of its per-relation aggregate column, round displayed values and differences to two decimals, and compare with `PAPER_GPT4`.

- [ ] **Step 5: Add artifact-path and CSV tests**

Extend the scoring tests:

```python
    def test_writes_exact_isolated_artifact_names(self):
        scores = score_completed_records(SOURCE, [COMPLETED], FakeMetrics())
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifacts(scores, Path(directory))
            self.assertEqual(paths.outputs.name, "gpt56_sol_paper_replication_outputs.csv")
            self.assertEqual(paths.instances.name, "gpt56_sol_paper_replication_instance_scores.csv")
            self.assertEqual(paths.relations.name, "gpt56_sol_paper_replication_relation_scores.csv")
            self.assertEqual(paths.topline.name, "gpt56_sol_paper_replication_topline.csv")
            for path in (paths.outputs, paths.instances, paths.relations, paths.topline):
                self.assertTrue(path.exists())

    def test_outputs_csv_contains_raw_and_json_parsed_output(self):
        scores = score_completed_records(SOURCE, [COMPLETED], FakeMetrics())
        self.assertIn("output_text", scores.outputs)
        self.assertIn("parsed_output_json", scores.outputs)
        self.assertNotIn("api_key", scores.outputs)
```

`ArtifactPaths.checkpoint` must be `output_dir / "gpt56_sol_paper_replication.jsonl"`; `write_artifacts` writes the other four frames and does not touch the checkpoint.

- [ ] **Step 6: Verify GREEN and commit Task 3**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_scoring -v
```

Expected: all scoring and artifact tests PASS.

Commit:

```bash
git add coral/paper_replication/__init__.py coral/paper_replication/scoring.py tests/test_paper_replication_scoring.py
git commit -m "feat: score paper-protocol outputs without synthetic rows"
```

### Task 4: Add cost-capped Azure execution with retryable checkpoints

**Files:**
- Create: `coral/paper_replication/runner.py`
- Create: `tests/test_paper_replication_runner.py`
- Modify: `coral/paper_replication/__init__.py`

**Interfaces:**
- Consumes `AzureSettings`, `Usage`, `estimate_cost`, and `load_azure_settings` from `coral.azure_evaluation` without modifying that module.
- Consumes `PAPER_PREAMBLE`, `MAX_OUTPUT_TOKENS`, and `REASONING_EFFORT` from Task 1.
- Produces frozen `PaperCheckpointRecord` with request identity, `status`, response text, token/cost/elapsed fields, and sanitized error.
- Produces `read_checkpoint(path: Path) -> list[dict[str, object]]`, ignoring a truncated final JSONL line.
- Produces `completed_keys(records: Iterable[Mapping[str, object]]) -> set[tuple[str, str, str, str]]` using only `status == "completed"`.
- Produces `project_grid_cost(grid: pd.DataFrame) -> float` and `checkpoint_spend(records) -> float`.
- Produces `smoke_is_complete(grid: pd.DataFrame, records: Iterable[Mapping[str, object]], model: str) -> bool` for the first eight grid keys.
- Produces `load_azure_settings(env: Mapping[str, str]) -> AzureSettings` as a public re-export of the existing tested settings loader.
- Produces `create_azure_client(settings: AzureSettings, openai_class: Callable[..., object] | None = None, azure_class: Callable[..., object] | None = None) -> ResponseClient`.
- Produces `run_replication(grid, client, settings, checkpoint_path, spend_cap, progress=None, sleep=time.sleep) -> pd.DataFrame`.

- [ ] **Step 1: Write failing settings and endpoint-selection tests**

Create fake constructors that record keyword arguments, then add:

```python
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coral.azure_evaluation import AzureSettings
from coral.paper_replication.protocol import MAX_OUTPUT_TOKENS, PAPER_PREAMBLE
from coral.paper_replication.runner import (
    create_azure_client,
    project_grid_cost,
    read_checkpoint,
    run_replication,
    smoke_is_complete,
)


class RecordingClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = FakeResponses([])


SETTINGS = AzureSettings("secret", "https://example.openai.azure.com")


class RateLimitError(Exception):
    status_code = 429


class PaperRunnerSettingsTests(unittest.TestCase):
    def test_foundry_v1_endpoint_is_preserved_as_openai_base_url(self):
        settings = AzureSettings(
            "secret",
            "https://wkt406-codehelp-resource.services.ai.azure.com/openai/v1",
            "gpt-5.6-sol",
        )
        client = create_azure_client(
            settings, openai_class=RecordingClient, azure_class=self.fail_constructor,
        )
        self.assertEqual(client.kwargs["base_url"], settings.endpoint)
        self.assertEqual(client.kwargs["api_key"], "secret")

    def test_standard_azure_endpoint_uses_azure_client(self):
        settings = AzureSettings(
            "secret", "https://example.openai.azure.com", "gpt-5.6-sol",
            "2025-04-01-preview",
        )
        client = create_azure_client(
            settings, openai_class=self.fail_constructor, azure_class=RecordingClient,
        )
        self.assertEqual(client.kwargs, {
            "azure_endpoint": settings.endpoint,
            "api_key": "secret",
            "api_version": "2025-04-01-preview",
        })

    @staticmethod
    def fail_constructor(**kwargs):
        raise AssertionError(f"wrong client selected: {sorted(kwargs)}")
```

- [ ] **Step 2: Write failing exact-request and no-correction tests**

Use these fakes and one-row grid:

```python
class FakeResponse:
    def __init__(self, text, input_tokens=100, output_tokens=20):
        self.output_text = text
        self.usage = type("Usage", (), {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })()


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


GRID = pd.DataFrame([{
    "doc_idx": "1", "section_name": "hpi", "task": "symptoms",
    "section_text": "Fatigue today.",
    "task_prompt": "\nPaper task prompt",
    "instructions": PAPER_PREAMBLE,
    "request_input": "Fatigue today.\nPaper task prompt",
}])


class PaperRunnerTests(unittest.TestCase):
    def test_sends_exact_paper_request_and_fixed_model_controls(self):
        client = FakeClient([FakeResponse("not parseable by design")])
        with tempfile.TemporaryDirectory() as directory:
            result = run_replication(
                GRID, client, AzureSettings("secret", "https://example.openai.azure.com"),
                Path(directory) / "checkpoint.jsonl", spend_cap=1.0,
            )
        call = client.responses.calls[0]
        self.assertEqual(call["input"], GRID.iloc[0].request_input)
        self.assertEqual(call["instructions"], PAPER_PREAMBLE)
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertEqual(call["max_output_tokens"], MAX_OUTPUT_TOKENS)
        self.assertNotIn("text", call)
        self.assertEqual(result.iloc[0].status, "completed")

    def test_malformed_text_is_completed_without_corrective_retry(self):
        client = FakeClient([FakeResponse("malformed")])
        with tempfile.TemporaryDirectory() as directory:
            run_replication(
                GRID, client, AzureSettings("secret", "https://example.openai.azure.com"),
                Path(directory) / "checkpoint.jsonl", spend_cap=1.0,
            )
        self.assertEqual(len(client.responses.calls), 1)
```

- [ ] **Step 3: Write failing transient retry, cap, and resume tests**

Define a fake exception with `status_code = 429`, then test:

```python
    def test_transient_retry_reuses_identical_arguments(self):
        client = FakeClient([RateLimitError("slow down"), FakeResponse("N/A")])
        with tempfile.TemporaryDirectory() as directory:
            result = run_replication(
                GRID, client, AzureSettings("secret", "https://example.openai.azure.com"),
                Path(directory) / "checkpoint.jsonl", spend_cap=1.0, sleep=lambda _: None,
            )
        self.assertEqual(client.responses.calls[0], client.responses.calls[1])
        self.assertEqual(result.iloc[0].status, "completed")

    def test_cap_is_checked_before_the_client_call_and_is_retryable(self):
        client = FakeClient([])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            result = run_replication(
                GRID, client, AzureSettings("secret", "https://example.openai.azure.com"),
                path, spend_cap=0.0,
            )
            self.assertEqual(result.iloc[0].status, "spend_cap_reached")
            self.assertEqual(client.responses.calls, [])
            resumed_client = FakeClient([FakeResponse("N/A")])
            resumed = run_replication(
                GRID, resumed_client,
                AzureSettings("secret", "https://example.openai.azure.com"),
                path, spend_cap=1.0,
            )
        self.assertEqual(resumed.iloc[0].status, "completed")

    def test_api_failure_is_recorded_but_retryable_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            first = run_replication(
                GRID, FakeClient([ValueError("bad request")]),
                AzureSettings("secret", "https://example.openai.azure.com"),
                path, spend_cap=1.0, sleep=lambda _: None,
            )
            second = run_replication(
                GRID, FakeClient([FakeResponse("N/A")]),
                AzureSettings("secret", "https://example.openai.azure.com"),
                path, spend_cap=1.0, sleep=lambda _: None,
            )
        self.assertEqual(first.iloc[0].status, "api_failed")
        self.assertEqual(second.iloc[0].status, "completed")

    def test_completed_request_is_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            run_replication(
                GRID, FakeClient([FakeResponse("N/A")]), SETTINGS, path, 1.0,
            )
            client = FakeClient([])
            resumed = run_replication(GRID, client, SETTINGS, path, 1.0)
        self.assertEqual(resumed.iloc[0].status, "skipped_on_resume")
        self.assertEqual(client.responses.calls, [])

    def test_truncated_checkpoint_tail_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(
                json.dumps({"status": "completed", "doc_idx": "1"})
                + "\n{\"status\":",
                encoding="utf-8",
            )
            records = read_checkpoint(path)
        self.assertEqual(records, [{"status": "completed", "doc_idx": "1"}])

    def test_cost_projection_uses_every_request_and_fixed_ceiling(self):
        one = project_grid_cost(GRID)
        self.assertGreater(one, 0.0)
        self.assertAlmostEqual(project_grid_cost(pd.concat([GRID, GRID])), one * 2)

    def test_smoke_gate_requires_all_first_eight_completed_keys(self):
        grid = pd.concat([
            GRID.assign(doc_idx=str(index)) for index in range(8)
        ], ignore_index=True)
        records = [
            {
                "doc_idx": row.doc_idx, "section_name": row.section_name,
                "task": row.task, "model": "gpt-5.6-sol", "status": "completed",
            }
            for row in grid.itertuples(index=False)
        ]
        self.assertTrue(smoke_is_complete(grid, records, "gpt-5.6-sol"))
        self.assertFalse(smoke_is_complete(grid, records[:-1], "gpt-5.6-sol"))

    def test_checkpoint_and_settings_never_render_api_key(self):
        client = FakeClient([ValueError("secret was rejected")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            run_replication(GRID, client, SETTINGS, path, 1.0, sleep=lambda _: None)
            checkpoint_text = path.read_text(encoding="utf-8")
        self.assertNotIn("secret", repr(SETTINGS))
        self.assertNotIn("secret", checkpoint_text)
```

- [ ] **Step 4: Run runner tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_runner -v
```

Expected: FAIL because `runner.py` is absent.

- [ ] **Step 5: Implement client selection, checkpoint accounting, and execution**

`create_azure_client` must select by parsed hostname:

```python
hostname = (urlsplit(settings.endpoint).hostname or "").lower()
if hostname.endswith(".services.ai.azure.com"):
    return openai_class(base_url=settings.endpoint, api_key=settings.api_key)
return azure_class(
    azure_endpoint=settings.endpoint,
    api_key=settings.api_key,
    api_version=settings.api_version,
)
```

Import `OpenAI` and `AzureOpenAI` lazily only when injected classes are absent. Never normalize or rewrite the Foundry `base_url`.

Use these checkpoint statuses:

- `completed`: a real API response with output text and usage; terminal for resume and scoreable even if its text is malformed;
- `api_failed`: no completed response after retry policy; non-terminal and excluded from scoring;
- `spend_cap_reached`: no request was sent; non-terminal and excluded from scoring;
- `skipped_on_resume`: returned to the current caller only and not appended.

Before each attempt, estimate input tokens as `max(1, ceil((len(instructions) + len(input)) / 4)) + 256`, pair it with 4,096 projected output tokens, and call the existing `estimate_cost`. Sum actual `cost` fields across all valid checkpoint lines. Stop before `spent + projected > spend_cap`.

Retry status codes 408, 409, 429, and 500–599 at most three total attempts with delays `(1.0, 2.0)` seconds. Other exceptions fail after one attempt. Build the request arguments once outside the retry loop:

```python
request = {
    "model": settings.deployment,
    "instructions": row["instructions"],
    "input": row["request_input"],
    "reasoning": {"effort": REASONING_EFFORT},
    "max_output_tokens": MAX_OUTPUT_TOKENS,
}
```

Append each final record as compact one-line JSON using a frozen dataclass converted with `dataclasses.asdict`. Include no section text, prompt, environment, client representation, or secret in checkpoint rows. Sanitize errors to exception class plus a message with the API key replaced by `[REDACTED]` and truncate to 500 characters. Progress messages contain only ordinal/total, document ID, section, task, status, cumulative spend, and checkpoint path. When the cap is reached, append the non-terminal cap record and stop the loop so no subsequent request is attempted.

Implement `smoke_is_complete` by taking the first eight grid keys, adding the deployment to each key, and checking they are a subset of `completed_keys(records)`. API-failed and spend-cap rows never satisfy this gate.

- [ ] **Step 6: Verify GREEN and commit Task 4**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_runner -v
```

Expected: all runner tests PASS without network access.

Commit:

```bash
git add coral/paper_replication/__init__.py coral/paper_replication/runner.py tests/test_paper_replication_runner.py
git commit -m "feat: run resumable paper-protocol Azure evaluation"
```

### Task 5: Build the clean marimo notebook and complete offline verification

**Files:**
- Create: `notebooks/evaluate_gpt56_sol_paper_replication.py`
- Create: `tests/test_paper_replication_notebook.py`
- Modify: `coral/paper_replication/__init__.py` only if a missing public export is discovered by notebook imports.

**Interfaces:**
- Consumes the public Task 1–4 API only; it does not import `coral.benchmarking.evaluate_model` or `coral.azure_evaluation` directly.
- Produces an eight-request smoke action, a separately confirmed 1,120-request action, an inert script mode, concise progress, isolated files, and one final four-column top-line table.
- Script controls are `CORAL_PAPER_REPLICATION_RUN=smoke` for eight requests or `CORAL_PAPER_REPLICATION_RUN=full` plus `CORAL_PAPER_REPLICATION_CONFIRM='RUN 1120'` for the full grid.

- [ ] **Step 1: Read the required marimo skill before editing**

Read `/Users/wkt406/.agents/skills/marimo-notebook/SKILL.md` completely and follow its cell-dependency, UI, and validation rules. If it references required supporting files, read those before editing.

- [ ] **Step 2: Write failing notebook safety and presentation tests**

Create `tests/test_paper_replication_notebook.py`:

```python
import unittest
from pathlib import Path


NOTEBOOK = Path("notebooks/evaluate_gpt56_sol_paper_replication.py")


class PaperReplicationNotebookTests(unittest.TestCase):
    def test_declares_required_runtime_and_azure_configuration(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        for dependency in ('"marimo"', '"openai"', '"pandas"', '"evaluate"', '"torch"'):
            self.assertIn(dependency, text)
        self.assertIn("AZURE_OPENAI_API_KEY", text)
        self.assertIn("AZURE_OPENAI_ENDPOINT", text)
        self.assertIn("AZURE_OPENAI_DEPLOYMENT", text)
        self.assertNotIn("OPENAI_API_KEY", text.replace("AZURE_OPENAI_API_KEY", ""))

    def test_full_run_has_exact_confirmation_and_isolated_prefix(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertIn("RUN 1120", text)
        self.assertIn("CORAL_PAPER_REPLICATION_CONFIRM", text)
        self.assertIn("gpt56_sol_paper_replication", text)
        self.assertIn("smoke_is_complete", text)
        self.assertNotIn('"output/gpt56_sol_azure', text)

    def test_protocol_controls_are_not_user_adjustable(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("Reasoning effort", text)
        self.assertNotIn("Max output tokens", text)
        self.assertIn("4,096", text)
        self.assertIn("low reasoning effort", text)

    def test_notebook_does_not_render_individual_outputs(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("run_results.to_string", text)
        self.assertNotIn("mo.ui.table(run_results", text)
        self.assertIn("topline", text)
```

- [ ] **Step 3: Run the notebook tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_notebook -v
```

Expected: ERROR with `FileNotFoundError` because the notebook is absent.

- [ ] **Step 4: Implement the PEP 723 marimo notebook**

Declare Python `>=3.12` and dependencies:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "absl-py",
#     "evaluate",
#     "marimo",
#     "nltk",
#     "numpy",
#     "openai",
#     "pandas",
#     "rouge-score",
#     "scikit-learn",
#     "torch",
#     "transformers",
# ]
# ///
```

Use `marimo.App(width="medium")` and cells with single-assignment variable names. Build these cells in dependency order:

1. imports, repository-path setup, and public `coral.paper_replication` imports;
2. protocol explanation and disclosure of the Responses API/low-reasoning difference;
3. exact paths rooted at `Path("data/coral_inference.csv")` and `Path("output")`;
4. source load, validation, 1,120-row grid construction, and cost projection;
5. redacted Azure setting validation;
6. spend-cap widget `mo.ui.number(start=0, step=1, value=100, label="Spend cap (USD)")`, smoke button, full confirmation text widget, and full button;
7. run selection that permits eight rows for smoke and all rows only when the confirmation equals `RUN 1120` **and** `smoke_is_complete(grid, checkpoint_records, settings.deployment)` is true;
8. explicit client construction and `run_replication` call only when a UI action or authorized script environment requests it;
9. checkpoint read, `score_completed_records`, and `write_artifacts` when at least one completed response exists;
10. concise status/elapsed/spend metadata;
11. one `mo.ui.table(scores.topline, selection=None, pagination=False)` final result.

Do not render `grid`, raw checkpoint records, `scores.outputs`, `scores.instances`, or `scores.relations`. Do not import or call the legacy scorer. The top-line columns must be exactly:

```python
["metric", "gpt56_sol", "paper_gpt4", "difference_vs_paper"]
```

Script mode must remain inert unless `CORAL_PAPER_REPLICATION_RUN` is `smoke` or `full`. A `full` value without exact `CORAL_PAPER_REPLICATION_CONFIRM='RUN 1120'`, or before all eight isolated smoke keys are completed, must print a rejection and make no client. Every actual run prints one starting line, one line per request start/finish via the progress callback, and one completion summary.

- [ ] **Step 5: Verify notebook structure and inert execution**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_paper_replication_notebook -v
UV_CACHE_DIR=/tmp/coral-uv-cache uvx marimo check notebooks/evaluate_gpt56_sol_paper_replication.py
env -u CORAL_PAPER_REPLICATION_RUN -u CORAL_PAPER_REPLICATION_CONFIRM \
  UV_CACHE_DIR=/tmp/coral-uv-cache uv run notebooks/evaluate_gpt56_sol_paper_replication.py
```

Expected: tests PASS; marimo reports no errors; script prints setup/inert guidance, creates no OpenAI client, and makes no Azure request.

- [ ] **Step 6: Run the complete offline regression suite**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest discover -s tests -p 'test_paper_replication*.py' -v
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest discover -s tests -v
git diff --check
```

Expected: all paper-replication tests PASS, the full existing suite PASS, and `git diff --check` prints nothing. No command in this step may set `CORAL_PAPER_REPLICATION_RUN`, construct a paid client, or alter an existing `gpt56_sol_azure` artifact.

- [ ] **Step 7: Inspect the final diff and commit Task 5**

Confirm the staged paths contain only this feature:

```bash
git status --short
git diff -- notebooks/evaluate_gpt56_sol_paper_replication.py tests/test_paper_replication_notebook.py coral/paper_replication
```

Commit only the notebook feature files:

```bash
git add notebooks/evaluate_gpt56_sol_paper_replication.py tests/test_paper_replication_notebook.py coral/paper_replication/__init__.py
git commit -m "feat: add GPT-5.6 paper replication notebook"
```

- [ ] **Step 8: Perform completion verification and hand off the paid smoke command**

Read and follow `superpowers:verification-before-completion`, rerun its required fresh checks, and report that automated verification made no Azure calls. Do not execute the paid smoke run on the user's behalf without a new explicit confirmation at execution time.

Provide these commands for the user-controlled smoke run:

```bash
test -n "${AZURE_OPENAI_API_KEY:-}" && test -n "${AZURE_OPENAI_ENDPOINT:-}"
CORAL_PAPER_REPLICATION_RUN=smoke \
  UV_CACHE_DIR=/tmp/coral-uv-cache \
  uv run notebooks/evaluate_gpt56_sol_paper_replication.py
```

After the user confirms the eight completed checkpoint rows and top-line artifact, the deliberate full-run command is:

```bash
CORAL_PAPER_REPLICATION_RUN=full \
CORAL_PAPER_REPLICATION_CONFIRM='RUN 1120' \
UV_CACHE_DIR=/tmp/coral-uv-cache \
uv run notebooks/evaluate_gpt56_sol_paper_replication.py
```

The full run resumes completed smoke keys and requests only the remaining 1,112 keys, subject to the configured spend cap.
