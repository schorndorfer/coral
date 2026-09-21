# Structured-Output 14B Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable, locally executed Qwen2.5-14B inference path that produces validated task-specific JSON with verbatim evidence and converts it to the existing CORAL scorer format.

**Architecture:** A new `coral.structured` package owns Pydantic schemas, JSON/evidence validation, retry orchestration, JSONL checkpointing, and legacy conversion. A thin benchmarking entry point reuses the existing Hugging Face model loader, performs deterministic batched generation on MPS, retries invalid responses once, and records canonical results plus run metadata before producing legacy CSV output.

**Tech Stack:** Python 3.12, Pydantic 2.x, PyTorch MPS, Hugging Face Transformers, pandas, standard-library JSON/JSONL, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-17-structured-output-14b-evaluation-design.md`

## Global Constraints

- Run all Python commands through `uv run` with `UV_CACHE_DIR=/tmp/coral-uv-cache`.
- Keep clinical text, model weights, prompts, and responses local.
- Use Qwen2.5-14B-Instruct in FP16 on MPS with deterministic greedy decoding.
- Begin model smoke testing at batch size 2.
- Use 512 new tokens for symptoms, genomics, biomarkers, histology, stage, TNM, and grade; use 1,024 for radiology, procedures, metastasis, and medication tasks.
- Require at least one verified verbatim evidence quote per accepted record.
- Retry invalid responses at most once and never silently turn a validation failure into a successful empty response.
- Treat canonical JSONL as the source of truth; legacy tuples exist only at the scorer boundary.
- Do not modify the known Cartesian-product flaw in `coral/benchmarking/evaluate_model.py` in this change.
- Preserve unrelated user changes in the working tree.

## File Structure

- Create `coral/structured/__init__.py`: public structured-output interfaces.
- Create `coral/structured/schemas.py`: Pydantic task record models, task dispatch, JSON schemas, and token limits.
- Create `coral/structured/prompts.py`: schema-bearing initial and corrective retry prompts.
- Create `coral/structured/validation.py`: safe JSON extraction, Pydantic validation, and verbatim-evidence checks.
- Create `coral/structured/generation.py`: two-attempt generation orchestration and result metadata.
- Create `coral/structured/checkpoint.py`: JSONL append, malformed-line handling, and resume keys.
- Create `coral/structured/legacy.py`: validated-record conversion to current CORAL named tuples and scorer CSV rows.
- Create `coral/benchmarking/structured_benchmarking.py`: CLI, model execution, batching, resume, provenance, and run summary.
- Create `tests/test_structured_schemas.py`: all task schemas and prompt-schema tests.
- Create `tests/test_structured_validation.py`: JSON and evidence validation tests.
- Create `tests/test_structured_generation.py`: first-pass, retry, and terminal-failure tests.
- Create `tests/test_structured_checkpoint.py`: JSONL persistence and resume tests.
- Create `tests/test_structured_legacy.py`: conversion coverage for all 14 tasks.
- Create `tests/test_structured_benchmarking.py`: runner behavior using fake generation callbacks.
- Modify `README.md`: add Pydantic installation and structured 14B commands.

---

### Task 1: Define all task schemas and schema-bearing prompts

**Files:**
- Create: `coral/structured/__init__.py`
- Create: `coral/structured/schemas.py`
- Create: `coral/structured/prompts.py`
- Create: `tests/test_structured_schemas.py`
- Modify: `README.md:52-56`

**Interfaces:**
- Produces: `TASK_RECORD_MODELS: dict[str, type[BaseRecord]]`
- Produces: `TASK_MAX_NEW_TOKENS: dict[str, int]`
- Produces: `validate_task_payload(payload: object, expected_task: str) -> ValidatedTaskResponse`
- Produces: `task_response_json_schema(task: str) -> dict[str, object]`
- Produces: `build_initial_messages(task: str, section_text: str, model_name_or_path: str) -> list[dict[str, str]]`

- [ ] **Step 1: Install the declared schema dependency**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv pip install "pydantic>=2,<3"
```

Expected: Pydantic 2.x installs into `.venv` without changing global Python.

- [ ] **Step 2: Write failing schema and prompt tests**

Create `tests/test_structured_schemas.py` with `unittest.TestCase` coverage that:

```python
EXPECTED_TASKS = {
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
}

class StructuredSchemaTests(unittest.TestCase):
    def test_all_fourteen_tasks_have_models_and_token_limits(self):
        self.assertEqual(set(TASK_RECORD_MODELS), EXPECTED_TASKS)
        self.assertEqual(set(TASK_MAX_NEW_TOKENS), EXPECTED_TASKS)

    def test_records_forbid_unknown_fields(self):
        payload = {
            "task": "symptoms",
            "records": [{
                "symptom": "pain",
                "datetimes": [],
                "evidence_quotes": ["reports pain"],
                "unexpected": True,
            }],
        }
        with self.assertRaises(ValidationError):
            validate_task_payload(payload, "symptoms")

    def test_medication_enums_are_enforced(self):
        payload = {
            "task": "future_med_consideration_ae",
            "records": [{
                "medication_name": "example drug",
                "consideration": "maybe someday",
                "potential_adverse_events": [],
                "evidence_quotes": ["example drug may be considered"],
            }],
        }
        with self.assertRaises(ValidationError):
            validate_task_payload(payload, "future_med_consideration_ae")

    def test_prompt_contains_selected_schema_and_note(self):
        messages = build_initial_messages("symptoms", "Patient reports pain.", "qwen")
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn('"evidence_quotes"', rendered)
        self.assertIn("Patient reports pain.", rendered)
        self.assertIn("Return exactly one JSON object", rendered)
```

Add table-driven valid-payload cases for every record model and assert that the
validated record class and serialized field names match the table below.

| Record model | Fields in addition to `evidence_quotes` |
|---|---|
| `SymptomRecord` | `symptom`, `datetimes` |
| `RadiologyRecord` | `radiology_test`, `datetimes`, `sites`, `reasons`, `results` |
| `ProcedureRecord` | `procedure_name`, `datetimes`, `sites`, `reasons`, `results` |
| `GenomicsRecord` | `genomic_test_name`, `datetimes`, `results` |
| `BiomarkerRecord` | `biomarker`, `datetimes` |
| `HistologyRecord` | `histology`, `datetimes` |
| `MetastasisRecord` | `metastasis`, `sites`, `procedures`, `datetimes` |
| `StageRecord` | `stage`, `datetimes`, `additional_testing` |
| `TnmRecord` | `tnm`, `datetimes`, `additional_testing` |
| `GradeRecord` | `grade`, `datetimes`, `additional_testing` |
| `PrescribedMedicationRecord` | `medication_name`, `begins`, `ends`, `reasons`, `continuity`, `confirmed_adverse_events`, `potential_adverse_events` |
| `FutureMedicationRecord` | `medication_name`, `consideration`, `potential_adverse_events` |

- [ ] **Step 3: Run the schema tests and confirm the expected import failure**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_schemas -v
```

Expected: FAIL because `coral.structured.schemas` does not exist.

- [ ] **Step 4: Implement the Pydantic schemas and dispatch**

Use this shared configuration in `schemas.py`:

```python
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

NonEmpty = str

class BaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_quotes: list[str] = Field(min_length=1)

class SymptomRecord(BaseRecord):
    symptom: str = Field(min_length=1)
    datetimes: list[str] = Field(default_factory=list)

class PrescribedMedicationRecord(BaseRecord):
    medication_name: str = Field(min_length=1)
    begins: list[str] = Field(default_factory=list)
    ends: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    continuity: Literal["continuing", "finished", "discontinued early"] | None
    confirmed_adverse_events: list[str] = Field(default_factory=list)
    potential_adverse_events: list[str] = Field(default_factory=list)

class FutureMedicationRecord(BaseRecord):
    medication_name: str = Field(min_length=1)
    consideration: Literal["planned", "hypothetical"] | None
    potential_adverse_events: list[str] = Field(default_factory=list)

@dataclass(frozen=True)
class ValidatedTaskResponse:
    task: str
    records: tuple[BaseRecord, ...]

    def model_dump(self) -> dict[str, object]:
        return {
            "task": self.task,
            "records": [record.model_dump(mode="json") for record in self.records],
        }
```

Implement every remaining model using the exact field table above. All anchor
fields are non-empty strings, collection fields default to empty lists, and
`additional_testing` is a list. Define `TASK_RECORD_MODELS` explicitly for all
14 task names; the three symptom tasks map to `SymptomRecord`.

Implement `validate_task_payload` by requiring exactly `task` and `records`,
checking `payload["task"] == expected_task`, then validating records with
`TypeAdapter(list[TASK_RECORD_MODELS[expected_task]])`. Generate the response
JSON schema as an object with a constant `task`, an array of the selected
record schema, `required=["task", "records"]`, and
`additionalProperties=False`.

Set 1,024-token limits for radiology, procedures, metastasis, prescribed
medications, and future medications; set 512 for the other nine tasks.

- [ ] **Step 5: Implement structured prompts**

In `prompts.py`, reuse `OncPrompt.get_system_message()` and define an explicit
`STRUCTURED_TASK_INSTRUCTIONS` mapping. Do not parse or modify the legacy prompt
strings at runtime, because their examples and post-example constraints are not
laid out consistently. Copy the clinical inclusion/exclusion rules while
omitting all named-tuple syntax according to this complete requirements table:

| Task | Required clinical instruction |
|---|---|
| `symptoms` | Current experienced symptoms with onset dates; exclude diagnoses, findings, tests, procedures, possible side effects, and explicitly absent symptoms. |
| `symptoms_at_diagnosis` | Symptoms present before or at first cancer diagnosis with onset dates; exclude later symptoms and non-symptom clinical concepts. |
| `symptoms_due_to_cancer` | Symptoms likely caused by cancer with onset dates; exclude diagnoses, findings, tests, and procedures. |
| `radtest_datetime_site_reason_result` | Oncology-relevant radiology studies with date, lateralized site, reason, and oncology-relevant result. |
| `procedure_datetime_site_reason_result` | Cancer-directed diagnostic/interventional procedures with bleeding risk, date, lateralized site, reason, and result; exclude radiology. |
| `genomictest_datetime_result` | Genomic/genetic tests with date and result; exclude radiology, surgery, and non-genomic biomarkers. |
| `biomarker_datetime` | Treatment-relevant biomarkers for the main cancer with identification dates; exclude radiology findings. |
| `histology_datetime` | Morphologic histology for the main cancer with identification dates; exclude radiology findings. |
| `metastasis_site_procedure_datetime` | Evidence of metastatic spread with site, identifying procedure, and procedure date. |
| `stage_datetime_addtest` | Non-TNM stage for the main cancer with date and additional testing needed to complete staging. |
| `tnm_datetime_addtest` | TNM stage for the main cancer with date and additional testing needed to complete staging. |
| `grade_datetime_addtest` | Combined Nottingham grade with date and additional testing needed to complete grading. |
| `prescribed_med_begin_end_reason_continuity_ae` | Prescribed cancer-directed medication with start/end, reason, continuity, confirmed and potential adverse events; exclude merely planned/hypothetical therapies. |
| `future_med_consideration_ae` | Planned or hypothetical future cancer-directed medication or drug class with potential adverse events; exclude previously or currently prescribed therapy. |

Append the selected task's JSON schema using:

```python
STRUCTURED_INSTRUCTION = """
Return exactly one JSON object matching this JSON Schema:
{schema}
Use null for unavailable scalar values and [] for unavailable collections.
Every record must include at least one evidence_quotes entry copied verbatim
from the clinical note. Do not include explanations or Markdown fences.
"""
```

Keep note text and instructions separated with two newlines. Serialize the
schema using `json.dumps(schema, sort_keys=True)` for deterministic prompts.

- [ ] **Step 6: Update dependency documentation and run tests**

Add `pydantic>=2,<3` to the README installation command. Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_schemas -v
```

Expected: all schema and prompt tests PASS.

- [ ] **Step 7: Commit the schema layer**

```bash
git add README.md coral/structured/__init__.py coral/structured/schemas.py coral/structured/prompts.py tests/test_structured_schemas.py
git commit -m "feat: define structured oncology schemas"
```

---

### Task 2: Validate JSON and verbatim evidence safely

**Files:**
- Create: `coral/structured/validation.py`
- Create: `tests/test_structured_validation.py`

**Interfaces:**
- Consumes: `validate_task_payload()` and `ValidatedTaskResponse` from Task 1.
- Produces: `extract_json_object(raw: str) -> object`
- Produces: `validate_evidence(response: ValidatedTaskResponse, section_text: str) -> None`
- Produces: `parse_and_validate(raw: str, task: str, section_text: str) -> ValidatedTaskResponse`
- Raises: `StructuredOutputError(errors: tuple[str, ...])`

- [ ] **Step 1: Write failing JSON and evidence tests**

Cover these exact cases in `tests/test_structured_validation.py`:

```python
class StructuredValidationTests(unittest.TestCase):
    def test_accepts_plain_json(self):
        self.assertEqual(extract_json_object('{"task":"symptoms","records":[]}')["task"], "symptoms")

    def test_accepts_single_json_fence(self):
        raw = '```json\n{"task":"symptoms","records":[]}\n```'
        self.assertEqual(extract_json_object(raw)["records"], [])

    def test_rejects_prose_around_json(self):
        with self.assertRaises(StructuredOutputError):
            extract_json_object('Here is the result: {"task":"symptoms","records":[]}')

    def test_rejects_quote_not_present_in_note(self):
        raw = json.dumps({"task": "symptoms", "records": [{
            "symptom": "pain", "datetimes": [],
            "evidence_quotes": ["fabricated quotation"],
        }]})
        with self.assertRaisesRegex(StructuredOutputError, "evidence_quotes"):
            parse_and_validate(raw, "symptoms", "Patient reports pain.")

    def test_normalizes_crlf_for_quote_matching(self):
        raw = json.dumps({"task": "symptoms", "records": [{
            "symptom": "pain", "datetimes": [],
            "evidence_quotes": ["reports pain.\nToday"],
        }]})
        response = parse_and_validate(raw, "symptoms", "Patient reports pain.\r\nToday")
        self.assertEqual(len(response.records), 1)
```

Also test invalid JSON, wrong task, missing required evidence, empty evidence,
wrong enum values, and extra top-level fields.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_validation -v
```

Expected: FAIL because `validation.py` does not exist.

- [ ] **Step 3: Implement strict extraction and validation**

Implement `extract_json_object` using `json.loads` on either the stripped raw
text or the contents of one complete ```` ```json ... ``` ```` fence. Do not
use `eval`, `ast.literal_eval`, or brace-scanning that accepts surrounding prose.

Implement `StructuredOutputError` with an immutable `errors` tuple and a joined
human-readable message. Convert `JSONDecodeError`, Pydantic `ValidationError`,
unknown task names, and evidence failures into this exception.

Normalize only `\r\n` and bare `\r` to `\n`. For every record and every quote,
require `quote.strip() == quote`, require a non-empty quote, and require the
normalized quote to occur exactly in normalized `section_text`.

- [ ] **Step 4: Run validation tests**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_validation -v
```

Expected: all validation tests PASS.

- [ ] **Step 5: Commit validation**

```bash
git add coral/structured/validation.py tests/test_structured_validation.py
git commit -m "feat: validate structured extraction evidence"
```

---

### Task 3: Add one-retry generation orchestration

**Files:**
- Modify: `coral/structured/prompts.py`
- Create: `coral/structured/generation.py`
- Create: `tests/test_structured_generation.py`

**Interfaces:**
- Consumes: `parse_and_validate()` from Task 2.
- Produces: `build_retry_messages(task: str, section_text: str, model_name_or_path: str, raw: str, errors: tuple[str, ...]) -> list[dict[str, str]]`
- Produces: `GenerationAttempt(raw: str, errors: tuple[str, ...])`
- Produces: `StructuredGenerationResult(task, response, attempts, validation_status, elapsed_seconds)`
- Produces: `validate_with_retry(task: str, section_text: str, model_name: str, initial_raw: str, retry_generate: Callable[[list[dict[str, str]], int], str]) -> StructuredGenerationResult`
- Callback: `retry_generate(messages: list[dict[str, str]], max_new_tokens: int) -> str`

- [ ] **Step 1: Write failing orchestration tests**

Use a queue-backed fake generator and cover:

```python
def generator_from(*responses):
    queued = iter(responses)
    calls = []
    def generate(messages, max_new_tokens):
        calls.append((messages, max_new_tokens))
        return next(queued)
    return generate, calls

class StructuredGenerationTests(unittest.TestCase):
    def test_valid_first_attempt_does_not_retry(self):
        valid = json.dumps({"task": "symptoms", "records": []})
        retry, calls = generator_from("unused")
        result = validate_with_retry("symptoms", "No symptoms.", "qwen", valid, retry)
        self.assertEqual(result.validation_status, "valid")
        self.assertEqual(calls, [])

    def test_invalid_first_attempt_retries_with_errors(self):
        valid = json.dumps({"task": "symptoms", "records": []})
        retry, calls = generator_from(valid)
        result = validate_with_retry("symptoms", "No symptoms.", "qwen", "not json", retry)
        self.assertEqual(result.validation_status, "valid_after_retry")
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("not json", str(calls[0][0]))
        self.assertIn("valid JSON", str(calls[0][0]))

    def test_second_failure_is_terminal_and_preserves_both_raw_responses(self):
        retry, calls = generator_from("still not json")
        result = validate_with_retry("symptoms", "No symptoms.", "qwen", "not json", retry)
        self.assertEqual(result.validation_status, "validation_failed")
        self.assertIsNone(result.response)
        self.assertEqual([attempt.raw for attempt in result.attempts],
                         ["not json", "still not json"])

    def test_task_token_limit_is_passed_to_retry(self):
        valid = json.dumps({"task": "radtest_datetime_site_reason_result", "records": []})
        retry, calls = generator_from(valid)
        validate_with_retry("radtest_datetime_site_reason_result", "No imaging.",
                            "qwen", "not json", retry)
        self.assertEqual(calls[0][1], 1024)
```

For the retry-success case, assert two calls, `retry_count == 1`, status
`"valid_after_retry"`, and that the second prompt contains both the invalid raw
response and the first attempt's validation error. For terminal failure, assert
status `"validation_failed"`, `response is None`, and two preserved attempts.

- [ ] **Step 2: Run the orchestration tests and verify failure**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_generation -v
```

Expected: FAIL because `generation.py` does not exist.

- [ ] **Step 3: Implement retry prompts and generation results**

Use frozen dataclasses. `GenerationAttempt` stores raw text plus validation
errors. `StructuredGenerationResult` exposes `retry_count` as
`max(0, len(attempts) - 1)` and serializes attempts without discarding errors.

`validate_with_retry` first validates the supplied `initial_raw` response. It
returns immediately after valid output; otherwise it builds a corrective prompt
and calls `retry_generate` exactly once. The corrective prompt repeats the
schema and source note, quotes the invalid response as data, lists each
validation error, and asks for one complete JSON replacement without Markdown.

- [ ] **Step 4: Run orchestration tests**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_generation -v
```

Expected: all orchestration tests PASS.

- [ ] **Step 5: Commit retry orchestration**

```bash
git add coral/structured/prompts.py coral/structured/generation.py tests/test_structured_generation.py
git commit -m "feat: retry invalid structured generations"
```

---

### Task 4: Add durable JSONL checkpointing and resume

**Files:**
- Create: `coral/structured/checkpoint.py`
- Create: `tests/test_structured_checkpoint.py`

**Interfaces:**
- Produces: `RunKey(doc_idx: str, section_name: str, task: str, model: str, schema_version: str)`
- Produces: `load_completed_keys(path: Path) -> tuple[set[RunKey], tuple[str, ...]]`
- Produces: `append_checkpoint(path: Path, record: dict[str, object]) -> None`

- [ ] **Step 1: Write failing checkpoint tests**

Use `tempfile.TemporaryDirectory()` to verify:

```python
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "results.jsonl"
        self.base = {
            "doc_idx": "1", "section_name": "hpi", "task": "symptoms",
            "model": "qwen", "schema_version": "1", "validation_status": "valid",
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def test_append_then_load_completed_key(self):
        append_checkpoint(self.path, self.base)
        keys, warnings = load_completed_keys(self.path)
        self.assertEqual(len(keys), 1)
        self.assertEqual(warnings, ())

    def test_duplicate_key_is_returned_once(self):
        append_checkpoint(self.path, self.base)
        append_checkpoint(self.path, self.base)
        keys, _ = load_completed_keys(self.path)
        self.assertEqual(len(keys), 1)

    def test_malformed_line_is_reported_and_ignored(self):
        self.path.write_text("not-json\n" + json.dumps(self.base) + "\n")
        keys, warnings = load_completed_keys(self.path)
        self.assertEqual(len(keys), 1)
        self.assertEqual(warnings, ("ignored malformed checkpoint line 1",))

    def test_validation_failed_record_counts_as_terminal(self):
        append_checkpoint(self.path, {**self.base, "validation_status": "validation_failed"})
        keys, _ = load_completed_keys(self.path)
        self.assertEqual(len(keys), 1)

    def test_incomplete_record_without_status_is_not_completed(self):
        incomplete = {key: value for key, value in self.base.items()
                      if key != "validation_status"}
        append_checkpoint(self.path, incomplete)
        keys, warnings = load_completed_keys(self.path)
        self.assertEqual(keys, set())
        self.assertEqual(warnings, ("ignored incomplete checkpoint line 1",))
```

The record fixture must include all five key fields and a terminal
`validation_status` of `valid`, `valid_after_retry`, or `validation_failed`.

- [ ] **Step 2: Run checkpoint tests and verify failure**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_checkpoint -v
```

Expected: FAIL because `checkpoint.py` does not exist.

- [ ] **Step 3: Implement append and recovery**

Open the checkpoint with UTF-8 and append one `json.dumps(record,
ensure_ascii=False, sort_keys=True)` line. Flush after every line. On load,
enumerate lines from 1, parse JSON, validate required key/status fields, collect
terminal keys, and return sanitized warnings such as
`"ignored malformed checkpoint line 3"` without including clinical content.

- [ ] **Step 4: Run checkpoint tests**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_checkpoint -v
```

Expected: all checkpoint tests PASS.

- [ ] **Step 5: Commit checkpointing**

```bash
git add coral/structured/checkpoint.py tests/test_structured_checkpoint.py
git commit -m "feat: checkpoint structured inference runs"
```

---

### Task 5: Convert every structured task to legacy tuples

**Files:**
- Create: `coral/structured/legacy.py`
- Create: `tests/test_structured_legacy.py`

**Interfaces:**
- Consumes: `ValidatedTaskResponse` and current named tuples from `coral.__init__`.
- Produces: `to_legacy_tuples(response: ValidatedTaskResponse) -> list[tuple]`
- Produces: `to_legacy_output(response: ValidatedTaskResponse) -> str`
- Produces: `checkpoint_to_legacy_row(record: dict[str, object], section_text: str) -> dict[str, object]`

- [ ] **Step 1: Write failing table-driven adapter tests**

Create one validated record for every task and assert the exact existing tuple
class and field conversion. Include these boundary assertions:

```python
self.assertEqual(empty_list_to_legacy([]), {"unknown"})
self.assertEqual(optional_scalar_to_legacy(None), "unknown")
self.assertNotIn("evidence", to_legacy_output(response).lower())
```

For a `validation_failed` checkpoint, pass its source `section_text` explicitly
and assert an output row exists with the task's
`task_to_default_tuple_dict` string and
`conversion_status == "validation_failed"`. For successful checkpoints, assert
`conversion_status == "validated"`.

- [ ] **Step 2: Run adapter tests and verify failure**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_legacy -v
```

Expected: FAIL because `legacy.py` does not exist.

- [ ] **Step 3: Implement explicit mappings for all 14 tasks**

Use an explicit task dispatch rather than positional reflection. Map plural JSON
fields to the sets expected by `SymptomEnt`, `RadTest`, `Proc`, `Genomics`,
`TxBiomarker`, `Histo`, `MetastasisEnt`, `StageEnt`, `TnmEnt`, `GradeEnt`,
`PrescribedMedEnt`, and `FutureMedEnt`. The three symptom task names share one
converter. Preserve list contents as sets and perform `unknown` conversion only
inside this module. `checkpoint_to_legacy_row` returns exactly these columns:

```python
{
    "doc_idx": record["doc_idx"],
    "section_name": record["section_name"],
    "section_text": section_text,
    "task": record["task"],
    "model": record["model"],
    "output": converted_namedtuple_lines,
    "conversion_status": "validated" if record["response"] else "validation_failed",
}
```

- [ ] **Step 4: Run adapter tests**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_legacy -v
```

Expected: all adapter tests PASS.

- [ ] **Step 5: Commit legacy conversion**

```bash
git add coral/structured/legacy.py tests/test_structured_legacy.py
git commit -m "feat: adapt structured results for legacy scoring"
```

---

### Task 6: Build the resumable structured benchmarking CLI

**Files:**
- Create: `coral/benchmarking/structured_benchmarking.py`
- Create: `tests/test_structured_benchmarking.py`

**Interfaces:**
- Consumes: Tasks 1-5 and `load_model`, `load_tokenizer`, `format_hf_chat_template` from `open_source_benchmarking.py`.
- Produces: `run_structured_benchmark(dataframe: pd.DataFrame, *, model_identifier: str, model_revision: str | None, jsonl_path: Path, legacy_csv_path: Path, summary_path: Path, batch_size: int, generate_batch: Callable[[list[list[dict[str, str]]], int], list[str]], limit: int | None = None, peak_memory_bytes: Callable[[], int] = lambda: 0) -> dict[str, object]`
- Produces CLI arguments: `-fdata`, `-dir_data`, `-model_name_or_path`, `-fjsonl`, `-flegacy`, `-dir_out`, `-batch_size`, `-limit`.

- [ ] **Step 1: Write failing runner tests with fake model generation**

Tests must avoid loading a real model. Inject a callback with signature
`generate_batch(batch_messages, max_new_tokens) -> list[str]` and cover:

```python
TASK_PATTERN = re.compile(r'"const":\s*"([^"]+)"')

def valid_empty_batch(messages, max_new_tokens):
    outputs = []
    for conversation in messages:
        prompt = "\n".join(item["content"] for item in conversation)
        task = TASK_PATTERN.search(prompt).group(1)
        outputs.append(json.dumps({"task": task, "records": []}))
    return outputs

class StructuredBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.jsonl = root / "results.jsonl"
        self.legacy = root / "legacy.csv"
        self.summary = root / "summary.json"
        self.dataframe = pd.DataFrame([
            {"doc_idx": "1", "section_name": "hpi", "task": "symptoms",
             "section_text": "No symptoms."},
            {"doc_idx": "1", "section_name": "a&p",
             "task": "prescribed_med_begin_end_reason_continuity_ae",
             "section_text": "No treatment."},
        ])

    def tearDown(self):
        self.tempdir.cleanup()

    def run_benchmark(self, generate_batch=valid_empty_batch, limit=None):
        return run_structured_benchmark(
            self.dataframe, model_identifier="qwen", model_revision=None,
            jsonl_path=self.jsonl, legacy_csv_path=self.legacy,
            summary_path=self.summary, batch_size=2,
            generate_batch=generate_batch, limit=limit,
            peak_memory_bytes=lambda: 1234,
        )

    def test_processes_each_input_once_and_writes_jsonl(self):
        summary = self.run_benchmark()
        rows = [json.loads(line) for line in self.jsonl.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["total"], 2)

    def test_resume_skips_terminal_keys(self):
        self.run_benchmark()
        calls = []
        def recording_generate(messages, max_new_tokens):
            calls.append(messages)
            return valid_empty_batch(messages, max_new_tokens)
        summary = self.run_benchmark(recording_generate)
        self.assertEqual(calls, [])
        self.assertEqual(summary["skipped_on_resume"], 2)

    def test_limit_stops_after_requested_inputs(self):
        self.run_benchmark(limit=1)
        self.assertEqual(len(self.jsonl.read_text().splitlines()), 1)

    def test_mixed_token_limits_are_split_into_separate_batches(self):
        limits = []
        def recording_generate(messages, max_new_tokens):
            limits.append(max_new_tokens)
            return valid_empty_batch(messages, max_new_tokens)
        self.run_benchmark(recording_generate)
        self.assertEqual(limits, [512, 1024])

    def test_writes_legacy_csv_with_conversion_status(self):
        self.run_benchmark()
        legacy = pd.read_csv(self.legacy)
        self.assertEqual(len(legacy), 2)
        self.assertEqual(set(legacy["conversion_status"]), {"validated"})

    def test_summary_counts_retry_and_failure(self):
        responses = iter([
            ["not json"],
            [json.dumps({"task": "symptoms", "records": []})],
            ["not json"],
            ["still not json"],
        ])
        def queued_generate(messages, max_new_tokens):
            return next(responses)
        summary = self.run_benchmark(queued_generate)
        self.assertEqual(summary["valid_after_retry"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["peak_mps_allocated_bytes"], 1234)
```

Use a synthetic dataframe with one symptom row and one medication row. Assert
that task groups with 512 and 1,024 token limits are never passed in the same
generation batch.

- [ ] **Step 2: Run runner tests and verify failure**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_benchmarking -v
```

Expected: FAIL because `structured_benchmarking.py` does not exist.

- [ ] **Step 3: Implement model-independent runner logic**

Read inference data through `read_data(..., get_annots=False)`, normalize
`doc_idx` to string, and build `RunKey` values. Skip completed keys before
batching. Group pending rows by `TASK_MAX_NEW_TOKENS`, then batch within each
group.

For each generated response, call the validation/retry orchestrator, append a
canonical checkpoint containing:

```python
{
    "doc_idx": str(row["doc_idx"]),
    "section_name": row["section_name"],
    "task": row["task"],
    "model": model_identifier,
    "model_revision": model_revision,
    "schema_version": SCHEMA_VERSION,
    "decoding": {"do_sample": False, "max_new_tokens": task_limit},
    "elapsed_seconds": result.elapsed_seconds,
    "validation_status": result.validation_status,
    "retry_count": result.retry_count,
    "attempts": [attempt.to_dict() for attempt in result.attempts],
    "response": result.response.model_dump() if result.response else None,
}
```

After every append, emit or refresh the legacy CSV from terminal checkpoint
records. Write a sibling summary JSON containing total, valid-first-pass,
valid-after-retry, failed, skipped-on-resume, elapsed seconds, throughput, and
peak MPS allocated memory.

- [ ] **Step 4: Implement the Hugging Face CLI wrapper**

Load the model/tokenizer through existing helpers. The batch callback must call
`get_model_response(..., do_sample=False, max_new_tokens=selected_limit)`.
Reject non-MPS execution on macOS unless an explicit `--allow_cpu` flag is
provided. `-limit` defaults to no limit and exists for smoke tests.

Use `model.config._commit_hash` when present; otherwise store `null`. Set the
model identifier to the resolved `model.name_or_path`. Seed PyTorch with zero
even though greedy generation should be deterministic.

- [ ] **Step 5: Run runner tests**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_structured_benchmarking -v
```

Expected: all runner tests PASS without loading model weights.

- [ ] **Step 6: Commit the CLI**

```bash
git add coral/benchmarking/structured_benchmarking.py tests/test_structured_benchmarking.py
git commit -m "feat: run resumable structured benchmarks"
```

---

### Task 7: Document, verify, and run Qwen2.5-14B

**Files:**
- Modify: `README.md`
- Runtime artifact: `models/qwen-2.5-14b-instruct/` (gitignored)
- Runtime artifact: `output/qwen25_14b_structured.jsonl` (gitignored)
- Runtime artifact: `output/qwen25_14b_structured_legacy.csv` (gitignored)
- Runtime artifact: `output/qwen25_14b_structured.summary.json` (gitignored)

**Interfaces:**
- Consumes: completed CLI from Task 6.
- Produces: documented commands and verified benchmark artifacts.

- [ ] **Step 1: Add exact structured benchmark commands to README**

Document download, smoke, resume, and full-run commands:

```bash
zsh -ic 'UV_CACHE_DIR=/tmp/coral-uv-cache uv run hf download \
  Qwen/Qwen2.5-14B-Instruct \
  --local-dir ./models/qwen-2.5-14b-instruct'

PYTORCH_ENABLE_MPS_FALLBACK=1 UV_CACHE_DIR=/tmp/coral-uv-cache \
uv run python -m coral.benchmarking.structured_benchmarking \
  -fdata coral_inference.csv -dir_data ./data \
  -model_name_or_path ./models/qwen-2.5-14b-instruct \
  -fjsonl qwen25_14b_structured_smoke.jsonl \
  -flegacy qwen25_14b_structured_smoke_legacy.csv \
  -dir_out ./output -batch_size 2 -limit 8
```

State that rerunning the same full command resumes terminal keys from JSONL.

- [ ] **Step 2: Run the full automated test suite**

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest discover -s tests -v
```

Expected: all existing and structured-output tests PASS.

- [ ] **Step 3: Download the 14B model**

Run the documented authenticated `hf download` command. Expected: exit 0 and
all safetensor shards plus the index are present under
`models/qwen-2.5-14b-instruct/`.

- [ ] **Step 4: Run and inspect the eight-input smoke benchmark**

Run the documented smoke command outside the sandbox for MPS access. Verify:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -c '
import json
from pathlib import Path
p = Path("output/qwen25_14b_structured_smoke.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines()]
assert len(rows) == 8
assert all(row["validation_status"] in {
    "valid", "valid_after_retry", "validation_failed"
} for row in rows)
assert all(row["response"] is None or all(
    record["evidence_quotes"] for record in row["response"]["records"]
) for row in rows)
print({status: sum(row["validation_status"] == status for row in rows)
       for status in {row["validation_status"] for row in rows}})
'
```

Expected: assertions PASS. Review summary memory and latency before retaining
batch size 2 or increasing it to 4.

- [ ] **Step 5: Run the full 515-input structured benchmark**

Use a new output name so smoke rows do not mix with the full run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 UV_CACHE_DIR=/tmp/coral-uv-cache \
uv run python -m coral.benchmarking.structured_benchmarking \
  -fdata coral_inference.csv -dir_data ./data \
  -model_name_or_path ./models/qwen-2.5-14b-instruct \
  -fjsonl qwen25_14b_structured.jsonl \
  -flegacy qwen25_14b_structured_legacy.csv \
  -dir_out ./output -batch_size 2
```

Expected: 515 terminal JSONL records and a summary with counts totaling 515.

- [ ] **Step 6: Run the unchanged legacy scorer and actual-prompt analysis**

Run the unchanged scorer outside the sandbox because Hugging Face Evaluate
writes lock files under `~/.cache`:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m coral.benchmarking.evaluate_model \
  -fdata coral_inference.csv \
  -fout qwen25_14b_structured_legacy.csv \
  -dir_data ./data -dir_out ./output \
  -fscores_inst qwen25_14b_relation_instance_scores.csv \
  -fscores_agg qwen25_14b_relation_aggregate_scores.csv \
  -fscores_reformatted qwen25_14b_relation_summary.csv
```

Write the actual-prompt-only summary without modifying the evaluator:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -c '
import pandas as pd
inputs = pd.read_csv("data/coral_inference.csv").rename(
    columns={"inference_subtype": "task"}
)
scores = pd.read_csv("output/qwen25_14b_relation_instance_scores.csv")
keys = ["doc_idx", "section_name", "task"]
observed = scores.merge(inputs[keys].drop_duplicates(), on=keys, how="inner")
summary = observed.groupby(["task", "subrelation"], as_index=False).agg(
    n_examples=("em_f1", "size"),
    mean_bleu4=("bleu4", "mean"),
    mean_rouge1=("rouge1", "mean"),
    mean_em_precision=("em_prec", "mean"),
    mean_em_recall=("em_recall", "mean"),
    mean_em_f1=("em_f1", "mean"),
)
summary.to_csv("output/qwen25_14b_relation_observed_summary.csv", index=False)
print(summary[["mean_bleu4", "mean_rouge1", "mean_em_f1"]].mean())
'
```

Report both the flawed repository macro metrics and actual-prompt-only macro
metrics, labeling them explicitly.

- [ ] **Step 7: Perform final verification**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest discover -s tests -v
git diff --check
git status --short
```

Verify that the canonical JSONL has 515 unique run keys, every key has a
terminal status, all accepted evidence quotes occur in their source section,
the legacy CSV has 515 rows, and all metric values are finite and within
`[0, 1]`.

- [ ] **Step 8: Commit documentation**

```bash
git add README.md
git commit -m "docs: document structured 14b evaluation"
```

---

## Follow-up Roadmap: Improve Reliability and Consistency

This roadmap applies after the current 14B run finishes. Preserve the current
JSONL, raw attempts, validation errors, and legacy scores as the baseline before
changing prompts, schemas, decoding, or model selection.

- [ ] **Step 1: Quantify the baseline by task and failure type**

Read the canonical JSONL and produce a table grouped by task and
`validation_status`, with counts for malformed JSON, schema failures, and
verbatim-evidence failures. Separately compute relation-level BLEU-4,
ROUGE-1, and exact-match precision/recall/F1 for the accepted legacy rows.
Keep strict structured-validity metrics separate from legacy accuracy metrics.

- [ ] **Step 2: Add exact evidence-span selection**

Add a post-generation evidence selector that receives the source section and
the model's extracted records, then chooses each `evidence_quotes` value as an
exact substring of the source text. Preserve the model's raw quote and the
selected source span in checkpoint metadata. Add tests for paraphrased quotes,
multiple matching spans, CRLF normalization, and records with no valid span.
Run this selector before declaring a response validation failure.

- [ ] **Step 3: Constrain JSON decoding**

Integrate schema-guided decoding into the Hugging Face generation callback so
the model can emit only syntactically valid JSON with task-appropriate fields
and enum values. Keep a plain-generation fallback and record the decoding mode
in each checkpoint. Add fake-generator tests for constrained and fallback
paths, then rerun the eight-input smoke benchmark before a full evaluation.

- [ ] **Step 4: Tighten task schemas and prompts**

Require date-like formats in date fields, use `null` when a date is unavailable,
and add task-specific examples that distinguish clinical facts from section
headings such as `Interim History`. Keep source evidence selection separate from
fact extraction. Add prompt tests for every tightened rule and rerun the smoke
benchmark to measure malformed-output and evidence-failure changes.

- [ ] **Step 5: Tune on a held-out development slice**

Create a fixed development slice from the annotated inputs and reserve the
remaining inputs for evaluation. Compare prompt variants using structured
validity, evidence-grounding rate, retry recovery, and legacy relation F1. Do
not select prompts using the final evaluation slice.

- [ ] **Step 6: Compare stronger models under the same protocol**

Run each candidate model on the same development slice, with identical tasks,
schemas, decoding limits, and evidence validation. Record latency, peak MPS
memory, malformed JSON rate, evidence failure rate, and legacy relation scores.
Only promote a model to the full evaluation when it improves reliability at an
acceptable memory and runtime cost.

- [ ] **Step 7: Add dual score reporting**

Report the paper-compatible legacy metrics and the stricter structured metrics
side by side. If terminology normalization is added later, report ontology
normalized scores as a third track and retain the original text-level metrics
for comparability with CORAL results.
