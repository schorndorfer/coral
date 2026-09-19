# Azure GPT-5.6 Sol Evaluation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a marimo notebook that runs a resumable, cost-capped Azure OpenAI GPT-5.6 Sol smoke test and promotes it to a deliberate full CORAL evaluation.

**Architecture:** Put all non-UI behavior in `coral.azure_evaluation`, where it can be unit-tested with fake Azure clients. The marimo notebook is a documented UI that reads Azure environment settings, runs eight inputs by default, writes checkpoints, and triggers exports only after an explicit request.

**Tech Stack:** Python 3.12, marimo, `openai` Azure client, Pydantic, pandas, unittest, existing CORAL scorer.

**Spec:** `docs/superpowers/specs/2026-09-19-azure-gpt56-sol-evaluation-notebook-design.md`

## Global Constraints

- Read `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`, never `OPENAI_API_KEY`.
- Read `AZURE_OPENAI_DEPLOYMENT`, defaulting to `gpt-5.6-sol`, and `AZURE_OPENAI_API_VERSION` with a documented default.
- Default to eight inputs; only the literal confirmation `RUN 515` permits all 515 inputs.
- Enforce a configurable spend cap before every Azure request.
- Keep artifacts under `output/` with the `gpt56_sol_azure` prefix; never persist a credential.
- Run requests one at a time, retry a validation failure exactly once, and resume from terminal JSONL records.
- Keep `coral.benchmarking.evaluate_model` unchanged and label its repository-wide macro as flawed.
- Run Python commands with `UV_CACHE_DIR=/tmp/coral-uv-cache uv run`.

---

## File Structure

- `coral/azure_evaluation.py`: settings, cost accounting, task schemas, validation, Azure execution, checkpointing, legacy conversion, and observed-only summary helpers.
- `tests/test_azure_evaluation.py`: unit tests with temporary files and a fake Responses client; no credential or network access.
- `notebooks/evaluate_gpt56_sol_azure.py`: PEP 723 marimo UI and explicit paid-run controls.

### Task 1: Add settings, cost, and JSONL checkpoint helpers

**Files:**
- Create: `coral/azure_evaluation.py`
- Create: `tests/test_azure_evaluation.py`

**Interfaces:**
- Produces `AzureSettings`, `Usage`, and `CheckpointRecord` frozen dataclasses.
- Produces `load_azure_settings(env: Mapping[str, str]) -> AzureSettings`.
- Produces `estimate_cost(usage: Usage) -> float`, `can_afford(spent: float, projected: float, cap: float) -> bool`, `append_checkpoint(path: Path, record: CheckpointRecord) -> None`, and `terminal_keys(path: Path) -> set[tuple[str, str, str, str]]`.

- [ ] **Step 1: Write failing settings and price tests**

```python
def test_load_azure_settings_uses_only_azure_environment_names(self):
    settings = load_azure_settings({
        "AZURE_OPENAI_API_KEY": "secret",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    })
    self.assertEqual(settings.deployment, "gpt-5.6-sol")
    self.assertEqual(settings.endpoint, "https://example.openai.azure.com")

def test_load_azure_settings_rejects_missing_key(self):
    with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_API_KEY"):
        load_azure_settings({"AZURE_OPENAI_ENDPOINT": "https://example"})

def test_estimate_cost_uses_sol_rates(self):
    self.assertEqual(estimate_cost(Usage(500_000, 100_000)), 4.0)
```

- [ ] **Step 2: Verify RED**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: FAIL because `coral.azure_evaluation` does not exist.

- [ ] **Step 3: Implement the smallest settings and accounting API**

```python
SOL_INPUT_PER_MILLION = 4.0
SOL_OUTPUT_PER_MILLION = 20.0

@dataclass(frozen=True)
class AzureSettings:
    api_key: str
    endpoint: str
    deployment: str = "gpt-5.6-sol"
    api_version: str = "2025-04-01-preview"

@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

def estimate_cost(usage: Usage) -> float:
    return usage.input_tokens / 1_000_000 * SOL_INPUT_PER_MILLION + usage.output_tokens / 1_000_000 * SOL_OUTPUT_PER_MILLION
```

Reject empty settings and negative caps. Redact `api_key` from the dataclass repr. Make `can_afford` evaluate `spent + projected <= cap`.

- [ ] **Step 4: Write a failing terminal-checkpoint test**

```python
def test_terminal_keys_skips_malformed_and_nonterminal_jsonl_lines(self):
    path.write_text('{"doc_idx":"1","section_name":"hpi","task":"symptoms","model":"gpt-5.6-sol","validation_status":"valid"}\nnot json\n{"validation_status":"running"}\n')
    self.assertEqual(terminal_keys(path), {("1", "hpi", "symptoms", "gpt-5.6-sol")})
```

- [ ] **Step 5: Verify RED, then implement JSONL helpers**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected before implementation: FAIL because `terminal_keys` is missing.

Implement `TERMINAL_STATUSES = {"valid", "valid_after_retry", "validation_failed", "api_failed", "spend_cap_reached"}`. `append_checkpoint` must create parents and append one JSON object plus one newline. `terminal_keys` must ignore malformed lines.

- [ ] **Step 6: Verify GREEN and commit**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: PASS.

```bash
git add coral/azure_evaluation.py tests/test_azure_evaluation.py
git commit -m "feat: add Azure evaluation primitives"
```

### Task 2: Add safe task schemas, validation, and legacy conversion

**Files:**
- Modify: `coral/azure_evaluation.py`
- Modify: `tests/test_azure_evaluation.py`

**Interfaces:**
- Produces `build_request(row: Mapping[str, str]) -> tuple[str, dict[str, object]]`.
- Produces `validate_response(task: str, section_text: str, raw: str) -> list[dict[str, object]]`.
- Produces `to_legacy_output(task: str, records: list[dict[str, object]]) -> str`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_validate_response_requires_verbatim_evidence(self):
    raw = json.dumps({"task": "symptoms", "records": [{
        "symptom": "low appetite", "datetimes": ["unknown"],
        "evidence_quotes": ["appetite is low"],
    }]})
    self.assertEqual(validate_response("symptoms", "The appetite is low.", raw)[0]["symptom"], "low appetite")

def test_validate_response_rejects_nonverbatim_evidence(self):
    with self.assertRaisesRegex(ValueError, "evidence"):
        validate_response("symptoms", "No appetite statement.", VALID_SYMPTOM_JSON)
```

- [ ] **Step 2: Verify RED**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: FAIL because `validate_response` is missing.

- [ ] **Step 3: Implement task schema and validation**

Create `TASK_FIELDS` for every task in `coral.task_to_default_tuple_dict`. `build_request` returns a system prompt and strict JSON schema containing the selected `task`, records, relevant tuple fields, and `evidence_quotes`. Parse with `json.loads`, never `eval`; reject wrong tasks, missing fields, malformed JSON, and evidence not literally present after CRLF normalization.

- [ ] **Step 4: Write failing legacy conversion test**

```python
def test_to_legacy_output_serializes_symptom_record(self):
    self.assertEqual(
        to_legacy_output("symptoms", [{"symptom": "low appetite", "datetimes": ["unknown"], "evidence_quotes": ["appetite is low"]}]),
        "SymptomEnt(Symptom='low appetite', Datetime={'unknown'})",
    )
```

- [ ] **Step 5: Verify RED, then implement conversion**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected before implementation: FAIL because `to_legacy_output` is missing.

Convert every `TASK_FIELDS` record to a deterministic, namedtuple-compatible string accepted by existing `parse_output`. Serialize scalars with `repr`, sort set-like values, omit `evidence_quotes`, and use the existing unknown tuple only for an empty record list.

- [ ] **Step 6: Verify GREEN and commit**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: PASS.

```bash
git add coral/azure_evaluation.py tests/test_azure_evaluation.py
git commit -m "feat: validate Azure structured CORAL responses"
```

### Task 3: Add a fake-client-tested Azure runner

**Files:**
- Modify: `coral/azure_evaluation.py`
- Modify: `tests/test_azure_evaluation.py`

**Interfaces:**
- Produces `run_evaluation(rows: pd.DataFrame, client: ResponseClient, settings: AzureSettings, checkpoint_path: Path, spend_cap: float, max_output_tokens: int, reasoning_effort: str) -> pd.DataFrame`.
- `ResponseClient` has `responses.create(**kwargs) -> object`; tests use a fake and the notebook supplies `AzureOpenAI(...).responses`.

- [ ] **Step 1: Write failing runner tests**

```python
def test_runner_checkpoints_valid_usage(self):
    result = run_evaluation(ONE_ROW, FakeClient(VALID_SYMPTOM_JSON, 100, 20), SETTINGS, CHECKPOINT, 1.0, 512, "low")
    self.assertEqual(result.iloc[0].validation_status, "valid")
    self.assertEqual(result.iloc[0].input_tokens, 100)

def test_runner_retries_once_then_marks_valid_after_retry(self):
    client = FakeClient.sequence(["not-json", VALID_SYMPTOM_JSON])
    result = run_evaluation(ONE_ROW, client, SETTINGS, CHECKPOINT, 1.0, 512, "low")
    self.assertEqual(result.iloc[0].validation_status, "valid_after_retry")
    self.assertEqual(client.calls, 2)

def test_runner_respects_cap_before_making_request(self):
    result = run_evaluation(ONE_ROW, FakeClient.fail_if_called(), SETTINGS, CHECKPOINT, 0.0, 512, "low")
    self.assertEqual(result.iloc[0].validation_status, "spend_cap_reached")
```

- [ ] **Step 2: Verify RED**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: FAIL because `run_evaluation` is missing.

- [ ] **Step 3: Implement one-at-a-time Responses API execution**

Call `client.responses.create` with `model=settings.deployment`, task-specific input, `reasoning={"effort": reasoning_effort}`, `max_output_tokens=max_output_tokens`, and structured JSON output. Estimate the upcoming cost from section length plus the maximum output before each call; append `spend_cap_reached` and stop before a cap breach. Capture output text, token usage, elapsed seconds, raw attempts, parsed records, and final status.

Retry only a validation failure, exactly once, with a corrective prompt containing the validation message. On client exceptions, checkpoint `api_failed` with exception class and message only. Resume using `(doc_idx, section_name, task, deployment)` and return rows marked `skipped_on_resume` without calling the fake client.

- [ ] **Step 4: Verify GREEN and commit**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: PASS.

```bash
git add coral/azure_evaluation.py tests/test_azure_evaluation.py
git commit -m "feat: run resumable Azure CORAL evaluations"
```

### Task 4: Export scorer inputs and actual-prompt-only summaries

**Files:**
- Modify: `coral/azure_evaluation.py`
- Modify: `tests/test_azure_evaluation.py`

**Interfaces:**
- Produces `write_legacy_csv(records: Iterable[dict[str, object]], path: Path, model: str) -> pd.DataFrame`.
- Produces `write_observed_summary(instance_scores: pd.DataFrame, input_rows: pd.DataFrame, path: Path) -> pd.DataFrame`.

- [ ] **Step 1: Write failing export tests**

```python
def test_write_legacy_csv_excludes_failed_records(self):
    frame = write_legacy_csv([VALID_RECORD, {**VALID_RECORD, "validation_status": "validation_failed"}], LEGACY_PATH, "gpt-5.6-sol")
    self.assertEqual(len(frame), 1)
    self.assertEqual(frame.iloc[0].conversion_status, "validated")

def test_observed_summary_excludes_synthetic_cartesian_rows(self):
    summary = write_observed_summary(SCORES_WITH_ONE_REAL_AND_ONE_SYNTHETIC_ROW, ONE_ROW, SUMMARY_PATH)
    self.assertEqual(summary.iloc[0].n_examples, 1)
```

- [ ] **Step 2: Verify RED, then implement exports**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected before implementation: FAIL because the export functions are missing.

Write only `valid` and `valid_after_retry` records to the legacy columns `doc_idx`, `section_name`, `section_text`, `task`, `model`, `output`, and `conversion_status`. Inner-join instance scores to real input keys (`doc_idx`, `section_name`, `task`); group by `task` and `subrelation`; write counts and all five mean metrics.

- [ ] **Step 3: Verify GREEN and commit**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: PASS.

```bash
git add coral/azure_evaluation.py tests/test_azure_evaluation.py
git commit -m "feat: export Azure evaluation scoring artifacts"
```

### Task 5: Build the clean marimo notebook and verify without paid calls

**Files:**
- Create: `notebooks/evaluate_gpt56_sol_azure.py`
- Modify: `tests/test_azure_evaluation.py`

**Interfaces:**
- Consumes all public helpers from `coral.azure_evaluation`.
- Produces an executable marimo app with a smoke-test default, a full-run confirmation, visible spending data, and artifact controls.

- [ ] **Step 1: Write a failing notebook structure test**

```python
def test_notebook_declares_azure_dependencies_and_full_run_guard(self):
    notebook = Path("notebooks/evaluate_gpt56_sol_azure.py").read_text()
    self.assertIn('"marimo"', notebook)
    self.assertIn('"openai"', notebook)
    self.assertIn("AZURE_OPENAI_API_KEY", notebook)
    self.assertIn("Run full 515-input evaluation", notebook)
    self.assertNotIn("OPENAI_API_KEY", notebook.replace("AZURE_OPENAI_API_KEY", ""))
```

- [ ] **Step 2: Verify RED**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v`

Expected: FAIL because the notebook is absent.

- [ ] **Step 3: Implement the marimo notebook**

Use PEP 723 dependencies `marimo`, `openai`, `pandas`, and `pydantic`. Add cells for: overview and pricing; safe Azure configuration display; data preview and cost projection; widgets for limit, token ceiling, reasoning effort, and cap; an eight-input smoke button; the `RUN 515` full-run confirmation and button; results; legacy export; and score-summary display.

Construct `AzureOpenAI(api_key=settings.api_key, azure_endpoint=settings.endpoint, api_version=settings.api_version)` only inside the explicit run cell. In script mode, do not construct a client or send a request unless `CORAL_AZURE_RUN=1`. State plainly that the repository-wide legacy macro is flawed and point to the observed-only summary.

- [ ] **Step 4: Verify notebook structure and marimo behavior**

Run:

```bash
UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest tests.test_azure_evaluation -v
UV_CACHE_DIR=/tmp/coral-uv-cache uvx marimo check notebooks/evaluate_gpt56_sol_azure.py
UV_CACHE_DIR=/tmp/coral-uv-cache uv run notebooks/evaluate_gpt56_sol_azure.py
```

Expected: all tests pass, `marimo check` has no errors, and script mode exits without Azure activity when `CORAL_AZURE_RUN` is unset.

- [ ] **Step 5: Run full regression tests and commit**

Run: `UV_CACHE_DIR=/tmp/coral-uv-cache uv run python -m unittest discover -s tests -v`

Expected: PASS; existing non-failing Streamlit bare-mode warnings may remain.

```bash
git add notebooks/evaluate_gpt56_sol_azure.py coral/azure_evaluation.py tests/test_azure_evaluation.py
git commit -m "feat: add Azure GPT-5.6 Sol evaluation notebook"
```

## Final Verification

- [ ] Run `git diff --check` and `git status --short`.
- [ ] Confirm no notebook or artifact source contains a literal credential or reads `OPENAI_API_KEY`.
- [ ] Confirm the smoke-test default is eight and full execution requires `RUN 515`.
- [ ] Confirm artifact names begin with `gpt56_sol_azure`.
- [ ] Confirm script mode makes no paid request without `CORAL_AZURE_RUN=1`.
