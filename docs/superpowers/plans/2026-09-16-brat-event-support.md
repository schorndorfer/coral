# BRAT Event and Multiline Annotation Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse multiline BRAT text annotations and events, expose event-linked relationships in the read-only viewer, and reduce the CORAL release warnings from 148 to the single genuine schema warning without changing published totals.

**Architecture:** `coral/viewer/data.py` will first assemble physical `.ann` lines into logical records, then parse immutable events alongside entities, attributes, and relationships. Relationships retain original endpoint IDs but use a public resolver to map event endpoints to trigger entities for presentation. `streamlit_app.py` will show event metrics, attached-event metadata, event-aware relationship rows, and grouped warning diagnostics.

**Tech Stack:** Python 3.12, frozen dataclasses, `pathlib`, `unittest`, Streamlit, `st.testing.v1.AppTest`.

**Spec:** `docs/superpowers/specs/2026-09-16-brat-event-support-design.md`

## Global Constraints

- Loading remains local and read-only; never modify source `.txt`, `.ann`, or `annotation.conf` files.
- Never include note text or annotation reference text in warnings, exceptions, logs, or test failure messages.
- Published totals remain exactly `40 / 9028 / 9986 / 5312`.
- Event and event-linked relationship counts are separate from published totals and both equal 2 for the CORAL release.
- The only release warning after this work is the unknown `TreatmentTypeRel` schema entry.
- Events are highlighted through their trigger entities, never as independent text spans.
- Preserve unrelated Apple Silicon working-tree changes; stage only files owned by the active task.

---

### Task 1: Assemble Multiline BRAT Records

**Files:**
- Modify: `coral/viewer/data.py`
- Test: `tests/test_viewer_data.py`

**Interfaces:**
- Consumes: physical lines returned by `Path.read_bytes().splitlines()`.
- Produces: `_read_logical_records(path: Path, filename: str, warnings: list[str]) -> tuple[tuple[int, str], ...]`, where each tuple contains the first physical line number and the decoded logical record.
- Preserves: `load_document(...) -> ViewerDocument` and sanitized warning strings.

- [ ] **Step 1: Write failing multiline and orphan-continuation tests**

Add tests equivalent to:

```python
def test_joins_multiline_text_bound_reference_text(self):
    document = self.load(
        note="Alpha\n  Beta",
        ann="T1\tClinicalCondition 0 5;8 12\tAlpha\n  Beta",
    )
    self.assertEqual(document.entities[0].text, "Alpha\n  Beta")
    self.assertEqual(document.warnings, ())

def test_orphan_continuation_warns_without_exposing_text(self):
    document = self.load(ann="  synthetic private continuation")
    self.assertEqual(document.entities, ())
    self.assertEqual(
        document.warnings,
        ("cohort-a/note.ann: line 1: orphan annotation continuation",),
    )
    self.assertNotIn("synthetic private continuation", document.warnings[0])
```

Also add a three-physical-line test so consecutive continuations remain attached to the same `T` record.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest \
  tests.test_viewer_data.ViewerDataTests.test_joins_multiline_text_bound_reference_text \
  tests.test_viewer_data.ViewerDataTests.test_orphan_continuation_warns_without_exposing_text -v
```

Expected: the multiline case reports unsupported record types and the orphan case reports the old unsupported-record warning.

- [ ] **Step 3: Implement logical-record assembly**

Replace the physical-line loop in `load_document` with `_read_logical_records`. Decode each physical line separately so invalid UTF-8 retains its physical line number. Skip blank lines. Append whitespace-prefixed lines only when the previous logical record ID begins with `T`; otherwise emit `orphan annotation continuation` without including line content.

The implementation shape is:

```python
def _read_logical_records(
    path: Path,
    filename: str,
    warnings: list[str],
) -> tuple[tuple[int, str], ...]:
    records: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            _warn(warnings, filename, line_number, "invalid UTF-8")
            continue
        if not line.strip():
            continue
        if line[:1].isspace():
            if records and records[-1][1].split("\t", 1)[0].startswith("T"):
                first_line, record = records[-1]
                records[-1] = (first_line, f"{record}\n{line}")
            else:
                _warn(warnings, filename, line_number, "orphan annotation continuation")
            continue
        records.append((line_number, line))
    return tuple(records)
```

- [ ] **Step 4: Verify GREEN and existing parser behavior**

Run:

```bash
.venv/bin/python -m unittest tests.test_viewer_data.ViewerDataTests -v
```

Expected: all viewer parser tests pass, including invalid UTF-8 and sanitized-warning tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add coral/viewer/data.py tests/test_viewer_data.py
git commit -m "fix: reconstruct multiline BRAT records"
```

### Task 2: Parse Events and Resolve Event Endpoints

**Files:**
- Modify: `coral/viewer/data.py`
- Test: `tests/test_viewer_data.py`

**Interfaces:**
- Produces: `ViewerEventArgument(role: str, target_id: str)`.
- Produces: `ViewerEvent(id: str, type: str, trigger_id: str, arguments: tuple[ViewerEventArgument, ...])`.
- Extends: `ViewerDocument.events: tuple[ViewerEvent, ...] = ()` after `warnings` to preserve existing positional constructors.
- Extends: `DatasetCounts.events: int = 0` and `DatasetCounts.event_linked_relationships: int = 0` after existing count fields.
- Produces: `resolve_entity(document: ViewerDocument, annotation_id: str) -> ViewerEntity | None`.

- [ ] **Step 1: Write failing event parser tests**

Add synthetic tests equivalent to:

```python
def test_parses_unary_and_argument_bearing_events(self):
    document = self.load(
        note="Dose drug reason",
        ann="\n".join([
            "T1\tTreatmentDosage 0 4\tDose",
            "T2\tMedicationName 5 9\tdrug",
            "T3\tClinicalCondition 10 16\treason",
            "E1\tTreatmentDosage:T1",
            "E2\tMedication:T2 Reason:T3",
        ]),
    )
    self.assertEqual(
        [(event.id, event.type, event.trigger_id) for event in document.events],
        [("E1", "TreatmentDosage", "T1"), ("E2", "Medication", "T2")],
    )
    self.assertEqual(
        [(argument.role, argument.target_id) for argument in document.events[1].arguments],
        [("Reason", "T3")],
    )
    self.assertEqual(document.warnings, ())

def test_drops_malformed_and_missing_target_events_with_sanitized_warnings(self):
    document = self.load(
        ann="\n".join([
            "T1\tMedicationName 0 9\tTreatment",
            "E1\tMalformed",
            "E2\tMedication:T404",
        ])
    )
    self.assertEqual(document.events, ())
    self.assertEqual(len(document.warnings), 2)
    self.assertTrue(all("Treatment" not in warning for warning in document.warnings))
```

- [ ] **Step 2: Run the event tests and verify RED**

Run the two new test methods with `python -m unittest ... -v`.

Expected: `ViewerDocument` has no `events` field and `E` records are unsupported.

- [ ] **Step 3: Add immutable event models and parsing**

Implement `_parse_event(line, filename, line_number, warnings) -> ViewerEvent | None`. Require an `E` ID, a first `EventType:TriggerID` token, valid `BRAT_ROLE_RE` names, and nonempty target IDs. Additional `Role:TargetID` tokens become `ViewerEventArgument` values.

Collect events with physical line numbers during parsing. After entity parsing, retain only events whose trigger and every argument target exist in `entity_ids`; emit `event references a missing entity` otherwise.

- [ ] **Step 4: Write failing event-linked relationship and counting tests**

Add a dataset fixture whose `annotation.conf` contains `TreatmentDesc Desc:TreatmentDosage, Therapy:MedicationName` and whose annotation contains:

```text
T1 TreatmentDosage 0 4 Dose
T2 MedicationName 5 9 drug
E1 TreatmentDosage:T1
R1 TreatmentDesc Desc:E1 Therapy:T2
```

Use tab separators in the fixture. Assert that:

```python
self.assertEqual(document.relationships[0].source_id, "E1")
self.assertEqual(resolve_entity(document, "E1"), document.entities[0])
self.assertEqual(result.counts.schema_valid_relationships, 0)
self.assertEqual(result.counts.events, 1)
self.assertEqual(result.counts.event_linked_relationships, 1)
```

- [ ] **Step 5: Run the relationship/count tests and verify RED**

Expected: the relationship is dropped because `E1` is not a recognized target and the new count fields do not exist.

- [ ] **Step 6: Resolve event targets and add separate metrics**

Validate relationships against `entity_ids | event_ids`. Keep their original IDs. Implement `resolve_entity` by returning a direct entity match or resolving an event ID to its trigger entity.

Compute counts as follows:

```python
event_ids = {event.id for event in document.events}

schema_valid_relationships = sum(
    relation.schema_valid
    and relation.source_id in expert_ids
    and relation.target_id in expert_ids
    for relation in document.relationships
)

event_linked_relationships = sum(
    relation.schema_valid
    and (relation.source_id in event_ids or relation.target_id in event_ids)
    for relation in document.relationships
)
```

Aggregate `events` and `event_linked_relationships` across documents. Do not resolve event IDs before calculating the published relationship count.

- [ ] **Step 7: Verify Task 2 and commit**

Run:

```bash
.venv/bin/python -m unittest tests.test_viewer_data -v
git diff --check
```

Expected: all data tests pass with no clinical text in output.

```bash
git add coral/viewer/data.py tests/test_viewer_data.py
git commit -m "feat: parse BRAT events in annotation viewer"
```

### Task 3: Present Events and Group Warnings

**Files:**
- Modify: `streamlit_app.py`
- Test: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `ViewerDocument.events`, `DatasetCounts.events`, `DatasetCounts.event_linked_relationships`, and `resolve_entity(...)` from Task 2.
- Produces: `event_rows(document: ViewerDocument, entity_id: str) -> list[dict[str, str]]`.
- Produces: `warning_summary(warnings: tuple[str, ...]) -> list[dict[str, str | int]]`.
- Extends: `related_rows(...)` with an `event` column while retaining relationship, entity, text, and offsets.

- [ ] **Step 1: Write failing presenter tests**

Construct a document with `E1` triggered by `T1` and a schema-valid relationship from `E1` to `T2`. Assert:

```python
self.assertEqual(
    event_rows(document, "T1"),
    [{"event": "TreatmentDosage (E1)", "trigger": "TreatmentDosage (T1)", "arguments": "None"}],
)
self.assertEqual(
    related_rows(document, "T1", "outgoing")[0]["event"],
    "TreatmentDosage (E1)",
)
self.assertEqual(
    related_rows(document, "T2", "incoming")[0]["entity"],
    "TreatmentDosage (T1)",
)
```

Add a warning-summary test:

```python
self.assertEqual(
    warning_summary((
        "a.ann: line 1: unknown relation type",
        "b.ann: line 2: unknown relation type",
        "c.ann: line 3: malformed event record",
    )),
    [
        {"category": "unknown relation type", "count": 2},
        {"category": "malformed event record", "count": 1},
    ],
)
```

- [ ] **Step 2: Run presenter tests and verify RED**

Run the new `PresenterHelperTests` methods. Expected: missing helpers and event-aware resolution failures.

- [ ] **Step 3: Implement event-aware presenters**

Use `resolve_entity` for both relationship endpoints. Match the selected entity against resolved endpoints, not raw IDs. Add the event label when either raw endpoint is an event. Keep invalid relationships excluded.

`event_rows` lists events whose `trigger_id` equals the selected entity ID. Render argument metadata using IDs and entity types; include argument annotation text only in the UI table, never in diagnostics.

`warning_summary` groups by the sanitized reason after the final `": "`, sorts by descending count then category, and returns no raw annotation content.

- [ ] **Step 4: Write failing AppTest coverage**

Extend the synthetic dataset with one unary event and one event-linked relationship. Update the metric assertion to include:

```python
"Events": "1",
"Event-linked relationships": "1",
```

After selecting the trigger entity, assert the attached-event table and the event-aware relationship row are present. Seed two sanitized warnings and assert the summary table contains category/count rows while the individual warnings appear only inside a `Warnings details` expander.

- [ ] **Step 5: Run AppTest and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_streamlit_app -v
```

Expected: missing metrics, event table, event relationships, and grouped diagnostics.

- [ ] **Step 6: Update the Streamlit UI**

Add `Events` and `Event-linked relationships` metrics in the sidebar. In `_show_entity_inspector`, render an `Events` subsection with `st.dataframe(..., hide_index=True, width="stretch")` when the selected entity triggers events, otherwise use the caption `No events`.

Replace one-warning-per-card output with:

```python
st.dataframe(warning_summary(warnings), hide_index=True, width="stretch")
with st.expander("Warning details"):
    for warning in warnings:
        st.warning(warning)
```

Keep details collapsed by default. Do not add custom CSS or expose note text through warnings.

- [ ] **Step 7: Verify Task 3 and commit**

Run:

```bash
.venv/bin/python -m unittest tests.test_streamlit_app tests.test_viewer_data -v
.venv/bin/python -m compileall -q coral streamlit_app.py tests
git diff --check
```

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: display BRAT events and grouped warnings"
```

### Task 4: Validate Against the CORAL Release

**Files:**
- Modify only if validation exposes a defect: files and tests owned by Tasks 1–3

**Interfaces:**
- Consumes: complete parser, metrics, and Streamlit presentation.
- Produces: acceptance evidence only; no new runtime API.

- [ ] **Step 1: Run the complete automated suite**

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q coral streamlit_app.py tests
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Validate aggregate-only release results**

Run a read-only script against:

```text
/Users/wkt406/Downloads/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated
```

The script must assert, without printing note or annotation text:

```python
assert result.counts.documents == 40
assert result.counts.expert_entities == 9028
assert result.counts.attributes == 9986
assert result.counts.schema_valid_relationships == 5312
assert result.counts.events == 2
assert result.counts.event_linked_relationships == 2
assert len(result.warnings) == 1
assert result.warnings[0].endswith("unknown relation type")
```

- [ ] **Step 3: Run a localhost health check**

Start the server with the repository configuration and dataset environment variable:

```bash
CORAL_DATA_DIR=/Users/wkt406/Downloads/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated \
.venv/bin/streamlit run streamlit_app.py --server.headless true --server.port 8501
```

Verify `curl --fail http://127.0.0.1:8501/_stcore/health` returns `ok`, inspect logs for uncaught exceptions or clinical text, then stop the process and confirm the port is closed.

- [ ] **Step 4: Review the working tree**

```bash
git status --short
git diff --check
git log --oneline --decorate -8
```

Confirm only event-support commits plus the previously known uncommitted Apple Silicon files are present. Do not stage `.DS_Store`, `.agents/`, `.claude/`, or unrelated benchmarking files.

- [ ] **Step 5: Commit validation corrections only when necessary**

If validation exposed a defect, add a failing synthetic regression first, fix it, rerun Steps 1–3, and commit only the affected viewer source and test files:

```bash
git add coral/viewer/data.py streamlit_app.py tests/test_viewer_data.py tests/test_streamlit_app.py
git commit -m "fix: validate BRAT event support against CORAL"
```

If no correction was necessary, do not create an empty commit.
