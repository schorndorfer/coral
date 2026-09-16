# Streamlit Annotation Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, read-only Streamlit application for browsing CORAL notes, highlighted expert annotations, attributes, and relationships.

**Architecture:** A viewer-specific BRAT loader produces immutable records without changing the benchmarking parser. A pure HTML renderer safely highlights overlapping and discontinuous spans. A thin Streamlit entry point manages local directory loading, filters, selections, and the approved three-panel layout.

**Tech Stack:** Python 3.12, standard-library dataclasses and HTML escaping, Streamlit, pandas, unittest

**Spec:** `docs/superpowers/specs/2026-09-16-streamlit-annotation-viewer-design.md`

## Global Constraints

- The app is read-only and never writes `.txt` or `.ann` source files.
- Clinical text remains local and must not appear in logs, exceptions, or test output.
- The browser accepts a server-local directory path; it does not upload files.
- Expert clinical annotations are visible by default; automated preannotations and structural section annotations are opt-in.
- Published summary counts include expert structural annotations and exclude automated `PROBLEM`, `TREATMENT`, `TEST`, and `SectionAnnotate` entities.
- Unknown relation types are excluded from schema-valid counts and surfaced as warnings.
- Do not modify the existing benchmarking parser behavior.
- Do not stage unrelated pre-existing working-tree changes when committing a task.

---

### Task 1: Parse One BRAT Document Into Viewer Records

**Files:**
- Create: `coral/viewer/__init__.py`
- Create: `coral/viewer/data.py`
- Create: `tests/test_viewer_data.py`

**Interfaces:**
- Produces: `Span(start: int, end: int)`
- Produces: `ViewerEntity(id: str, type: str, spans: tuple[Span, ...], text: str, auxiliary: bool)`
- Produces: `ViewerAttribute(id: str, type: str, entity_id: str, value: str | None)`
- Produces: `ViewerRelation(id: str, type: str, source_id: str, target_id: str, schema_valid: bool)`
- Produces: `ViewerDocument(key: str, document_id: str, cohort: str, text: str, entities: tuple[ViewerEntity, ...], attributes: tuple[ViewerAttribute, ...], relationships: tuple[ViewerRelation, ...], warnings: tuple[str, ...])`
- Produces: `load_document(text_path: Path, ann_path: Path, root: Path, known_relation_types: frozenset[str]) -> ViewerDocument`

- [ ] **Step 1: Write failing parser tests**

Create synthetic note and annotation files in `tempfile.TemporaryDirectory`. Tests must use invented content only:

```python
NOTE = "Treatment started Monday. Mass remained stable."
ANN = "\n".join([
    "T1\tMedicationName 0 9\tTreatment",
    "T2\tDatetime 18 24\tMonday",
    "T3\tClinicalCondition 26 30;40 46\tMass stable",
    "A1\tNegationModalityVal T1 affirmed",
    "R1\tBeginsOnOrAt Arg1:T1 Arg2:T2",
    "*\tCoreference T1 T3",
])
```

Assert that `load_document`:

- Preserves `T3` as two spans: `(26, 30)` and `(40, 46)`.
- Parses `A1`, `R1`, and the `*` relation.
- Uses the note's relative path as a stable key and its parent directory as cohort.
- Marks `PROBLEM`, `TREATMENT`, `TEST`, `SectionAnnotate`, `SectionSkip`, `hpi_start`, `hpi_end`, `ap_start`, and `ap_end` as auxiliary.

Add separate tests asserting malformed offsets, out-of-range spans, missing entity targets, unsupported record prefixes, and invalid UTF-8 create warnings containing the relative filename and line number but not annotation text.

- [ ] **Step 2: Run the parser tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_viewer_data -v
```

Expected: import failure because `coral.viewer.data` does not exist.

- [ ] **Step 3: Implement immutable records and the single-document parser**

In `coral/viewer/data.py`, define frozen dataclasses and constants:

```python
AUTOMATED_ENTITY_TYPES = frozenset({"PROBLEM", "TREATMENT", "TEST", "SectionAnnotate"})
STRUCTURAL_ENTITY_TYPES = frozenset({"SectionSkip", "hpi_start", "hpi_end", "ap_start", "ap_end"})
AUXILIARY_ENTITY_TYPES = AUTOMATED_ENTITY_TYPES | STRUCTURAL_ENTITY_TYPES
```

Parse each annotation line independently. For entities, split the descriptor once into type and offset expression, then split discontinuous offsets on `;`. Validate `0 <= start < end <= len(text)`. For `R` records, extract the IDs following `Arg1:` and `Arg2:`. For `*` records, assign a stable generated ID such as `*:<line_number>` and treat the second and third tokens as source and target. Do not include note or annotation text in warning messages.

After parsing, validate attribute and relationship targets against the entity-ID set. Drop records with missing targets and append sanitized warnings.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.test_viewer_data -v
```

Expected: all parser tests pass without warnings or tracebacks.

- [ ] **Step 5: Commit the parser task**

```bash
git add coral/viewer/__init__.py coral/viewer/data.py tests/test_viewer_data.py
git commit -m "feat: parse BRAT annotations for viewer"
```

---

### Task 2: Discover and Summarize a Dataset

**Files:**
- Modify: `coral/viewer/data.py`
- Modify: `tests/test_viewer_data.py`

**Interfaces:**
- Consumes: Task 1 viewer records and `load_document(...)`
- Produces: `DatasetLoadResult(documents: tuple[ViewerDocument, ...], warnings: tuple[str, ...], counts: DatasetCounts)`
- Produces: `DatasetCounts(documents: int, expert_entities: int, attributes: int, schema_valid_relationships: int)`
- Produces: `parse_relation_types(config_path: Path) -> frozenset[str]`
- Produces: `load_dataset(root: str | Path) -> DatasetLoadResult`
- Produces: `visible_entities(document: ViewerDocument, show_auxiliary: bool, entity_types: frozenset[str] | None = None) -> tuple[ViewerEntity, ...]`

- [ ] **Step 1: Write failing dataset discovery and count tests**

Build a temporary tree containing:

```text
annotated/
  annotation.conf
  pdac/1.txt
  pdac/1.ann
  breastca/1.txt
  breastca/1.ann
  breastca/orphan.txt
```

The config must define two valid relation types in `[relations]`, followed by `[events]`. The annotation fixtures must include expert, automated, and structural entities; valid and unknown relationships; and duplicate document stems in different cohorts.

Assert that:

- Documents are keyed as `pdac/1` and `breastca/1` and naturally sorted.
- `orphan.txt` is skipped with a sanitized warning.
- Only configured relationship names have `schema_valid=True` and contribute to the published count.
- `DatasetCounts.expert_entities` excludes only `AUTOMATED_ENTITY_TYPES` and includes structural entities.
- Default `visible_entities` excludes all `AUXILIARY_ENTITY_TYPES`.
- `show_auxiliary=True` returns every entity.
- An entity-type filter intersects with the visibility filter.
- A missing path, file path, and empty directory return actionable results without raising.

- [ ] **Step 2: Run dataset tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_viewer_data -v
```

Expected: failures for missing `DatasetLoadResult`, `parse_relation_types`, `load_dataset`, and `visible_entities`.

- [ ] **Step 3: Implement schema parsing, discovery, filtering, and counts**

Parse relation names only between `[relations]` and `[events]`. Ignore blank lines, comments, macro definitions beginning with `<`, and indented continuation lines. Discover notes with `Path.rglob("*.txt")`; pair with `with_suffix(".ann")`.

Compute counts with IDs qualified by document:

```python
expert_ids_by_document = {
    document.key: {
        entity.id
        for entity in document.entities
        if entity.type not in AUTOMATED_ENTITY_TYPES
    }
    for document in documents
}
expert_entity_count = sum(len(ids) for ids in expert_ids_by_document.values())
attribute_count = sum(
    attribute.entity_id in expert_ids_by_document[document.key]
    for document in documents
    for attribute in document.attributes
)
relationship_count = sum(
    relation.schema_valid
    and relation.source_id in expert_ids_by_document[document.key]
    and relation.target_id in expert_ids_by_document[document.key]
    for document in documents
    for relation in document.relationships
)
```

- [ ] **Step 4: Run dataset tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.test_viewer_data -v
```

Expected: all parser and dataset tests pass.

- [ ] **Step 5: Commit the dataset task**

```bash
git add coral/viewer/data.py tests/test_viewer_data.py
git commit -m "feat: discover and summarize BRAT datasets"
```

---

### Task 3: Render Safe Highlighted Notes

**Files:**
- Create: `coral/viewer/rendering.py`
- Create: `tests/test_viewer_rendering.py`

**Interfaces:**
- Consumes: `ViewerEntity` and `Span` from `coral.viewer.data`
- Produces: `entity_color(entity_type: str) -> str`
- Produces: `render_highlighted_text(text: str, entities: Sequence[ViewerEntity], selected_entity_id: str | None = None) -> str`
- Produces: `format_offsets(entity: ViewerEntity) -> str`

- [ ] **Step 1: Write failing escaping and basic highlight tests**

Use a literal input containing `<script>`, `&`, and one annotated word. Assert that the result contains `&lt;script&gt;`, never contains the raw `<script>`, preserves whitespace through CSS `white-space: pre-wrap`, and wraps the annotated word with an entity type and ID label.

Assert that `entity_color("MedicationName")` returns the same CSS color on repeated calls and a different color from `entity_color("Datetime")`.

- [ ] **Step 2: Run renderer tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_viewer_rendering -v
```

Expected: import failure because `coral.viewer.rendering` does not exist.

- [ ] **Step 3: Implement escaping, colors, and non-overlapping highlights**

Use `html.escape(..., quote=True)` for every text fragment and visible label. Derive colors from a fixed accessible palette using a stable SHA-256 digest of the entity type rather than Python's randomized `hash()`. Emit one outer note container with `white-space: pre-wrap; overflow-wrap: anywhere`.

- [ ] **Step 4: Run renderer tests and verify GREEN**

Run the Task 3 test command and confirm the basic tests pass.

- [ ] **Step 5: Write failing overlap, discontinuous, and selection tests**

Add tests with hand-checked expected fragments for:

- Two entities sharing part of a span.
- One entity containing two discontinuous spans.
- Adjacent spans with no duplicated or dropped characters.
- A selected entity receiving `data-selected="true"` on each of its fragments.
- Zero entities returning the entire escaped note unchanged inside the container.

In every test, strip HTML tags with a test-only `HTMLParser` and assert that the reconstructed text equals the original note exactly.

- [ ] **Step 6: Run advanced renderer tests and verify RED**

Run the Task 3 test command. Expected: overlap/discontinuous assertions fail while the basic tests remain green.

- [ ] **Step 7: Implement boundary-sweep rendering**

Collect boundaries `{0, len(text)} ∪ {span.start, span.end}`. For each adjacent interval, find active entities that fully cover it. Escape the interval once, then render:

- Plain escaped text when no entity is active.
- A solid background for one entity.
- A CSS striped `linear-gradient` for multiple active entities.
- A selected outline when any active entity matches `selected_entity_id`.

Include active IDs and types in escaped `data-entity-ids`, `data-entity-types`, and `title` attributes.

- [ ] **Step 8: Run all renderer and parser tests**

```bash
.venv/bin/python -m unittest tests.test_viewer_data tests.test_viewer_rendering -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit the renderer task**

```bash
git add coral/viewer/rendering.py tests/test_viewer_rendering.py
git commit -m "feat: render safe annotation highlights"
```

---

### Task 4: Build the Streamlit Explorer

**Files:**
- Create: `streamlit_app.py`
- Create: `tests/test_streamlit_app.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `load_dataset`, `visible_entities`, `render_highlighted_text`, and `format_offsets`
- Produces: `annotation_label(entity: ViewerEntity) -> str`
- Produces: `related_rows(document: ViewerDocument, entity_id: str, direction: str) -> list[dict[str, str]]`
- Produces: Streamlit entry point runnable with `streamlit run streamlit_app.py`

- [ ] **Step 1: Install Streamlit in the existing local environment**

```bash
UV_CACHE_DIR=/private/tmp/coral-uv-cache uv pip install --python .venv/bin/python streamlit
```

This modifies only the ignored `.venv`; add `streamlit` to the README dependency command in Step 7.

- [ ] **Step 2: Write failing presenter-helper tests**

In `tests/test_streamlit_app.py`, import `annotation_label` and `related_rows`. Use synthetic viewer records to assert:

- Labels distinguish duplicate text using entity ID and type.
- Incoming rows identify the source entity and outgoing rows identify the target entity.
- Missing related entities are omitted safely.
- No helper returns or prints full note text.

- [ ] **Step 3: Run helper tests and verify RED**

```bash
.venv/bin/python -m unittest tests.test_streamlit_app -v
```

Expected: import failure because `streamlit_app.py` does not exist.

- [ ] **Step 4: Implement presenter helpers and minimal app shell**

Create `streamlit_app.py` with `st.set_page_config(page_title="CORAL Annotation Viewer", layout="wide")`. Keep pure helper functions above `main()` and call `main()` only under `if __name__ == "__main__":` so tests can import helpers without executing the app.

Implement the sidebar path input using `os.environ.get("CORAL_DATA_DIR", "")`. On **Load dataset**, call `load_dataset`, store its result and resolved path in `st.session_state`, and reset document/entity selections.

- [ ] **Step 5: Run helper tests and verify GREEN**

Run the Task 4 test command and confirm all helper tests pass.

- [ ] **Step 6: Write a failing Streamlit smoke test**

Build a synthetic two-document BRAT directory in a temporary directory. Set `CORAL_DATA_DIR` with `unittest.mock.patch.dict`, run `AppTest.from_file("streamlit_app.py")`, click **Load dataset**, and assert:

- No `Exception` elements appear.
- The page shows two documents.
- Selecting the second document changes its entity selector.
- Selecting an entity displays its type, offsets, one attribute, and one relationship.
- Toggling auxiliary annotations makes a synthetic `PROBLEM` entity selectable.

- [ ] **Step 7: Implement the three-panel UI and documentation**

Use `st.columns([2, 5, 3])`:

- Left: cohort filter, document search, and document `selectbox`.
- Center: legend plus `st.markdown(rendered_note, unsafe_allow_html=True)`.
- Right: entity-type filter, auxiliary toggle, entity `selectbox`, compact `st.dataframe`, attributes, incoming relationships, outgoing relationships, and warnings.

Use the selected entity ID as the `selectbox` value and pass it to the renderer. Never log document text. Update the README installation command to include `streamlit` and add:

```bash
CORAL_DATA_DIR=/absolute/path/to/coral/annotated \
streamlit run streamlit_app.py
```

- [ ] **Step 8: Run the Streamlit smoke test and full unit suite**

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected: all hardware, parser, renderer, presenter, and Streamlit tests pass.

- [ ] **Step 9: Commit the Streamlit app task**

```bash
git add streamlit_app.py tests/test_streamlit_app.py README.md
git commit -m "feat: add Streamlit annotation viewer"
```

---

### Task 5: Validate Against the CORAL Release

**Files:**
- Modify only if validation exposes a defect: files and tests owned by Tasks 1–4

**Interfaces:**
- Consumes: complete viewer application and the local CORAL release
- Produces: verified acceptance evidence; no new runtime interface

- [ ] **Step 1: Run the complete automated suite**

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q coral streamlit_app.py tests
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Validate the real release through the loader**

Run a short read-only command against:

```text
/Users/wkt406/Downloads/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated
```

Assert and print counts only—not note text:

```python
result = load_dataset(DATASET_PATH)
assert result.counts.documents == 40
assert result.counts.expert_entities == 9028
assert result.counts.attributes == 9986
assert result.counts.schema_valid_relationships == 5312
```

- [ ] **Step 3: Start Streamlit and perform a local health check**

```bash
CORAL_DATA_DIR=/Users/wkt406/Downloads/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated \
.venv/bin/streamlit run streamlit_app.py --server.headless true --server.port 8501
```

From a second shell, verify readiness:

```bash
curl --fail http://localhost:8501/_stcore/health
```

Wait for the local health endpoint to return success, then verify the process logs contain no clinical note text or uncaught exceptions.

- [ ] **Step 4: Review the final diff and working tree**

```bash
git diff --check
git status --short
git log --oneline --decorate -8
```

Confirm only intended viewer files, documentation, tests, and previously known Apple Silicon changes are present. Confirm `data/`, `models/`, `output/`, `.venv/`, and `.superpowers/` remain ignored.

- [ ] **Step 5: Commit any validation-only corrections**

If Step 2 or Step 3 required a tested correction, commit only the corrected viewer files and their regression tests:

```bash
git add coral/viewer/data.py coral/viewer/rendering.py streamlit_app.py \
  tests/test_viewer_data.py tests/test_viewer_rendering.py tests/test_streamlit_app.py
git commit -m "fix: validate annotation viewer against CORAL release"
```

If no correction was required, do not create an empty commit.
