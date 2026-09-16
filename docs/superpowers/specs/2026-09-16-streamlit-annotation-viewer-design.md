# Streamlit Annotation Viewer Design

## Purpose

Add a local, read-only Streamlit application for browsing CORAL clinical notes and their BRAT annotations. The application must keep note content on the local machine, default to the expert-curated annotations counted in the publication, and make entity attributes and relationships easy to inspect.

## Goals

- Load a local directory recursively containing paired `.txt` and `.ann` files.
- Browse notes by cohort and document ID.
- Render the full note with entity spans highlighted by entity type.
- Select an annotation from a table and emphasize its spans in the note.
- Display the selected entity's metadata, attributes, and incoming and outgoing relationships.
- Hide automated preannotations and section markers by default, with an option to show them.
- Report invalid data without exposing note content in logs or error messages.

## Non-goals

- Editing annotations or writing `.ann` files.
- Uploading data through the browser.
- Calling external APIs or storing notes in a database.
- Running model inference or evaluation from the viewer.
- Reproducing BRAT's arc visualization.

## Architecture

### Entry point

`streamlit_app.py` configures the page, manages Streamlit session state, and composes the three-panel interface. It contains presentation orchestration but no BRAT parsing or HTML-generation logic.

### Viewer data model and loader

`coral/viewer/data.py` provides immutable view-oriented records:

- `ViewerDocument`: document ID, cohort, text, entities, attributes, relationships, and warnings.
- `ViewerEntity`: BRAT ID, type, one or more `(start, end)` spans, annotated text, and whether it is a preannotation.
- `ViewerAttribute`: BRAT ID, type, target entity ID, and optional value.
- `ViewerRelation`: BRAT ID, type, source entity ID, and target entity ID.
- `DatasetLoadResult`: successfully loaded documents plus dataset-level warnings.

The loader recursively discovers `.txt` files, pairs each with the same-stem `.ann` file, and sorts documents by cohort and natural document ID. It parses BRAT data directly into the viewer records so discontinuous spans remain exact and malformed lines can be isolated. It shares the existing preannotation type definitions from `coral.dataprocessor.brat` but does not change the benchmarking parser. Structural section annotations (`SectionSkip`, `hpi_start`, `hpi_end`, `ap_start`, and `ap_end`) are tracked separately so the published entity total and the visible clinical-entity count are both clear.

Supported annotation records are text-bound entities (`T`), attributes/modifiers (`A` or `M`), binary relations (`R`), and equivalence-style relations (`*`). Relation types are checked against `annotation.conf`; unknown types are retained as invalid records for diagnostics but excluded from display and published counts, which excludes the release's single obsolete `TreatmentTypeRel` record. Unsupported records are skipped with a warning. Attributes and relationships whose target entities are missing are retained only as warnings.

### Highlight renderer

`coral/viewer/rendering.py` converts a document and the active filters into HTML suitable for `st.markdown(..., unsafe_allow_html=True)`.

The renderer must HTML-escape the original note and annotation labels before adding markup. It uses a boundary-sweep algorithm over all entity span starts and ends, which preserves discontinuous annotations and prevents invalid nested HTML for overlapping entities. A deterministic color palette is keyed by entity type. Overlapping spans use a striped background derived from their active entity colors. The selected entity receives a strong outline across all of its spans.

The rendered text is display-only. Selection occurs through the annotation table, avoiding a custom bidirectional Streamlit component.

## User Interface

### Dataset controls

The sidebar contains:

- A local directory path field, optionally initialized from `CORAL_DATA_DIR`.
- A **Load dataset** button.
- Cohort and entity-type filters.
- A **Show preannotations and section markers** toggle, off by default. The default visible view contains clinical expert annotations; the dataset summary separately reports the full published expert count, including structural annotations.
- A compact summary of loaded documents, entities, attributes, relationships, and warnings.

Changing filters does not reload files. Loading a new directory resets document and annotation selection.

### Three-panel layout

The main area follows the approved explorer layout:

1. **Document explorer:** searchable document list showing cohort and ID.
2. **Clinical note:** full note text with color-coded highlights and a visible entity-type legend.
3. **Annotation inspector:** selectable entity table followed by details for the selected row.

The entity table contains ID, type, text, and offsets. Selecting a row updates the highlighted note and inspector. The inspector shows all spans, attributes, incoming relationships, and outgoing relationships. Related entities are shown by ID, type, and annotated text.

## Data Flow

1. The user enters a local directory and selects **Load dataset**.
2. The loader discovers and parses paired BRAT files into `DatasetLoadResult`.
3. Streamlit stores the result in session state and presents document filters.
4. Selecting a document supplies its entities to both the highlight renderer and annotation table.
5. Selecting an annotation updates the center-panel emphasis and right-panel details without reparsing the dataset.

The loader may be cached with `st.cache_data` using the resolved directory path and the `.txt`/`.ann` file modification times as the cache key. Note content must never be written to repository logs or new persistent cache files.

## Validation and Error Handling

- A missing or non-directory path produces an actionable sidebar error.
- A `.txt` file without a matching `.ann` file is skipped and listed by relative path.
- Invalid UTF-8, malformed annotation fields, nonnumeric offsets, out-of-range spans, and missing relation targets become document warnings rather than application crashes.
- An empty valid directory displays an empty-state message.
- Duplicate document IDs from different subdirectories remain distinct by using their relative paths as stable keys.
- User-visible failures contain paths, record IDs, and line numbers where available, but never include clinical note text.

## Testing

Use the standard-library `unittest` framework already introduced in this branch.

Unit tests cover:

- Recursive discovery and `.txt`/`.ann` pairing.
- Stable document keys for duplicate filenames.
- Parsing entities, discontinuous spans, attributes, `R` relations, and `*` relations.
- Expert-only filtering and optional preannotation visibility.
- Malformed lines, missing targets, and invalid offsets.
- HTML escaping, overlap rendering, discontinuous highlighting, deterministic colors, and selected-entity emphasis.

A Streamlit smoke test uses `streamlit.testing.v1.AppTest` with a synthetic temporary BRAT dataset. It verifies successful loading and selection without placing real clinical note content in test fixtures or output.

## Dependencies and Running

Add Streamlit to the documented environment dependencies. The app runs from the repository root:

```bash
streamlit run streamlit_app.py
```

The browser connects only to the local Streamlit server. The app performs no network requests after dependencies are installed.

## Acceptance Criteria

- The downloaded CORAL annotation directory loads all 40 paired documents.
- The dataset summary reports the published totals of 9,028 expert entities (including structural annotations), 9,986 attributes/modifiers, and 5,312 schema-valid relationships.
- The note panel renders without altering or leaking source text.
- Every visible entity can be selected through the table and inspected with its attributes and relationships.
- Enabling preannotations adds the automated entity types and section markers without changing the source files.
- All parser, renderer, and Streamlit smoke tests pass.
