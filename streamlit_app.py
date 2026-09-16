"""Local, read-only Streamlit explorer for CORAL BRAT annotations."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path

import streamlit as st

from coral.viewer.data import DatasetLoadResult, ViewerDocument, ViewerEntity, load_dataset, visible_entities
from coral.viewer.rendering import entity_color, format_offsets, render_highlighted_text


def annotation_label(entity: ViewerEntity) -> str:
    """Return an unambiguous, metadata-only label for an annotation selector."""
    return f"{entity.type} ({entity.id}) · {format_offsets(entity)}"


def related_rows(document: ViewerDocument, entity_id: str, direction: str) -> list[dict[str, str]]:
    """Return schema-valid relationship rows adjacent to an entity.

    Missing relation endpoints are ignored so malformed input cannot break the
    inspector.
    """
    if direction not in {"incoming", "outgoing"}:
        raise ValueError("direction must be incoming or outgoing")

    entities_by_id = {entity.id: entity for entity in document.entities}
    rows: list[dict[str, str]] = []
    for relation in document.relationships:
        if not relation.schema_valid:
            continue
        if direction == "incoming" and relation.target_id == entity_id:
            related_entity_id = relation.source_id
        elif direction == "outgoing" and relation.source_id == entity_id:
            related_entity_id = relation.target_id
        else:
            continue
        related_entity = entities_by_id.get(related_entity_id)
        if related_entity is None:
            continue
        rows.append(
            {
                "relationship": f"{relation.type} ({relation.id})",
                "entity": f"{related_entity.type} ({related_entity.id})",
                "text": related_entity.text,
                "offsets": format_offsets(related_entity),
            }
        )
    return rows


def _document_label(document: ViewerDocument) -> str:
    return f"{document.cohort} / {document.document_id}" if document.cohort else document.document_id


def legend_html(entity_types: tuple[str, ...] | list[str]) -> str:
    """Return an escaped HTML legend using the renderer's type colors."""
    if not entity_types:
        return '<div class="coral-legend">No visible annotation types</div>'
    items = "".join(
        '<span class="coral-legend-item" style="display: inline-flex; align-items: center; '
        'gap: 0.3rem; margin: 0 0.75rem 0.4rem 0">'
        '<span class="coral-legend-swatch" '
        f'style="display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px; '
        f'background-color: {entity_color(entity_type)}"></span>'
        f"{escape(entity_type, quote=True)}</span>"
        for entity_type in entity_types
    )
    return f'<div class="coral-legend">{items}</div>'


def _reset_document_state() -> None:
    st.session_state.pop("selected_entity_id", None)
    st.session_state.pop("selected_entity_types", None)


def _reset_type_filter() -> None:
    st.session_state.pop("selected_entity_types", None)


def _load_dataset_from_sidebar() -> None:
    raw_path = st.session_state.get("dataset_directory", "")
    try:
        loaded = load_dataset(raw_path)
        resolved_path = str(Path(raw_path).expanduser().resolve()) if raw_path else ""
    except (OSError, ValueError):
        st.session_state.pop("viewer_dataset", None)
        st.session_state.pop("viewer_dataset_path", None)
        st.session_state.pop("effective_document_key", None)
        st.session_state["viewer_load_error"] = "Unable to load the local dataset. Check the directory."
        return

    st.session_state["viewer_dataset"] = loaded
    if loaded.documents:
        st.session_state["viewer_dataset_path"] = resolved_path
    else:
        st.session_state.pop("viewer_dataset_path", None)
    st.session_state.pop("viewer_load_error", None)
    st.session_state.pop("selected_document_key", None)
    st.session_state.pop("effective_document_key", None)
    _reset_document_state()


def _select_document(documents: tuple[ViewerDocument, ...]) -> ViewerDocument | None:
    options = [document.key for document in documents]
    if not options:
        if st.session_state.get("effective_document_key") is not None:
            _reset_document_state()
            st.session_state["effective_document_key"] = None
        return None
    selected_key = st.session_state.get("selected_document_key")
    effective_key = selected_key if selected_key in options else options[0]
    if st.session_state.get("effective_document_key") != effective_key:
        _reset_document_state()
        st.session_state["effective_document_key"] = effective_key
    if selected_key != effective_key:
        st.session_state["selected_document_key"] = effective_key
    documents_by_key = {document.key: document for document in documents}
    selected_key = st.selectbox(
        "Document",
        options,
        format_func=lambda key: _document_label(documents_by_key[key]),
        key="selected_document_key",
    )
    return documents_by_key[selected_key]


def _show_entity_inspector(
    document: ViewerDocument,
    displayed_entities: tuple[ViewerEntity, ...],
    selected_entity: ViewerEntity | None,
) -> None:
    st.subheader("Annotations")
    entity_rows = [
        {
            "ID": entity.id,
            "Type": entity.type,
            "Text": entity.text,
            "Offsets": format_offsets(entity),
        }
        for entity in displayed_entities
    ]
    st.dataframe(entity_rows, hide_index=True, width="stretch")

    if selected_entity is None:
        st.info("Choose an annotation to inspect its metadata.")
        return

    st.subheader("Selected annotation")
    st.write(f"Type: {selected_entity.type}")
    st.write(f"Offsets: {format_offsets(selected_entity)}")

    attributes = [
        {"attribute": attribute.type, "value": attribute.value or ""}
        for attribute in document.attributes
        if attribute.entity_id == selected_entity.id
    ]
    st.subheader("Attributes")
    if attributes:
        st.dataframe(attributes, hide_index=True, width="stretch")
    else:
        st.caption("No attributes")

    for direction, heading in (("incoming", "Incoming relationships"), ("outgoing", "Outgoing relationships")):
        st.subheader(heading)
        rows = related_rows(document, selected_entity.id, direction)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("None")


def _show_warnings(dataset: DatasetLoadResult, document: ViewerDocument | None = None) -> None:
    document_warnings = document.warnings if document is not None else ()
    warnings = tuple(dict.fromkeys((*dataset.warnings, *document_warnings)))
    if not warnings:
        return
    st.subheader("Warnings")
    for warning in warnings:
        st.warning(warning)


def main() -> None:
    st.set_page_config(page_title="CORAL Annotation Viewer", layout="wide")
    st.title("CORAL Annotation Viewer")
    st.caption("Local, read-only exploration of BRAT annotations")

    with st.sidebar:
        st.header("Dataset")
        st.text_input(
            "Dataset directory",
            value=os.environ.get("CORAL_DATA_DIR", ""),
            key="dataset_directory",
        )
        st.button("Load dataset", on_click=_load_dataset_from_sidebar, type="primary")
        if st.session_state.get("viewer_dataset_path"):
            st.caption(f"Loaded: {st.session_state['viewer_dataset_path']}")

    if load_error := st.session_state.get("viewer_load_error"):
        st.error(load_error)

    dataset = st.session_state.get("viewer_dataset")
    if not isinstance(dataset, DatasetLoadResult):
        st.info("Choose a local BRAT annotation directory, then load the dataset.")
        return
    if not dataset.documents:
        st.warning("No annotated documents are available in this directory.")
        _show_warnings(dataset)
        return

    with st.sidebar:
        st.caption("Published expert totals (independent of active filters)")
        st.metric("Documents", dataset.counts.documents)
        st.metric("Published expert entities", dataset.counts.expert_entities)
        st.metric("Published attributes", dataset.counts.attributes)
        st.metric(
            "Published schema-valid relationships",
            dataset.counts.schema_valid_relationships,
        )
        st.metric("Warnings", len(dataset.warnings))

    left, center, right = st.columns([2, 5, 3])
    with left:
        st.subheader("Documents")
        cohorts = sorted({document.cohort for document in dataset.documents})
        cohort = st.selectbox("Cohort", ["All cohorts", *cohorts])
        search = st.text_input("Document search")
        filtered_documents = tuple(
            document
            for document in dataset.documents
            if (cohort == "All cohorts" or document.cohort == cohort)
            and search.casefold() in _document_label(document).casefold()
        )
        document = _select_document(filtered_documents)
        if document is None:
            st.info("No documents match the active filters.")
            return

    with right:
        st.subheader("Filters")
        show_auxiliary = st.toggle(
            "Show auxiliary annotations",
            value=False,
            on_change=_reset_type_filter,
        )
        type_options = sorted(
            {entity.type for entity in visible_entities(document, show_auxiliary)}
        )
        selected_types = st.multiselect(
            "Entity types",
            type_options,
            default=type_options,
            key="selected_entity_types",
        )
        displayed_entities = visible_entities(
            document, show_auxiliary, frozenset(selected_types)
        )
        st.metric("Visible annotations", len(displayed_entities))
        entity_ids = [entity.id for entity in displayed_entities]
        entities_by_id = {entity.id: entity for entity in displayed_entities}
        if st.session_state.get("selected_entity_id") not in entity_ids:
            st.session_state["selected_entity_id"] = entity_ids[0] if entity_ids else None
        selected_entity_id = st.selectbox(
            "Annotation",
            entity_ids,
            format_func=lambda entity_id: annotation_label(entities_by_id[entity_id]),
            key="selected_entity_id",
            disabled=not entity_ids,
        ) if entity_ids else None
        selected_entity = entities_by_id.get(selected_entity_id)
        _show_entity_inspector(document, displayed_entities, selected_entity)
        _show_warnings(dataset, document)

    with center:
        st.subheader("Annotated note")
        legend_types = sorted({entity.type for entity in displayed_entities})
        st.markdown(legend_html(legend_types), unsafe_allow_html=True)
        rendered_note = render_highlighted_text(
            document.text,
            displayed_entities,
            selected_entity_id,
        )
        st.markdown(rendered_note, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
