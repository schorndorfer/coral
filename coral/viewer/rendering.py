"""Pure HTML rendering helpers for safely highlighted annotation notes."""

from __future__ import annotations

import hashlib
from html import escape
from typing import Sequence

from coral.viewer.data import Span, ViewerEntity


_ACCESSIBLE_COLORS = (
    "#b8e3ff",
    "#ffd6a5",
    "#c7f0d8",
    "#e4c1f9",
    "#fde2e4",
    "#d9ed92",
    "#cddafd",
    "#f8c8dc",
)


def entity_color(entity_type: str) -> str:
    """Return a deterministic, readable highlight color for an entity type."""
    digest = hashlib.sha256(entity_type.encode("utf-8")).digest()
    return _ACCESSIBLE_COLORS[int.from_bytes(digest[:8], "big") % len(_ACCESSIBLE_COLORS)]


def format_offsets(entity: ViewerEntity) -> str:
    """Format the BRAT character ranges for display in an entity label."""
    return "; ".join(f"{span.start}-{span.end}" for span in entity.spans)


def render_highlighted_text(
    text: str,
    entities: Sequence[ViewerEntity],
    selected_entity_id: str | None = None,
) -> str:
    """Return escaped note HTML with all valid entity spans highlighted.

    A boundary sweep means each source interval is emitted once, even when
    entities overlap, touch, or contain multiple discontinuous spans.
    """
    spans = tuple(_valid_spans(text, entities))
    boundaries = {0, len(text)}
    for _, span in spans:
        boundaries.update((span.start, span.end))

    rendered: list[str] = []
    ordered_boundaries = sorted(boundaries)
    for start, end in zip(ordered_boundaries, ordered_boundaries[1:]):
        active_entities = _active_entities(spans, start, end)
        fragment = escape(text[start:end], quote=True)
        if active_entities:
            rendered.append(_highlight(fragment, active_entities, selected_entity_id))
        else:
            rendered.append(fragment)

    return (
        '<div class="coral-note" '
        'style="white-space: pre-wrap; overflow-wrap: anywhere">'
        f"{''.join(rendered)}</div>"
    )


def _valid_spans(text: str, entities: Sequence[ViewerEntity]):
    for entity in entities:
        for span in entity.spans:
            if 0 <= span.start < span.end <= len(text):
                yield entity, span


def _active_entities(
    spans: Sequence[tuple[ViewerEntity, Span]], start: int, end: int
) -> tuple[ViewerEntity, ...]:
    active_by_id: dict[str, ViewerEntity] = {}
    for entity, span in spans:
        if span.start <= start and end <= span.end:
            active_by_id[entity.id] = entity
    return tuple(
        active_by_id[entity_id]
        for entity_id in sorted(active_by_id, key=lambda identifier: (identifier, active_by_id[identifier].type))
    )


def _highlight(
    fragment: str, active_entities: Sequence[ViewerEntity], selected_entity_id: str | None
) -> str:
    entity_ids = " ".join(entity.id for entity in active_entities)
    entity_types = " ".join(entity.type for entity in active_entities)
    label = ", ".join(f"{entity.type} ({entity.id})" for entity in active_entities)
    colors = tuple(entity_color(entity.type) for entity in active_entities)
    style = _highlight_style(colors, any(entity.id == selected_entity_id for entity in active_entities))
    selected = ' data-selected="true"' if any(
        entity.id == selected_entity_id for entity in active_entities
    ) else ""
    return (
        '<span class="coral-annotation"'
        f' data-entity-ids="{escape(entity_ids, quote=True)}"'
        f' data-entity-types="{escape(entity_types, quote=True)}"'
        f' title="{escape(label, quote=True)}"'
        f' aria-label="{escape(label, quote=True)}"'
        f' style="{style}"{selected}>{fragment}</span>'
    )


def _highlight_style(colors: Sequence[str], selected: bool) -> str:
    if len(colors) == 1:
        background = f"background-color: {colors[0]}"
    else:
        stops = ", ".join(
            f"{color} {index * 100 / len(colors):.6g}% {(index + 1) * 100 / len(colors):.6g}%"
            for index, color in enumerate(colors)
        )
        background = f"background: linear-gradient(135deg, {stops})"
    outline = "; outline: 3px solid #111827; outline-offset: 1px" if selected else ""
    return f"{background}{outline}"
