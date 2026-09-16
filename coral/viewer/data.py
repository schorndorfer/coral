"""Safe, read-only parsing of individual BRAT annotation documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


AUTOMATED_ENTITY_TYPES = frozenset({"PROBLEM", "TREATMENT", "TEST", "SectionAnnotate"})
STRUCTURAL_ENTITY_TYPES = frozenset({"SectionSkip", "hpi_start", "hpi_end", "ap_start", "ap_end"})
AUXILIARY_ENTITY_TYPES = AUTOMATED_ENTITY_TYPES | STRUCTURAL_ENTITY_TYPES


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class ViewerEntity:
    id: str
    type: str
    spans: tuple[Span, ...]
    text: str
    auxiliary: bool


@dataclass(frozen=True)
class ViewerAttribute:
    id: str
    type: str
    entity_id: str
    value: str | None


@dataclass(frozen=True)
class ViewerRelation:
    id: str
    type: str
    source_id: str
    target_id: str
    schema_valid: bool


@dataclass(frozen=True)
class ViewerDocument:
    key: str
    document_id: str
    cohort: str
    text: str
    entities: tuple[ViewerEntity, ...]
    attributes: tuple[ViewerAttribute, ...]
    relationships: tuple[ViewerRelation, ...]
    warnings: tuple[str, ...]


def load_document(
    text_path: Path,
    ann_path: Path,
    root: Path,
    known_relation_types: frozenset[str],
) -> ViewerDocument:
    """Load one note and its BRAT sidecar without exposing malformed input in warnings."""
    text_path = Path(text_path)
    ann_path = Path(ann_path)
    root = Path(root)
    warnings: list[str] = []
    text = _read_text(text_path, root, warnings)
    annotation_filename = _relative_filename(ann_path, root)

    entities: list[ViewerEntity] = []
    attributes_with_lines: list[tuple[int, ViewerAttribute]] = []
    relations_with_lines: list[tuple[int, ViewerRelation]] = []

    for line_number, raw_line in enumerate(_read_lines(ann_path), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            _warn(warnings, annotation_filename, line_number, "invalid UTF-8")
            continue

        if not line.strip():
            continue
        record_id = line.split("\t", 1)[0]
        if record_id.startswith("T"):
            entity = _parse_entity(line, len(text), annotation_filename, line_number, warnings)
            if entity is not None:
                entities.append(entity)
        elif record_id.startswith("A"):
            attribute = _parse_attribute(line, annotation_filename, line_number, warnings)
            if attribute is not None:
                attributes_with_lines.append((line_number, attribute))
        elif record_id.startswith("R"):
            relation = _parse_relation(
                line, annotation_filename, line_number, known_relation_types, warnings
            )
            if relation is not None:
                relations_with_lines.append((line_number, relation))
        elif record_id == "*":
            relation = _parse_equivalence_relation(
                line, line_number, annotation_filename, known_relation_types, warnings
            )
            if relation is not None:
                relations_with_lines.append((line_number, relation))
        else:
            _warn(warnings, annotation_filename, line_number, "unsupported record type")

    entity_ids = {entity.id for entity in entities}
    attributes = tuple(
        attribute
        for line_number, attribute in attributes_with_lines
        if _has_attribute_target(attribute, entity_ids, annotation_filename, line_number, warnings)
    )
    relationships = tuple(
        relation
        for line_number, relation in relations_with_lines
        if _has_relation_targets(relation, entity_ids, annotation_filename, line_number, warnings)
    )

    relative_text_path = _relative_filename(text_path, root)
    cohort = _relative_parent(text_path, root)
    return ViewerDocument(
        key=relative_text_path,
        document_id=text_path.stem,
        cohort=cohort,
        text=text,
        entities=tuple(entities),
        attributes=attributes,
        relationships=relationships,
        warnings=tuple(warnings),
    )


def _parse_entity(
    line: str, text_length: int, filename: str, line_number: int, warnings: list[str]
) -> ViewerEntity | None:
    fields = line.split("\t", 2)
    if len(fields) != 3:
        _warn(warnings, filename, line_number, "malformed entity record")
        return None
    descriptor = fields[1].split(maxsplit=1)
    if len(descriptor) != 2:
        _warn(warnings, filename, line_number, "malformed entity descriptor")
        return None
    entity_type, offsets = descriptor
    spans: list[Span] = []
    try:
        for offset in offsets.split(";"):
            start_end = offset.split()
            if len(start_end) != 2:
                raise ValueError
            start, end = (int(value) for value in start_end)
            if not 0 <= start < end <= text_length:
                raise ValueError
            spans.append(Span(start, end))
    except ValueError:
        _warn(warnings, filename, line_number, "invalid entity offsets")
        return None
    return ViewerEntity(
        id=fields[0],
        type=entity_type,
        spans=tuple(spans),
        text=fields[2],
        auxiliary=entity_type in AUXILIARY_ENTITY_TYPES,
    )


def _parse_attribute(
    line: str, filename: str, line_number: int, warnings: list[str]
) -> ViewerAttribute | None:
    fields = line.split("\t", 1)
    if len(fields) != 2:
        _warn(warnings, filename, line_number, "malformed attribute record")
        return None
    descriptor = fields[1].split(maxsplit=2)
    if len(descriptor) < 2:
        _warn(warnings, filename, line_number, "malformed attribute descriptor")
        return None
    return ViewerAttribute(
        id=fields[0],
        type=descriptor[0],
        entity_id=descriptor[1],
        value=descriptor[2] if len(descriptor) == 3 else None,
    )


def _parse_relation(
    line: str,
    filename: str,
    line_number: int,
    known_relation_types: frozenset[str],
    warnings: list[str],
) -> ViewerRelation | None:
    fields = line.split("\t", 1)
    if len(fields) != 2:
        _warn(warnings, filename, line_number, "malformed relation record")
        return None
    descriptor = fields[1].split()
    if not descriptor:
        _warn(warnings, filename, line_number, "malformed relation descriptor")
        return None
    source_id = _argument_id(descriptor[1:], "Arg1:")
    target_id = _argument_id(descriptor[1:], "Arg2:")
    if source_id is None or target_id is None:
        _warn(warnings, filename, line_number, "malformed relation arguments")
        return None
    schema_valid = descriptor[0] in known_relation_types
    if not schema_valid:
        _warn(warnings, filename, line_number, "unknown relation type")
    return ViewerRelation(
        id=fields[0],
        type=descriptor[0],
        source_id=source_id,
        target_id=target_id,
        schema_valid=schema_valid,
    )


def _parse_equivalence_relation(
    line: str,
    line_number: int,
    filename: str,
    known_relation_types: frozenset[str],
    warnings: list[str],
) -> ViewerRelation | None:
    fields = line.split("\t", 1)
    descriptor = fields[1].split() if len(fields) == 2 else []
    if len(descriptor) < 3:
        _warn(warnings, filename, line_number, "malformed equivalence relation")
        return None
    schema_valid = descriptor[0] in known_relation_types
    if not schema_valid:
        _warn(warnings, filename, line_number, "unknown relation type")
    return ViewerRelation(
        id=f"*:{line_number}",
        type=descriptor[0],
        source_id=descriptor[1],
        target_id=descriptor[2],
        schema_valid=schema_valid,
    )


def _argument_id(arguments: list[str], prefix: str) -> str | None:
    for argument in arguments:
        if argument.startswith(prefix) and len(argument) > len(prefix):
            return argument[len(prefix) :]
    return None


def _has_attribute_target(
    attribute: ViewerAttribute,
    entity_ids: set[str],
    filename: str,
    line_number: int,
    warnings: list[str],
) -> bool:
    if attribute.entity_id in entity_ids:
        return True
    _warn(warnings, filename, line_number, "attribute references a missing entity")
    return False


def _has_relation_targets(
    relation: ViewerRelation,
    entity_ids: set[str],
    filename: str,
    line_number: int,
    warnings: list[str],
) -> bool:
    if relation.source_id in entity_ids and relation.target_id in entity_ids:
        return True
    _warn(warnings, filename, line_number, "relationship references a missing entity")
    return False


def _read_text(path: Path, root: Path, warnings: list[str]) -> str:
    raw_text = path.read_bytes()
    try:
        return raw_text.decode("utf-8")
    except UnicodeDecodeError as error:
        _warn(warnings, _relative_filename(path, root), raw_text[: error.start].count(b"\n") + 1, "invalid UTF-8")
        return raw_text.decode("utf-8", errors="replace")


def _read_lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines()


def _relative_filename(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _relative_parent(path: Path, root: Path) -> str:
    try:
        parent = path.parent.relative_to(root)
    except ValueError:
        return path.parent.name
    return "" if parent == Path(".") else parent.as_posix()


def _warn(warnings: list[str], filename: str, line_number: int, reason: str) -> None:
    warnings.append(f"{filename}: line {line_number}: {reason}")
