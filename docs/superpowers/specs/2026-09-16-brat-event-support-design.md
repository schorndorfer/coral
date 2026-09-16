# BRAT Event and Multiline Annotation Support

## Purpose

Reduce misleading viewer warnings without hiding genuine dataset problems. The viewer will reconstruct multiline text-bound annotations, parse BRAT events, and expose event-linked relationships while preserving CORAL's published relationship total.

## Current behavior

Loading the CORAL release produces 148 warnings:

- 143 continuation lines belonging to multiline `T` records are treated as unsupported records.
- Two `E` event records are unsupported.
- Two otherwise usable relationships are dropped because they target those events.
- One `TreatmentTypeRel` equivalence relation is not declared in `annotation.conf`.

The first three groups are viewer limitations. The undeclared relation is a genuine schema warning and remains visible.

## Parsing model

### Logical BRAT records

Before dispatching records by prefix, the loader assembles physical `.ann` lines into logical records. A physical line beginning with whitespace is appended to the preceding text-bound (`T`) record's reference-text field with a newline. A continuation without a preceding `T` record remains malformed and produces a sanitized warning.

Offsets remain authoritative for rendering. Reconstructed reference text is used only for annotation labels and inspection. The loader never includes annotation text in warnings or exceptions.

### Events

Introduce immutable event records containing an event ID, event type, trigger entity ID, and zero or more named argument targets.

For `E1\tTreatmentDosage:T659`, `T659` is the trigger. Events are retained only when the trigger and every argument target resolve to parsed entities. Malformed records and missing targets produce sanitized structural warnings.

The release has two unary events. The parser will also support standard BRAT events with additional named arguments so the model is not release-specific.

## Relationship resolution

Relationships preserve their original endpoint IDs. When an endpoint references an event, the viewer resolves it to the event's trigger entity for incoming/outgoing discovery, navigation, and related annotation display.

The inspector identifies the event endpoint explicitly, including its event ID and type, so event semantics are not silently flattened. Unknown relationship types remain parsed with `schema_valid=False`; relationships with unresolved entity or event targets remain excluded.

## Metrics

Published metrics retain their established definitions: 40 documents, 9,028 expert entities, 9,986 attributes, and 5,312 schema-valid relationships.

Event-linked relationships are excluded from the published relationship total, matching the release convention. The UI separately displays parsed event and event-linked relationship counts, both expected to equal 2 for this release.

## Streamlit presentation

The dataset summary adds event and event-linked relationship counts. Selecting an event trigger shows attached events. Event-linked relationship rows show direction, relation type, event ID/type, resolved trigger, related entity, and related annotation text. The note highlights the trigger entity rather than events independently.

Warnings are grouped by sanitized category and count. The remaining undeclared `TreatmentTypeRel` warning stays visible, with file/line details in a collapsed diagnostic section. The expected release warning count becomes 1.

## Error handling and privacy

- Never include note text or annotation reference text in warnings, exceptions, logs, or test failure messages.
- Invalid UTF-8, malformed event syntax, orphan continuations, missing targets, and unknown relation types remain sanitized diagnostics.
- Loading remains local and read-only; source files are never modified.

## Testing and acceptance

Tests cover multiline joining, orphan continuations, unary and multi-argument events, malformed events, missing targets, event-linked relationship resolution in both directions, unchanged ordinary relationships and published counts, grouped warnings, and Streamlit event presentation.

Real-release acceptance requires:

- all tests, compilation, and `git diff --check` pass;
- published totals remain `40 / 9028 / 9986 / 5312`;
- parsed events equal 2;
- event-linked relationships equal 2;
- sanitized warnings equal 1 and identify only `TreatmentTypeRel` as unknown;
- a localhost health check succeeds without clinical text in logs.

## Scope exclusions

- Editing source annotation files.
- Changing published CORAL metric definitions.
- Highlighting events independently of triggers.
- Supporting unrelated BRAT normalization or annotator-note records.
