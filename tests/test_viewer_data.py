from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coral.viewer.data import (
    AUXILIARY_ENTITY_TYPES,
    Span,
    load_dataset,
    load_document,
    parse_relation_types,
    visible_entities,
)


NOTE = "Treatment started Monday. Mass remained stable."
ANN = "\n".join(
    [
        "T1\tMedicationName 0 9\tTreatment",
        "T2\tDatetime 18 24\tMonday",
        "T3\tClinicalCondition 26 30;40 46\tMass stable",
        "A1\tNegationModalityVal T1 affirmed",
        "R1\tBeginsOnOrAt Arg1:T1 Arg2:T2",
        "*\tCoreference T1 T3",
    ]
)


class ViewerDataTests(unittest.TestCase):
    def load(self, note: str = NOTE, ann: str = ANN, *, ann_bytes: bytes | None = None):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        note_path = root / "cohort-a" / "note.txt"
        ann_path = note_path.with_suffix(".ann")
        note_path.parent.mkdir()
        note_path.write_text(note, encoding="utf-8")
        if ann_bytes is None:
            ann_path.write_text(ann, encoding="utf-8")
        else:
            ann_path.write_bytes(ann_bytes)
        return load_document(
            note_path,
            ann_path,
            root,
            frozenset({"BeginsOnOrAt", "Coreference"}),
        )

    def test_loads_document_with_discontinuous_entities_and_links(self):
        document = self.load()

        self.assertEqual(document.key, "cohort-a/note.txt")
        self.assertEqual(document.document_id, "note")
        self.assertEqual(document.cohort, "cohort-a")
        self.assertEqual(document.text, NOTE)
        self.assertEqual(
            document.entities[2].spans,
            (Span(26, 30), Span(40, 46)),
        )
        self.assertEqual(document.attributes[0].entity_id, "T1")
        self.assertEqual(document.attributes[0].value, "affirmed")
        self.assertEqual(
            [(relation.id, relation.source_id, relation.target_id) for relation in document.relationships],
            [("R1", "T1", "T2"), ("*:6", "T1", "T3")],
        )
        self.assertTrue(all(relation.schema_valid for relation in document.relationships))
        self.assertEqual(document.warnings, ())

    def test_parses_m_modifiers_with_and_without_values(self):
        document = self.load(
            ann="\n".join(
                [
                    "T1\tMedicationName 0 9\tTreatment",
                    "M1\tNegationModalityVal T1 negated",
                    "M2\tHistorical T1",
                ]
            )
        )

        self.assertEqual(
            [
                (attribute.id, attribute.type, attribute.entity_id, attribute.value)
                for attribute in document.attributes
            ],
            [
                ("M1", "NegationModalityVal", "T1", "negated"),
                ("M2", "Historical", "T1", None),
            ],
        )
        self.assertEqual(document.warnings, ())

    def test_m_modifier_missing_target_is_dropped_with_sanitized_warning(self):
        document = self.load(ann="M1\tHistorical T404 synthetic-value")

        self.assertEqual(document.attributes, ())
        self.assertEqual(
            document.warnings,
            ("cohort-a/note.ann: line 1: attribute references a missing entity",),
        )

    def test_marks_automated_and_structural_entities_as_auxiliary(self):
        annotations = "\n".join(
            f"T{index}\t{entity_type} 0 1\tX"
            for index, entity_type in enumerate(sorted(AUXILIARY_ENTITY_TYPES), start=1)
        )

        document = self.load(note="X", ann=annotations)

        self.assertEqual(len(document.entities), len(AUXILIARY_ENTITY_TYPES))
        self.assertTrue(all(entity.auxiliary for entity in document.entities))

    def test_malformed_offsets_warn_without_annotation_text(self):
        document = self.load(ann="T1\tMedicationName bad 9\tsecret annotation")

        self.assertEqual(document.entities, ())
        self.assertEqual(len(document.warnings), 1)
        self.assertIn("cohort-a/note.ann: line 1", document.warnings[0])
        self.assertNotIn("secret annotation", document.warnings[0])

    def test_out_of_range_spans_warn_without_annotation_text(self):
        document = self.load(ann="T1\tMedicationName 0 99\tsecret annotation")

        self.assertEqual(document.entities, ())
        self.assertEqual(len(document.warnings), 1)
        self.assertIn("cohort-a/note.ann: line 1", document.warnings[0])
        self.assertNotIn("secret annotation", document.warnings[0])

    def test_missing_entity_targets_are_dropped_with_sanitized_warnings(self):
        document = self.load(
            ann="\n".join(
                [
                    "A1\tNegationModalityVal T99 secret-value",
                    "R1\tBeginsOnOrAt Arg1:T1 Arg2:T99",
                ]
            )
        )

        self.assertEqual(document.attributes, ())
        self.assertEqual(document.relationships, ())
        self.assertEqual(len(document.warnings), 2)
        self.assertTrue(all("cohort-a/note.ann: line" in warning for warning in document.warnings))
        self.assertTrue(all("secret-value" not in warning for warning in document.warnings))

    def test_unsupported_record_prefix_warns_without_annotation_text(self):
        document = self.load(ann="E1\tEvent:secret annotation")

        self.assertEqual(len(document.warnings), 1)
        self.assertIn("cohort-a/note.ann: line 1", document.warnings[0])
        self.assertNotIn("secret annotation", document.warnings[0])

    def test_joins_multiline_text_bound_reference_text(self):
        document = self.load(
            note="Alpha\n  Beta",
            ann="T1\tClinicalCondition 0 5;8 12\tAlpha\n  Beta",
        )

        self.assertEqual(document.entities[0].text, "Alpha\n  Beta")
        self.assertEqual(document.warnings, ())

    def test_joins_consecutive_multiline_continuations(self):
        document = self.load(
            note="Alpha\n  Beta\n  Gamma",
            ann="T1\tClinicalCondition 0 5;8 12;15 20\tAlpha\n  Beta\n  Gamma",
        )

        self.assertEqual(document.entities[0].text, "Alpha\n  Beta\n  Gamma")
        self.assertEqual(document.warnings, ())

    def test_orphan_continuation_warns_without_exposing_text(self):
        document = self.load(ann="  synthetic private continuation")

        self.assertEqual(document.entities, ())
        self.assertEqual(
            document.warnings,
            ("cohort-a/note.ann: line 1: orphan annotation continuation",),
        )
        self.assertNotIn("synthetic private continuation", document.warnings[0])

    def test_invalid_utf8_annotation_line_warns_without_annotation_text(self):
        document = self.load(ann_bytes=b"T1\tMedicationName 0 9\tTreatment\n\xffsecret annotation")

        self.assertEqual(len(document.entities), 1)
        self.assertEqual(len(document.warnings), 1)
        self.assertIn("cohort-a/note.ann: line 2", document.warnings[0])
        self.assertNotIn("secret annotation", document.warnings[0])

    def test_unknown_standard_relation_is_retained_as_invalid_with_sanitized_warning(self):
        document = self.load(
            ann="\n".join(
                [
                    "T1\tMedicationName 0 9\tTreatment",
                    "T2\tDatetime 18 24\tMonday",
                    "R1\tUnexpectedRelation Arg1:T1 Arg2:T2",
                ]
            )
        )

        self.assertEqual(len(document.relationships), 1)
        self.assertFalse(document.relationships[0].schema_valid)
        self.assertEqual(len(document.warnings), 1)
        self.assertIn("cohort-a/note.ann: line 3: unknown relation type", document.warnings[0])
        self.assertNotIn("UnexpectedRelation", document.warnings[0])
        self.assertNotIn(NOTE, document.warnings[0])

    def test_loads_relation_with_schema_named_arguments(self):
        document = self.load(
            ann="\n".join(
                [
                    "T1\tMedicationName 0 9\tTreatment",
                    "T2\tDatetime 18 24\tMonday",
                    "R1\tBeginsOnOrAt Treatment:T1 Date:T2",
                ]
            )
        )

        self.assertEqual(
            [(relation.source_id, relation.target_id) for relation in document.relationships],
            [("T1", "T2")],
        )
        self.assertTrue(document.relationships[0].schema_valid)
        self.assertEqual(document.warnings, ())

    def test_drops_relation_with_empty_role_label(self):
        document = self.load(
            ann="\n".join(
                [
                    "T1\tMedicationName 0 9\tTreatment",
                    "T2\tDatetime 18 24\tMonday",
                    "R1\tBeginsOnOrAt :T1 Date:T2",
                ]
            )
        )

        self.assertEqual(document.relationships, ())
        self.assertEqual(
            document.warnings,
            ("cohort-a/note.ann: line 3: malformed relation arguments",),
        )

    def test_unknown_equivalence_relation_is_retained_as_invalid_with_sanitized_warning(self):
        document = self.load(
            ann="\n".join(
                [
                    "T1\tMedicationName 0 9\tTreatment",
                    "T2\tDatetime 18 24\tMonday",
                    "*\tUnexpectedRelation T1 T2",
                ]
            )
        )

        self.assertEqual(len(document.relationships), 1)
        self.assertFalse(document.relationships[0].schema_valid)
        self.assertEqual(len(document.warnings), 1)
        self.assertIn("cohort-a/note.ann: line 3: unknown relation type", document.warnings[0])
        self.assertNotIn("UnexpectedRelation", document.warnings[0])
        self.assertNotIn(NOTE, document.warnings[0])


class DatasetDiscoveryTests(unittest.TestCase):
    def make_dataset(self) -> Path:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "annotated"
        root.mkdir()
        (root / "annotation.conf").write_text(
            "\n".join(
                [
                    "[relations]",
                    "Relates Arg1:<ENTITY>, Arg2:<ENTITY>",
                    "  <ENTITY> continued definition",
                    "<RELATION_MACRO>",
                    "Coreference Arg1:<ENTITY>, Arg2:<ENTITY>",
                    "# this is a comment",
                    "",
                    "[events]",
                    "Event:T1",
                ]
            ),
            encoding="utf-8",
        )
        for cohort, annotation in {
            "pdac": "\n".join(
                [
                    "T1\tMedicationName 0 1\tA",
                    "T2\tPROBLEM 1 2\tB",
                    "T3\tSectionSkip 2 3\tC",
                    "A1\tCertainty T1 high",
                    "A2\tCertainty T2 low",
                    "R1\tRelates Arg1:T1 Arg2:T3",
                    "R2\tUnexpectedRelation Arg1:T1 Arg2:T2",
                ]
            ),
            "breastca": "\n".join(
                [
                    "T1\tMedicationName 0 1\tD",
                    "T2\tSectionSkip 1 2\tE",
                    "T3\tTEST 2 3\tF",
                    "A1\tCertainty T1 high",
                    "R1\tCoreference Arg1:T1 Arg2:T2",
                    "R2\tUnexpectedRelation Arg1:T1 Arg2:T3",
                ]
            ),
        }.items():
            note = root / cohort / "1.txt"
            note.parent.mkdir()
            note.write_text("ABC" if cohort == "pdac" else "DEF", encoding="utf-8")
            note.with_suffix(".ann").write_text(annotation, encoding="utf-8")
        (root / "breastca" / "orphan.txt").write_text("private note", encoding="utf-8")
        return root

    def test_parses_only_relation_names_in_relations_section(self):
        root = self.make_dataset()

        self.assertEqual(
            parse_relation_types(root / "annotation.conf"),
            frozenset({"Relates", "Coreference"}),
        )

    def test_discovers_naturally_sorted_documents_and_counts_published_annotations(self):
        root = self.make_dataset()

        result = load_dataset(root)

        self.assertEqual([document.key for document in result.documents], ["breastca/1", "pdac/1"])
        self.assertEqual(result.counts.documents, 2)
        self.assertEqual(result.counts.expert_entities, 4)
        self.assertEqual(result.counts.attributes, 2)
        self.assertEqual(result.counts.schema_valid_relationships, 2)
        self.assertEqual(
            [relation.schema_valid for document in result.documents for relation in document.relationships],
            [True, False, True, False],
        )
        self.assertTrue(any("breastca/orphan.txt" in warning for warning in result.warnings))
        self.assertTrue(all("private note" not in warning for warning in result.warnings))

    def test_schema_validity_uses_declared_roles_with_generic_role_positions(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "annotated"
        root.mkdir()
        (root / "annotation.conf").write_text(
            "\n".join(
                [
                    "[relations]",
                    "ResultOfTest Desc:<TEST-RES>, Test:<TEST>|ProcedureName",
                    "TestOrProcedureReveals Test:<TEST>|ProcedureName, Desc:ClinicalCondition",
                    "InclusionCriteriaFor Arg1:ClinicalCondition, Arg2:ProcedureName",
                    "[events]",
                ]
            ),
            encoding="utf-8",
        )
        text_path = root / "cohort" / "1.txt"
        text_path.parent.mkdir()
        text_path.write_text("ABC", encoding="utf-8")
        text_path.with_suffix(".ann").write_text(
            "\n".join(
                [
                    "T1\tTestResult 0 1\tA",
                    "T2\tTumorTest 1 2\tB",
                    "T3\tClinicalCondition 2 3\tC",
                    "R1\tResultOfTest Desc:T1 Test:T2",
                    "R2\tResultOfTest Arg1:T2 Reason:T3",
                    "R3\tTestOrProcedureReveals Test:T2 Prob:T3",
                    "R4\tInclusionCriteriaFor Treatment:T2 Prob:T3",
                ]
            ),
            encoding="utf-8",
        )

        result = load_dataset(root)

        self.assertEqual(
            [relation.schema_valid for relation in result.documents[0].relationships],
            [True, False, False, True],
        )
        self.assertEqual(result.counts.schema_valid_relationships, 2)

    def test_visibility_hides_auxiliary_entities_unless_requested(self):
        document = load_dataset(self.make_dataset()).documents[1]

        self.assertEqual([entity.id for entity in visible_entities(document, False)], ["T1"])
        self.assertEqual([entity.id for entity in visible_entities(document, True)], ["T1", "T2", "T3"])
        self.assertEqual(
            visible_entities(document, False, frozenset({"SectionSkip"})),
            (),
        )
        self.assertEqual(
            [entity.id for entity in visible_entities(document, True, frozenset({"SectionSkip"}))],
            ["T3"],
        )

    def test_invalid_or_empty_dataset_paths_return_actionable_warnings(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        file_path = root / "file.txt"
        file_path.write_text("not a directory", encoding="utf-8")
        empty_directory = root / "empty"
        empty_directory.mkdir()

        for path, phrase in (
            (root / "missing", "does not exist"),
            (file_path, "not a directory"),
            (empty_directory, "no .txt files"),
        ):
            result = load_dataset(path)
            self.assertEqual(result.documents, ())
            self.assertEqual(result.counts.documents, 0)
            self.assertTrue(any(phrase in warning for warning in result.warnings))

    def test_natural_sort_handles_numeric_and_alphabetic_document_names(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for name in ("10", "2", "alpha"):
            text_path = root / f"{name}.txt"
            text_path.write_text("X", encoding="utf-8")
            text_path.with_suffix(".ann").write_text("", encoding="utf-8")

        result = load_dataset(root)

        self.assertEqual([document.key for document in result.documents], ["2", "10", "alpha"])


if __name__ == "__main__":
    unittest.main()
