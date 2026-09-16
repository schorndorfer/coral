from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coral.viewer.data import AUXILIARY_ENTITY_TYPES, Span, load_document


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


if __name__ == "__main__":
    unittest.main()
