"""Smoke and presenter tests for the local annotation viewer."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coral.viewer.data import Span, ViewerDocument, ViewerEntity, ViewerRelation
from streamlit_app import annotation_label, related_rows


class PresenterHelperTests(unittest.TestCase):
    """Protect metadata-only selector and relationship presentation."""

    def setUp(self) -> None:
        self.source = ViewerEntity("T1", "Diagnosis", (Span(0, 4),), "same", False)
        self.target = ViewerEntity("T2", "Medication", (Span(9, 13),), "same", False)
        self.document = ViewerDocument(
            key="cohort/note",
            document_id="note",
            cohort="cohort",
            text="same note same",
            entities=(self.source, self.target),
            attributes=(),
            relationships=(ViewerRelation("R1", "Treats", "T1", "T2", True),),
            warnings=(),
        )

    def test_annotation_label_identifies_duplicate_text_by_type_and_id(self) -> None:
        """A label losing type or ID would make duplicate-text annotations ambiguous."""
        self.assertEqual(annotation_label(self.source), "Diagnosis (T1) · 0-4")
        self.assertEqual(annotation_label(self.target), "Medication (T2) · 9-13")

    def test_related_rows_identify_other_endpoint_without_note_text(self) -> None:
        """A row choosing the selected endpoint or including text misleads/leaks note content."""
        self.assertEqual(
            related_rows(self.document, "T1", "outgoing"),
            [
                {
                    "relationship": "Treats (R1)",
                    "entity": "Medication (T2)",
                    "offsets": "9-13",
                }
            ],
        )
        self.assertEqual(
            related_rows(self.document, "T2", "incoming"),
            [
                {
                    "relationship": "Treats (R1)",
                    "entity": "Diagnosis (T1)",
                    "offsets": "0-4",
                }
            ],
        )
        for row in related_rows(self.document, "T1", "outgoing"):
            self.assertNotIn(self.document.text, " ".join(row.values()))
            self.assertNotIn(self.source.text, " ".join(row.values()))

    def test_related_rows_omit_missing_related_entities(self) -> None:
        """A dangling relationship must not expose an incomplete presenter row."""
        document = ViewerDocument(
            key="cohort/note",
            document_id="note",
            cohort="cohort",
            text="synthetic note",
            entities=(self.source,),
            attributes=(),
            relationships=(ViewerRelation("R2", "Treats", "T1", "T99", True),),
            warnings=(),
        )
        self.assertEqual(related_rows(document, "T1", "outgoing"), [])


class StreamlitAppSmokeTests(unittest.TestCase):
    """Exercise the app against local synthetic BRAT records only."""

    def make_dataset(self) -> Path:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "annotated"
        root.mkdir()
        (root / "annotation.conf").write_text(
            "[relations]\nRelates Arg1:<ENTITY>, Arg2:<ENTITY>\n[events]\n",
            encoding="utf-8",
        )
        first = root / "cohort-a" / "first.txt"
        first.parent.mkdir()
        first.write_text("Alpha", encoding="utf-8")
        first.with_suffix(".ann").write_text(
            "T1\tMedicationName 0 5\tAlpha",
            encoding="utf-8",
        )
        second = root / "cohort-b" / "second.txt"
        second.parent.mkdir()
        second.write_text("Beta", encoding="utf-8")
        second.with_suffix(".ann").write_text(
            "\n".join(
                [
                    "T1\tMedicationName 0 1\tB",
                    "T2\tPROBLEM 1 2\teta",
                    "A1\tCertainty T1 high",
                    "R1\tRelates Arg1:T1 Arg2:T2",
                ]
            ),
            encoding="utf-8",
        )
        return root

    def test_loads_and_explores_local_synthetic_annotations(self) -> None:
        """Broken UI state would hide documents, inspector metadata, or auxiliary choices."""
        from streamlit.testing.v1 import AppTest

        root = self.make_dataset()
        with patch.dict(os.environ, {"CORAL_DATA_DIR": str(root)}, clear=False):
            app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py")
            app.run()
            app.button[0].click().run()

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[0].value, "2")
            self.assertEqual(app.selectbox[1].options, ["cohort-a / first", "cohort-b / second"])

            app.selectbox[1].set_value("cohort-b / second").run()
            self.assertEqual(app.selectbox[2].options, ["MedicationName (T1) · 0-1"])
            self.assertEqual(
                app.dataframe[0].value.to_dict("records"),
                [{"ID": "T1", "Type": "MedicationName", "Offsets": "0-1"}],
            )
            app.selectbox[2].set_value("MedicationName (T1) · 0-1").run()
            markdown_values = [element.value for element in app.markdown]
            self.assertIn("Type: MedicationName", markdown_values)
            self.assertIn("Offsets: 0-1", markdown_values)
            dataframe_records = [element.value.to_dict("records") for element in app.dataframe]
            self.assertIn([{"attribute": "Certainty", "value": "high"}], dataframe_records)
            self.assertIn(
                [
                    {
                        "relationship": "Relates (R1)",
                        "entity": "PROBLEM (T2)",
                        "offsets": "1-2",
                    }
                ],
                dataframe_records,
            )

            app.toggle[0].set_value(True).run()
            self.assertIn("PROBLEM (T2) · 1-2", app.selectbox[2].options)
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
