"""Smoke and presenter tests for the local annotation viewer."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit import config

import streamlit_app
from coral.viewer.data import Span, ViewerDocument, ViewerEntity, ViewerRelation
from streamlit_app import annotation_label, related_rows


class PresenterHelperTests(unittest.TestCase):
    """Protect selector and relationship presentation."""

    def setUp(self) -> None:
        self.source = ViewerEntity("T1", "Diagnosis", (Span(0, 4),), "synthetic source", False)
        self.target = ViewerEntity("T2", "Medication", (Span(9, 13),), "synthetic target", False)
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

    def test_related_rows_identify_other_endpoint_and_its_annotation_text(self) -> None:
        """A relationship row must identify the other endpoint and its annotated text."""
        self.assertEqual(
            related_rows(self.document, "T1", "outgoing"),
            [
                {
                    "relationship": "Treats (R1)",
                    "entity": "Medication (T2)",
                    "text": "synthetic target",
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
                    "text": "synthetic source",
                    "offsets": "0-4",
                }
            ],
        )
        for row in related_rows(self.document, "T1", "outgoing"):
            self.assertNotIn(self.document.text, " ".join(row.values()))

    def test_related_rows_exclude_schema_invalid_relations(self) -> None:
        document = ViewerDocument(
            key="cohort/note",
            document_id="note",
            cohort="cohort",
            text="synthetic note",
            entities=(self.source, self.target),
            attributes=(),
            relationships=(ViewerRelation("R9", "Obsolete", "T1", "T2", False),),
            warnings=("cohort/note.ann: line 3: unknown relation type",),
        )

        self.assertEqual(related_rows(document, "T1", "outgoing"), [])

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

    def test_failed_reload_clears_previous_loaded_path_caption(self) -> None:
        state = {
            "dataset_directory": "/synthetic/unavailable",
            "viewer_dataset": object(),
            "viewer_dataset_path": "/synthetic/previous",
        }
        with (
            patch.object(streamlit_app.st, "session_state", state),
            patch.object(streamlit_app, "load_dataset", side_effect=OSError),
        ):
            streamlit_app._load_dataset_from_sidebar()

        self.assertNotIn("viewer_dataset_path", state)
        self.assertNotIn("viewer_dataset", state)
        self.assertIn("viewer_load_error", state)

    def test_project_streamlit_config_is_local_only_and_disables_telemetry(self) -> None:
        self.assertEqual(config.get_option("server.address"), "127.0.0.1")
        self.assertFalse(config.get_option("browser.gatherUsageStats"))

    def test_legend_uses_renderer_color_swatches(self) -> None:
        legend = streamlit_app.legend_html(("MedicationName",))

        self.assertIn('class="coral-legend-swatch"', legend)
        self.assertIn("background-color: #ffd6a5", legend)
        self.assertIn("MedicationName", legend)


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
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(
                metrics,
                {
                    "Documents": "2",
                    "Published expert entities": "2",
                    "Published attributes": "1",
                    "Published schema-valid relationships": "0",
                    "Warnings": "0",
                    "Visible annotations": "1",
                },
            )
            self.assertEqual(app.selectbox[1].options, ["cohort-a / first", "cohort-b / second"])

            app.selectbox[1].set_value("cohort-b / second").run()
            self.assertEqual(app.selectbox[2].options, ["MedicationName (T1) · 0-1"])
            self.assertEqual(
                app.dataframe[0].value.to_dict("records"),
                [{"ID": "T1", "Type": "MedicationName", "Text": "B", "Offsets": "0-1"}],
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
                        "text": "eta",
                        "offsets": "1-2",
                    }
                ],
                dataframe_records,
            )

            app.toggle[0].set_value(True).run()
            self.assertIn("PROBLEM (T2) · 1-2", app.selectbox[2].options)
            self.assertTrue(
                any('class="coral-legend-swatch"' in element.value for element in app.markdown)
            )
            self.assertEqual(len(app.exception), 0)

    def test_cohort_fallback_resets_document_specific_entity_state(self) -> None:
        from streamlit.testing.v1 import AppTest

        root = self.make_dataset()
        with patch.dict(os.environ, {"CORAL_DATA_DIR": str(root)}, clear=False):
            app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py")
            app.run()
            app.button[0].click().run()
            app.selectbox[1].set_value("cohort-b / second").run()
            app.toggle[0].set_value(True).run()
            app.multiselect[0].set_value(["PROBLEM"]).run()

            app.selectbox[0].set_value("cohort-a").run()

            self.assertEqual(app.selectbox[1].value, "cohort-a/first")
            self.assertEqual(app.multiselect[0].value, ["MedicationName"])
            self.assertEqual(app.selectbox[2].options, ["MedicationName (T1) · 0-5"])
            self.assertEqual(len(app.exception), 0)

    def test_search_fallback_resets_document_specific_entity_state(self) -> None:
        from streamlit.testing.v1 import AppTest

        root = self.make_dataset()
        with patch.dict(os.environ, {"CORAL_DATA_DIR": str(root)}, clear=False):
            app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py")
            app.run()
            app.button[0].click().run()
            app.selectbox[1].set_value("cohort-b / second").run()
            app.toggle[0].set_value(True).run()
            app.multiselect[0].set_value(["PROBLEM"]).run()

            app.text_input[0].set_value("first").run()

            self.assertEqual(app.selectbox[1].value, "cohort-a/first")
            self.assertEqual(app.multiselect[0].value, ["MedicationName"])
            self.assertEqual(app.selectbox[2].options, ["MedicationName (T1) · 0-5"])
            self.assertEqual(len(app.exception), 0)

    def test_unsuccessful_reload_clears_loaded_path_caption(self) -> None:
        from streamlit.testing.v1 import AppTest

        root = self.make_dataset()
        with patch.dict(os.environ, {"CORAL_DATA_DIR": str(root)}, clear=False):
            app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py")
            app.run()
            app.button[0].click().run()
            self.assertTrue(any("Loaded:" in caption.value for caption in app.caption))

            app.text_input[1].set_value(str(root / "missing")).run()
            app.button[0].click().run()

            self.assertFalse(any("Loaded:" in caption.value for caption in app.caption))
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
