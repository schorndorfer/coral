import tempfile
import unittest
from io import StringIO
from pathlib import Path

import pandas as pd

from scripts.rescore_gpt56_sol_paper_parser import rescore_records
from tests.test_paper_replication_scoring import COMPLETED, SOURCE, FakeMetrics


class PaperParserRescoreTests(unittest.TestCase):
    def test_rescore_writes_separate_comparison_without_changing_current_artifact(self):
        record = {
            **COMPLETED,
            "output_text": (
                "Here is private model prose:\n"
                "- SymptomEnt(Symptom='fatigue', Datetime={'today'})"
            ),
        }
        current_topline = pd.DataFrame([
            {
                "metric": metric,
                "gpt56_sol": 0.0,
                "paper_gpt4": paper,
                "difference_vs_paper": -paper,
            }
            for metric, paper in (("BLEU-4", 0.73), ("ROUGE-1", 0.72), ("EM F1", 0.51))
        ])

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            current_path = output_dir / "gpt56_sol_paper_replication_topline.csv"
            current_topline.to_csv(current_path, index=False)
            current_bytes = current_path.read_bytes()
            output = StringIO()

            result = rescore_records(
                SOURCE,
                [record],
                output_dir=output_dir,
                metrics=FakeMetrics(),
                output=output,
            )

            self.assertEqual(current_path.read_bytes(), current_bytes)
            self.assertEqual(result.changed_records, 1)
            self.assertEqual(result.comparison.paper_source_parser.tolist(), [1.0] * 3)
            self.assertEqual(result.comparison.parser_difference.tolist(), [1.0] * 3)
            self.assertEqual(
                result.artifacts.topline.name,
                "gpt56_sol_paper_replication_paper_parser_topline.csv",
            )
            self.assertTrue(result.artifacts.topline.exists())
            self.assertTrue(result.comparison_path.exists())
            rendered = output.getvalue()
            self.assertIn("Changed parses: 1/1", rendered)
            self.assertNotIn("private model prose", rendered)
            self.assertNotIn("Fatigue today", rendered)


if __name__ == "__main__":
    unittest.main()
