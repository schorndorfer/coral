import unittest
from io import StringIO

import pandas as pd

from scripts.diagnose_paper_replication_scoring import diagnose_records


class ExplodingMetrics:
    def compute_bleu_score(self, **_kwargs):
        raise RuntimeError("sensitive metric failure")


class ScoringDiagnosticTests(unittest.TestCase):
    def test_failure_reports_safe_identity_without_sensitive_payloads(self):
        source = pd.DataFrame([{
            "doc_idx": "7",
            "section_name": "hpi",
            "section_text": "private source text",
            "task": "symptoms",
            "annotation_set": "SymptomEnt(Symptom='fatigue', Datetime={'unknown'})",
        }])
        records = [{
            "doc_idx": "7",
            "section_name": "hpi",
            "task": "symptoms",
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output_text": (
                "SymptomEnt(Symptom='private model output', Datetime={'unknown'})"
            ),
        }]
        output = StringIO()

        exit_code = diagnose_records(
            source,
            records,
            metrics=ExplodingMetrics(),
            model="gpt-5.6-sol",
            output=output,
        )

        rendered = output.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("FAILED: 1 7 hpi symptoms RuntimeError", rendered)
        for sensitive in (
            "private source text",
            "private model output",
            "sensitive metric failure",
        ):
            self.assertNotIn(sensitive, rendered)


if __name__ == "__main__":
    unittest.main()
