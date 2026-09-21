import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coral import task_to_default_tuple_dict
from coral.benchmarking.evaluate_model import get_outputs


class EvaluateModelTests(unittest.TestCase):
    def test_get_outputs_treats_blank_csv_output_as_empty_model_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "outputs.csv"
            pd.DataFrame([
                {
                    "doc_idx": "1",
                    "section_name": "hpi",
                    "section_text": "No symptoms.",
                    "task": "symptoms",
                    "model": "qwen",
                    "output": "",
                    "conversion_status": "validated",
                }
            ]).to_csv(output_path, index=False)

            parsed = get_outputs(output_path.name, temporary_directory)

        self.assertEqual(
            parsed.iloc[0].proc_outputs,
            [task_to_default_tuple_dict["symptoms"]],
        )
