import warnings
import unittest

import pandas as pd

from coral.benchmarking.evaluate_model import _reorganize_scores_df


class ReformatScoresTests(unittest.TestCase):
    def test_reformat_replaces_plotting_label_without_chained_assignment_warning(self):
        scores = pd.DataFrame(
            [
                {
                    "task": "other_task",
                    "model": "gpt-5.6-sol",
                    "subrelation": "PrescribedMedicationName PotentialAdvEvent",
                    "mean_bleu4": 0.5,
                }
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ChainedAssignmentError)
            reformatted = _reorganize_scores_df(scores, "relation")

        self.assertEqual(
            reformatted.iloc[0]["Relation"],
            "PrescribedMedicationName PotentialAdverseEvent",
        )


if __name__ == "__main__":
    unittest.main()
