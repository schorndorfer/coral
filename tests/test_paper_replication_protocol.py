import hashlib
import unittest
from pathlib import Path

import pandas as pd

from coral.paper_replication.grid import build_request_grid, load_source
from coral.paper_replication.protocol import (
    MAX_OUTPUT_TOKENS,
    PAPER_PREAMBLE,
    PAPER_SOURCE_COMMIT,
    REASONING_EFFORT,
    TASK_ORDER,
    TASK_PROMPTS,
)


EXPECTED_PROMPT_HASHES = {
    "symptoms": "8a8a8b45bd1317bfb810412329bd09874370ba6caf64e8cd9d42af6b7ef91fb6",
    "symptoms_at_diagnosis": "2cc8c90dad86ebd4ae17b15816e7c7e03e026aaf2434155683ca17afbba11d0c",
    "symptoms_due_to_cancer": "b295dc2feca3d5efee332d49876943b7bb5b88390da09de4731d638d51ef7e7e",
    "radtest_datetime_site_reason_result": "61bf4578d25c42f435387ddb103a4ffa34ecb7ed64d9e511aec931c40cc7c472",
    "procedure_datetime_site_reason_result": "234a7d06f10190cb3415c493bbf44fa1504cdbf6b555d9d1b7f7f1b47cf5c886",
    "genomictest_datetime_result": "ef01001655a00f11c7ea733c53093a6fd1e597a06e8cf3805db2f5301eca161b",
    "biomarker_datetime": "1d167fac5fbbba18f383e439eead7c89ea55f7128aca137d86cc953881ac8dbb",
    "histology_datetime": "34b8a648ca7743cbf77a5992ef8267da0427e1ac483f2675269cbd81ce259862",
    "metastasis_site_procedure_datetime": "8b9e66a0641a0aab787ba53ccba3f8eaab73b3655b4b447927608d66176b0434",
    "stage_datetime_addtest": "a3e0d85dc34316dab086edd8116c01dc5832224d0c842087c204d292d7f10a8d",
    "tnm_datetime_addtest": "99da426bc5c3d4e5ec548d0d655dee37c33b45ff560b8e93f970163f4a79a95f",
    "grade_datetime_addtest": "7fd5c63d53c7542d0f5481a8d8fa206ee4f4fa544cc57828ff73a0f375f391c0",
    "prescribed_med_begin_end_reason_continuity_ae": "01a1cdf5b8863f9b83d761b28b91e67780ed1245c8d8296efd3a18a93a55059e",
    "future_med_consideration_ae": "59b7058620ddb667879765eab5c36175b6c127034b27925c8ba83b0a2c203e0b",
}


class PaperProtocolTests(unittest.TestCase):
    def test_protocol_is_frozen_to_paper_commit(self):
        self.assertEqual(
            PAPER_SOURCE_COMMIT,
            "ddf1792ee5e1433be9ee4150d67d97b604a8daec",
        )
        self.assertEqual(
            hashlib.sha256(PAPER_PREAMBLE.encode()).hexdigest(),
            "53258763d20b6de42633a87d6c5a260184e2d06de44d3453f5f6010d0407cfde",
        )
        self.assertEqual(tuple(EXPECTED_PROMPT_HASHES), TASK_ORDER)
        self.assertEqual(
            {
                name: hashlib.sha256(prompt.encode()).hexdigest()
                for name, prompt in TASK_PROMPTS.items()
            },
            EXPECTED_PROMPT_HASHES,
        )

    def test_model_controls_are_fixed(self):
        self.assertEqual(MAX_OUTPUT_TOKENS, 4096)
        self.assertEqual(REASONING_EFFORT, "low")


class PaperGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_source(Path("data/coral_inference.csv"))

    def test_real_data_builds_complete_unique_grid(self):
        grid = build_request_grid(self.source)
        self.assertEqual(len(grid), 1120)
        self.assertEqual(grid.doc_idx.nunique(), 40)
        self.assertEqual(set(grid.section_name), {"hpi", "a&p"})
        self.assertEqual(tuple(grid.task.drop_duplicates()), TASK_ORDER)
        self.assertFalse(grid.duplicated(["doc_idx", "section_name", "task"]).any())

    def test_request_is_exact_section_text_plus_paper_prompt(self):
        grid = build_request_grid(self.source)
        row = grid.iloc[0]
        self.assertEqual(row.request_input, row.section_text + TASK_PROMPTS[row.task])
        self.assertEqual(row.instructions, PAPER_PREAMBLE)
        self.assertNotIn("evidence_quotes", row.request_input)

    def test_conflicting_section_text_is_rejected(self):
        bad = pd.concat([
            self.source,
            pd.DataFrame([{**self.source.iloc[0].to_dict(), "section_text": "conflict"}]),
        ], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "conflicting section text"):
            build_request_grid(bad)

    def test_incomplete_document_section_set_is_rejected(self):
        bad = self.source[self.source.doc_idx != self.source.doc_idx.iloc[0]]
        with self.assertRaisesRegex(ValueError, "40 documents"):
            build_request_grid(bad)

    def test_unknown_or_missing_task_definition_is_rejected(self):
        bad = self.source.copy()
        bad.loc[bad.index[0], "task"] = "not_a_paper_task"
        with self.assertRaisesRegex(ValueError, "task set"):
            build_request_grid(bad)
