import json
import tempfile
import unittest
from pathlib import Path

from coral.azure_evaluation import (
    CheckpointRecord,
    Usage,
    append_checkpoint,
    can_afford,
    estimate_cost,
    load_azure_settings,
    terminal_keys,
)


class AzureEvaluationHelpersTests(unittest.TestCase):
    def test_load_azure_settings_uses_only_azure_environment_names(self):
        settings = load_azure_settings(
            {
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            }
        )
        self.assertEqual(settings.deployment, "gpt-5.6-sol")
        self.assertEqual(settings.endpoint, "https://example.openai.azure.com")

    def test_load_azure_settings_rejects_missing_key(self):
        with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_API_KEY"):
            load_azure_settings({"AZURE_OPENAI_ENDPOINT": "https://example"})

    def test_estimate_cost_uses_sol_rates(self):
        self.assertEqual(estimate_cost(Usage(500_000, 100_000)), 4.0)

    def test_settings_repr_redacts_api_key(self):
        settings = load_azure_settings(
            {"AZURE_OPENAI_API_KEY": "secret", "AZURE_OPENAI_ENDPOINT": "https://example"}
        )
        self.assertNotIn("secret", repr(settings))

    def test_can_afford_includes_projected_cost_and_rejects_negative_cap(self):
        self.assertTrue(can_afford(1.0, 2.0, 3.0))
        self.assertFalse(can_afford(1.0, 2.01, 3.0))
        with self.assertRaises(ValueError):
            can_afford(0.0, 0.0, -1.0)

    def test_terminal_keys_skips_malformed_and_nonterminal_jsonl_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.jsonl"
            path.write_text(
                '{"doc_idx":"1","section_name":"hpi","task":"symptoms","model":"gpt-5.6-sol","validation_status":"valid"}\n'
                "not json\n"
                '{"validation_status":"running"}\n'
            )
            self.assertEqual(
                terminal_keys(path), {("1", "hpi", "symptoms", "gpt-5.6-sol")}
            )

    def test_append_checkpoint_creates_parent_and_writes_one_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoints.jsonl"
            record = CheckpointRecord(
                doc_idx="1",
                section_name="hpi",
                task="symptoms",
                model="gpt-5.6-sol",
                validation_status="valid",
            )
            append_checkpoint(path, record)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertEqual(json.loads(path.read_text())["doc_idx"], "1")


if __name__ == "__main__":
    unittest.main()
