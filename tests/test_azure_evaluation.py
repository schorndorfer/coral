import ast
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coral.utils.dataprocessing import parse_output
from coral.azure_evaluation import (
    AzureSettings,
    CheckpointRecord,
    Usage,
    append_checkpoint,
    build_request,
    can_afford,
    estimate_cost,
    load_azure_settings,
    run_evaluation,
    terminal_keys,
    to_legacy_output,
    validate_response,
    write_legacy_csv,
    write_observed_summary,
)


VALID_SYMPTOM_JSON = json.dumps(
    {
        "task": "symptoms",
        "records": [
            {
                "symptom": "low appetite",
                "datetimes": ["unknown"],
                "evidence_quotes": ["appetite is low"],
            }
        ],
    }
)

ONE_ROW = pd.DataFrame([{
    "doc_idx": "1",
    "section_name": "hpi",
    "task": "symptoms",
    "section_text": "The appetite is low.",
}])
VALID_RECORD = {
    "doc_idx": "1",
    "section_name": "hpi",
    "section_text": "The appetite is low.",
    "task": "symptoms",
    "validation_status": "valid",
    "parsed_records": [{
        "symptom": "low appetite",
        "datetimes": ["unknown"],
        "evidence_quotes": ["appetite is low"],
    }],
    "api_key": "must-not-be-exported",
}
SCORES_WITH_ONE_REAL_AND_ONE_SYNTHETIC_ROW = pd.DataFrame([
    {
        "doc_idx": "1", "section_name": "hpi", "task": "symptoms",
        "subrelation": "Symptom", "bleu4": 0.8, "rouge1": 0.7,
        "em_prec": 1.0, "em_recall": 0.5, "em_f1": 2 / 3,
    },
    {
        "doc_idx": "999", "section_name": "hpi", "task": "symptoms",
        "subrelation": "Symptom", "bleu4": 0.0, "rouge1": 0.0,
        "em_prec": 0.0, "em_recall": 0.0, "em_f1": 0.0,
    },
])
SETTINGS = AzureSettings("test-key", "https://example.openai.azure.com")


class FakeResponse:
    def __init__(self, output_text, input_tokens=0, output_tokens=0):
        self.output_text = output_text
        self.usage = type(
            "Usage", (), {"input_tokens": input_tokens, "output_tokens": output_tokens}
        )()


class FakeResponses:
    def __init__(self, responses, fail_if_called=False):
        self._responses = iter(responses)
        self._fail_if_called = fail_if_called
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self._fail_if_called:
            raise AssertionError("client must not be called")
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, output_text, input_tokens=0, output_tokens=0):
        self.responses = FakeResponses([FakeResponse(output_text, input_tokens, output_tokens)])

    @property
    def calls(self):
        return self.responses.calls

    @classmethod
    def sequence(cls, output_texts):
        client = cls.__new__(cls)
        client.responses = FakeResponses([
            text if isinstance(text, BaseException) else FakeResponse(text, 100, 20)
            for text in output_texts
        ])
        return client

    @classmethod
    def fail_if_called(cls):
        client = cls.__new__(cls)
        client.responses = FakeResponses([], fail_if_called=True)
        return client


class AzureEvaluationHelpersTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temporary_directory.name) / "checkpoints.jsonl"
        self.legacy_path = Path(self.temporary_directory.name) / "legacy.csv"
        self.summary_path = Path(self.temporary_directory.name) / "summary.csv"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_runner_checkpoints_valid_usage(self):
        """A valid model result is persisted with its real usage."""
        result = run_evaluation(
            ONE_ROW, FakeClient(VALID_SYMPTOM_JSON, 100, 20), SETTINGS,
            self.checkpoint, 1.0, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "valid")
        self.assertEqual(result.iloc[0].input_tokens, 100)
        checkpoint = json.loads(self.checkpoint.read_text())
        self.assertEqual(checkpoint["output_tokens"], 20)

    def test_runner_results_export_the_original_section_text(self):
        """Legacy exports retain the source section from an accepted runner result."""
        results = run_evaluation(
            ONE_ROW, FakeClient(VALID_SYMPTOM_JSON, 100, 20), SETTINGS,
            self.checkpoint, 1.0, 512, "low",
        )
        exported = write_legacy_csv(
            results.to_dict("records"), self.legacy_path, SETTINGS.deployment,
        )
        self.assertEqual(exported.iloc[0].section_text, "The appetite is low.")

    def test_write_legacy_csv_excludes_failed_records(self):
        """Only validated records become legacy scorer inputs."""
        frame = write_legacy_csv(
            [VALID_RECORD, {**VALID_RECORD, "validation_status": "validation_failed"}],
            self.legacy_path,
            "gpt-5.6-sol",
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].conversion_status, "validated")
        self.assertEqual(
            list(frame.columns),
            ["doc_idx", "section_name", "section_text", "task", "model", "output", "conversion_status"],
        )
        self.assertNotIn("must-not-be-exported", self.legacy_path.read_text())

    def test_write_legacy_csv_requires_section_text(self):
        """Accepted records without source text cannot become scorer inputs."""
        with self.assertRaisesRegex(ValueError, "section_text"):
            write_legacy_csv(
                [{key: value for key, value in VALID_RECORD.items() if key != "section_text"}],
                self.legacy_path,
                "gpt-5.6-sol",
            )

    def test_observed_summary_excludes_synthetic_cartesian_rows(self):
        """Summary means and count reflect only rows that were prompted."""
        summary = write_observed_summary(
            SCORES_WITH_ONE_REAL_AND_ONE_SYNTHETIC_ROW, ONE_ROW, self.summary_path,
        )
        self.assertEqual(summary.iloc[0].n_examples, 1)
        self.assertEqual(summary.iloc[0].mean_bleu4, 0.8)
        self.assertEqual(summary.iloc[0].mean_em_f1, 2 / 3)

    def test_observed_summary_uses_only_validated_export_keys(self):
        """A failed prompt cannot admit a synthetic legacy scorer row."""
        failed_record = {
            **VALID_RECORD,
            "doc_idx": "999",
            "validation_status": "validation_failed",
        }
        legacy_frame = write_legacy_csv(
            [VALID_RECORD, failed_record], self.legacy_path, SETTINGS.deployment,
        )
        summary = write_observed_summary(
            SCORES_WITH_ONE_REAL_AND_ONE_SYNTHETIC_ROW,
            legacy_frame.loc[:, ["doc_idx", "section_name", "task"]],
            self.summary_path,
        )
        self.assertEqual(summary.iloc[0].n_examples, 1)
        self.assertEqual(summary.iloc[0].mean_bleu4, 0.8)

    def test_notebook_checkpoint_reader_ignores_a_truncated_jsonl_tail(self):
        """An interrupted write does not block export of prior complete records."""
        notebook_tree = ast.parse(Path("notebooks/evaluate_gpt56_sol_azure.py").read_text())
        reader_node = next(
            node
            for node in ast.walk(notebook_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "read_checkpoint_records"
        )
        namespace = {"Path": Path, "json": json}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[reader_node], type_ignores=[])), "notebook", "exec"),
            namespace,
        )
        checkpoint_path = Path(self.temporary_directory.name) / "truncated.jsonl"
        checkpoint_path.write_text('{"doc_idx":"1"}\n{"doc_idx":')
        self.assertEqual(namespace["read_checkpoint_records"](checkpoint_path), [{"doc_idx": "1"}])

    def test_runner_retries_once_then_marks_valid_after_retry(self):
        """A malformed first result gets one corrective retry."""
        client = FakeClient.sequence(["not-json", VALID_SYMPTOM_JSON])
        result = run_evaluation(
            ONE_ROW, client, SETTINGS, self.checkpoint, 1.0, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "valid_after_retry")
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(result.iloc[0].raw_attempts), 2)

    def test_runner_reports_start_and_terminal_progress_for_each_row(self):
        messages: list[str] = []
        run_evaluation(
            ONE_ROW,
            FakeClient(VALID_SYMPTOM_JSON, 100, 20),
            SETTINGS,
            self.checkpoint,
            1.0,
            512,
            "low",
            progress=messages.append,
        )
        self.assertEqual(messages[0], "Starting 1/1: doc 1, hpi, symptoms")
        self.assertEqual(messages[-1], "Finished 1/1: valid")

    def test_runner_respects_cap_before_making_request(self):
        """A zero cap prevents a network request and is checkpointed."""
        result = run_evaluation(
            ONE_ROW, FakeClient.fail_if_called(), SETTINGS,
            self.checkpoint, 0.0, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "spend_cap_reached")

    def test_runner_cap_uses_full_initial_prompt_not_section_alone(self):
        """Prompt/schema overhead blocks a request that section-only math admits."""
        result = run_evaluation(
            ONE_ROW, FakeClient.fail_if_called(), SETTINGS,
            self.checkpoint, 0.0105, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "spend_cap_reached")

    def test_runner_cap_uses_full_corrective_prompt_before_retry(self):
        """Retry-only corrective text counts toward the next request's cap."""
        client = FakeClient.sequence(["not-json", VALID_SYMPTOM_JSON])
        result = run_evaluation(
            ONE_ROW, client, SETTINGS, self.checkpoint, 0.012, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "spend_cap_reached")
        self.assertEqual(client.calls, 1)

    def test_runner_respects_cap_before_retrying_invalid_result(self):
        """A retry is not sent when its worst-case cost would exceed the cap."""
        client = FakeClient.sequence(["not-json", VALID_SYMPTOM_JSON])
        result = run_evaluation(
            ONE_ROW, client, SETTINGS, self.checkpoint, 0.012, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "spend_cap_reached")
        self.assertEqual(client.calls, 1)

    def test_runner_skips_terminal_checkpoint_without_calling_client(self):
        """A terminal checkpoint identifies a completed row during resume."""
        append_checkpoint(self.checkpoint, CheckpointRecord(
            "1", "hpi", "symptoms", SETTINGS.deployment, "valid",
        ))
        result = run_evaluation(
            ONE_ROW, FakeClient.fail_if_called(), SETTINGS,
            self.checkpoint, 1.0, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "skipped_on_resume")

    def test_runner_skips_duplicate_key_in_same_run(self):
        """A repeated identity is skipped after its first terminal checkpoint."""
        client = FakeClient(VALID_SYMPTOM_JSON, 100, 20)
        result = run_evaluation(
            pd.concat([ONE_ROW, ONE_ROW], ignore_index=True),
            client, SETTINGS,
            self.checkpoint, 1.0, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "valid")
        self.assertEqual(result.iloc[1].validation_status, "skipped_on_resume")
        self.assertEqual(client.calls, 1)

    def test_runner_records_api_failure_without_exception_details(self):
        """A client exception produces a safe terminal checkpoint row."""
        client = FakeClient.sequence([RuntimeError("connection unavailable")])
        result = run_evaluation(
            ONE_ROW, client, SETTINGS, self.checkpoint, 1.0, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "api_failed")
        self.assertEqual(result.iloc[0].error, "RuntimeError: connection unavailable")

    def test_runner_treats_client_value_error_as_api_failure(self):
        """A ValueError raised by the injected client is not a validation retry."""
        result = run_evaluation(
            ONE_ROW, FakeClient.sequence([ValueError("request rejected")]), SETTINGS,
            self.checkpoint, 1.0, 512, "low",
        )
        self.assertEqual(result.iloc[0].validation_status, "api_failed")
        self.assertEqual(result.iloc[0].error, "ValueError: request rejected")

    def test_validate_response_requires_verbatim_evidence(self):
        raw = json.dumps({"task": "symptoms", "records": [{
            "symptom": "low appetite", "datetimes": ["unknown"],
            "evidence_quotes": ["appetite is low"],
        }]})
        self.assertEqual(
            validate_response("symptoms", "The appetite is low.", raw)[0]["symptom"],
            "low appetite",
        )

    def test_validate_response_rejects_nonverbatim_evidence(self):
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_response("symptoms", "No appetite statement.", VALID_SYMPTOM_JSON)

    def test_validate_response_rejects_empty_evidence_quote(self):
        raw = json.dumps({"task": "symptoms", "records": [{
            "symptom": "low appetite",
            "datetimes": ["unknown"],
            "evidence_quotes": [""],
        }]})
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_response("symptoms", "The appetite is low.", raw)

    def test_request_schema_requires_nonempty_evidence_quotes(self):
        _, response_format = build_request({"task": "symptoms"})
        schema = response_format["schema"]
        evidence_items = schema["properties"]["records"]["items"]["properties"]["evidence_quotes"]["items"]
        self.assertEqual(evidence_items["minLength"], 1)

    def test_to_legacy_output_serializes_symptom_record(self):
        self.assertEqual(
            to_legacy_output(
                "symptoms",
                [{
                    "symptom": "low appetite",
                    "datetimes": ["unknown"],
                    "evidence_quotes": ["appetite is low"],
                }],
            ),
            "SymptomEnt(Symptom='low appetite', Datetime={'unknown'})",
        )

    def test_to_legacy_output_round_trips_empty_set_like_field(self):
        output = to_legacy_output(
            "symptoms",
            [{
                "symptom": "low appetite",
                "datetimes": [],
                "evidence_quotes": ["appetite is low"],
            }],
        )
        parsed, errors = parse_output(output, "symptoms")
        self.assertEqual(errors, 0)
        self.assertEqual(parsed[0].Datetime, set())

    def test_load_azure_settings_uses_only_azure_environment_names(self):
        settings = load_azure_settings(
            {
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            }
        )
        self.assertEqual(settings.deployment, "gpt-5.6-sol")
        self.assertEqual(settings.endpoint, "https://example.openai.azure.com")

    def test_load_azure_settings_normalizes_a_v1_api_base_to_resource_root(self):
        settings = load_azure_settings(
            {
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/openai/v1",
            }
        )
        self.assertEqual(settings.endpoint, "https://example.openai.azure.com")

    def test_load_azure_settings_preserves_foundry_v1_api_base(self):
        settings = load_azure_settings(
            {
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com/openai/v1",
            }
        )
        self.assertEqual(
            settings.endpoint,
            "https://example.services.ai.azure.com/openai/v1",
        )

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

    def test_notebook_declares_azure_dependencies_and_full_run_guard(self):
        notebook = Path("notebooks/evaluate_gpt56_sol_azure.py").read_text()
        self.assertIn('"marimo"', notebook)
        self.assertIn('"openai"', notebook)
        self.assertIn('"torch"', notebook)
        self.assertIn("AZURE_OPENAI_API_KEY", notebook)
        self.assertIn("AZURE_OPENAI_DEPLOYMENT", notebook)
        self.assertIn("AZURE_OPENAI_API_VERSION", notebook)
        self.assertIn("Run full 515-input evaluation", notebook)
        self.assertIn("projection_display = mo.vstack", notebook)
        self.assertIn("exported_keys = legacy_frame.loc[:, [\"doc_idx\", \"section_name\", \"task\"]]", notebook)
        self.assertIn("except json.JSONDecodeError:", notebook)
        self.assertIn("from openai import AzureOpenAI, OpenAI", notebook)
        self.assertIn("base_url=settings.endpoint", notebook)
        self.assertIn("progress=print if is_script_mode else None", notebook)
        self.assertNotIn("OPENAI_API_KEY", notebook.replace("AZURE_OPENAI_API_KEY", ""))


if __name__ == "__main__":
    unittest.main()
