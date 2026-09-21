import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd

from coral.azure_evaluation import AzureSettings
from coral.paper_replication.protocol import MAX_OUTPUT_TOKENS, PAPER_PREAMBLE
from coral.paper_replication.runner import (
    PaperCheckpointRecord,
    checkpoint_spend,
    completed_keys,
    create_azure_client,
    load_azure_settings,
    project_grid_cost,
    read_checkpoint,
    run_replication,
    smoke_is_complete,
)


class FakeResponse:
    def __init__(self, text, input_tokens=100, output_tokens=20):
        self.output_text = text
        self.usage = type("Usage", (), {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })()


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


class RecordingClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = FakeResponses([])


class RateLimitError(Exception):
    status_code = 429


SETTINGS = AzureSettings("secret", "https://example.openai.azure.com")
GRID = pd.DataFrame([{
    "doc_idx": "1", "section_name": "hpi", "task": "symptoms",
    "section_text": "Fatigue today.",
    "task_prompt": "\nPaper task prompt",
    "instructions": PAPER_PREAMBLE,
    "request_input": "Fatigue today.\nPaper task prompt",
}])
COMPLETED = {
    "doc_idx": "1", "section_name": "hpi", "task": "symptoms",
    "model": "gpt-5.6-sol", "status": "completed", "cost": 0.0008,
}


class PaperRunnerSettingsTests(unittest.TestCase):
    def test_foundry_v1_endpoint_is_preserved_as_openai_base_url(self):
        settings = AzureSettings(
            "secret",
            "https://wkt406-codehelp-resource.services.ai.azure.com/openai/v1",
            "gpt-5.6-sol",
        )
        client = create_azure_client(
            settings, openai_class=RecordingClient, azure_class=self.fail_constructor,
        )
        self.assertEqual(client.kwargs, {
            "base_url": settings.endpoint, "api_key": "secret",
        })

    def test_standard_azure_endpoint_uses_azure_client(self):
        client = create_azure_client(
            SETTINGS, openai_class=self.fail_constructor, azure_class=RecordingClient,
        )
        self.assertEqual(client.kwargs, {
            "azure_endpoint": SETTINGS.endpoint, "api_key": "secret",
            "api_version": "2025-04-01-preview",
        })

    def test_endpoint_selection_uses_hostname_not_url_suffix(self):
        for endpoint in (
            "https://EXAMPLE.services.ai.azure.com:443/openai/v1/",
            "https://example.services.ai.azure.com/openai/v1?query=keep",
        ):
            with self.subTest(endpoint=endpoint):
                client = create_azure_client(
                    AzureSettings("secret", endpoint), openai_class=RecordingClient,
                    azure_class=self.fail_constructor,
                )
                self.assertEqual(client.kwargs["base_url"], endpoint)
        client = create_azure_client(
            AzureSettings("secret", "https://example.com/path.services.ai.azure.com"),
            openai_class=self.fail_constructor, azure_class=RecordingClient,
        )
        self.assertIn("azure_endpoint", client.kwargs)

    def test_sdk_retries_are_disabled_so_runner_owns_attempt_limit(self):
        class OptionsClient(RecordingClient):
            def with_options(self, **kwargs):
                configured = RecordingClient()
                configured.options = kwargs
                return configured

        client = create_azure_client(SETTINGS, azure_class=OptionsClient)
        self.assertEqual(client.options, {"max_retries": 0})

    def test_public_settings_loader_preserves_foundry_v1_path(self):
        settings = load_azure_settings({
            "AZURE_OPENAI_API_KEY": "secret",
            "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com/openai/v1",
        })
        self.assertEqual(settings.endpoint, "https://example.services.ai.azure.com/openai/v1")
        self.assertEqual(settings.deployment, "gpt-5.6-sol")
        self.assertNotIn("secret", repr(settings))

    @staticmethod
    def fail_constructor(**kwargs):
        raise AssertionError(f"wrong client selected: {sorted(kwargs)}")


class PaperRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "checkpoint.jsonl"

    def run_grid(self, client, grid=GRID, spend_cap=1.0, **kwargs):
        return run_replication(
            grid, client, SETTINGS, self.path, spend_cap,
            sleep=kwargs.pop("sleep", lambda _: None), **kwargs,
        )

    def write_records(self, records, tail=""):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(record) + "\n" for record in records) + tail,
            encoding="utf-8",
        )

    def test_sends_exact_paper_request_and_fixed_model_controls(self):
        client = FakeClient([FakeResponse("not parseable by design")])
        result = self.run_grid(client)
        self.assertEqual(client.responses.calls, [{
            "model": "gpt-5.6-sol", "instructions": PAPER_PREAMBLE,
            "input": "Fatigue today.\nPaper task prompt",
            "reasoning": {"effort": "low"}, "max_output_tokens": MAX_OUTPUT_TOKENS,
        }])
        self.assertEqual(result.iloc[0].status, "completed")

    def test_malformed_text_is_completed_without_corrective_retry(self):
        client = FakeClient([FakeResponse("malformed")])
        result = self.run_grid(client)
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(result.iloc[0].output_text, "malformed")
        self.assertEqual(result.iloc[0].status, "completed")

    def test_transient_retry_reuses_identical_arguments(self):
        client = FakeClient([RateLimitError("slow down"), FakeResponse("N/A")])
        delays = []
        result = self.run_grid(client, sleep=delays.append)
        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(client.responses.calls[0], client.responses.calls[1])
        self.assertEqual(delays, [1.0])
        self.assertEqual(result.iloc[0].status, "completed")
        self.assertEqual(len(read_checkpoint(self.path)), 1)

    def test_transient_statuses_stop_after_three_attempts(self):
        for status in (408, 409, 429, 500, 503, 599):
            with self.subTest(status=status):
                error = type("TransientError", (Exception,), {"status_code": status})
                client = FakeClient([error("retry")] * 3)
                delays = []
                result = self.run_grid(client, sleep=delays.append)
                self.assertEqual(result.iloc[0].status, "api_failed")
                self.assertEqual(len(client.responses.calls), 3)
                self.assertEqual(delays, [1.0, 2.0])
                self.assertTrue(all(call == client.responses.calls[0] for call in client.responses.calls))

    def test_nontransient_errors_fail_without_retry(self):
        for status in (400, 401, 404, 499, 600, None):
            with self.subTest(status=status):
                error = type("PermanentError", (Exception,), {"status_code": status})
                client = FakeClient([error("bad request")])
                delays = []
                result = self.run_grid(client, sleep=delays.append)
                self.assertEqual(result.iloc[0].status, "api_failed")
                self.assertEqual(len(client.responses.calls), 1)
                self.assertEqual(delays, [])

    def test_cap_is_checked_before_client_call_and_is_retryable(self):
        client = FakeClient([])
        result = self.run_grid(client, spend_cap=0.0)
        self.assertEqual(result.iloc[0].status, "spend_cap_reached")
        self.assertEqual(client.responses.calls, [])
        resumed = self.run_grid(FakeClient([FakeResponse("N/A")]))
        self.assertEqual(resumed.iloc[0].status, "completed")

    def test_actual_spend_stops_grid_before_next_request(self):
        grid = pd.concat([GRID, GRID.assign(doc_idx="2"), GRID.assign(doc_idx="3")])
        client = FakeClient([FakeResponse("N/A", input_tokens=1000, output_tokens=50)])
        # First request costs $0.005; the second projection exceeds the remainder.
        result = self.run_grid(client, grid=grid, spend_cap=0.085)
        self.assertEqual(result.status.tolist(), ["completed", "spend_cap_reached"])
        self.assertEqual(len(client.responses.calls), 1)
        self.assertAlmostEqual(checkpoint_spend(read_checkpoint(self.path)), 0.005)

    def test_checkpoint_spend_counts_other_models_and_nonterminal_rows(self):
        self.write_records([
            {**COMPLETED, "model": "other", "cost": 0.5},
            {**COMPLETED, "status": "api_failed", "cost": 0.45},
        ])
        client = FakeClient([])
        result = self.run_grid(client)
        self.assertEqual(result.iloc[0].status, "spend_cap_reached")
        self.assertEqual(client.responses.calls, [])

    def test_cap_equal_to_projection_allows_request(self):
        # 85 preamble characters + 32 input characters => 30 + 256 input tokens.
        result = self.run_grid(FakeClient([FakeResponse("N/A")]), spend_cap=0.083064)
        self.assertEqual(result.iloc[0].status, "completed")

    def test_invalid_spend_cap_fails_before_checkpoint_or_client(self):
        for cap in (-1.0, float("nan"), float("inf")):
            with self.subTest(cap=cap):
                client = FakeClient([])
                with self.assertRaises(ValueError):
                    self.run_grid(client, spend_cap=cap)
                self.assertEqual(client.responses.calls, [])
                self.assertFalse(self.path.exists())

    def test_api_failure_is_recorded_but_retryable_on_resume(self):
        first = self.run_grid(FakeClient([ValueError("bad request")]))
        second = self.run_grid(FakeClient([FakeResponse("N/A")]))
        self.assertEqual(first.iloc[0].status, "api_failed")
        self.assertEqual(second.iloc[0].status, "completed")
        self.assertEqual(len(read_checkpoint(self.path)), 2)

    def test_completed_request_is_skipped_without_appending(self):
        self.run_grid(FakeClient([FakeResponse("N/A")]))
        before = self.path.read_bytes()
        client = FakeClient([])
        resumed = self.run_grid(client)
        self.assertEqual(resumed.iloc[0].status, "skipped_on_resume")
        self.assertEqual(client.responses.calls, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_response_text_or_usage_never_completes(self):
        for response in (object(), FakeResponse(None), FakeResponse("N/A", input_tokens=None)):
            with self.subTest(response=type(response).__name__):
                client = FakeClient([response])
                result = self.run_grid(client)
                self.assertEqual(result.iloc[0].status, "api_failed")
                self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(completed_keys(read_checkpoint(self.path)), set())

    def test_completed_checkpoint_has_only_identity_result_and_accounting(self):
        self.run_grid(FakeClient([FakeResponse("N/A")]))
        record, = read_checkpoint(self.path)
        self.assertEqual(set(record), {
            "doc_idx", "section_name", "task", "model", "status", "output_text",
            "input_tokens", "output_tokens", "cost", "elapsed_seconds", "error",
        })
        self.assertEqual(record["input_tokens"], 100)
        self.assertEqual(record["output_tokens"], 20)
        self.assertAlmostEqual(record["cost"], 0.0008)
        self.assertGreaterEqual(record["elapsed_seconds"], 0.0)
        self.assertEqual(record["output_text"], "N/A")
        self.assertIsNone(record["error"])
        frozen = PaperCheckpointRecord(**record)
        with self.assertRaises(FrozenInstanceError):
            frozen.status = "api_failed"

    def test_truncated_checkpoint_tail_is_ignored_and_resume_stays_readable(self):
        self.write_records([COMPLETED], tail='{"status":')
        self.assertEqual(read_checkpoint(self.path), [COMPLETED])
        self.run_grid(FakeClient([FakeResponse("N/A")]), grid=GRID.assign(doc_idx="2"))
        records = read_checkpoint(self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["doc_idx"], "2")
        client = FakeClient([])
        result = self.run_grid(client, grid=GRID.assign(doc_idx="2"))
        self.assertEqual(result.iloc[0].status, "skipped_on_resume")
        self.assertEqual(client.responses.calls, [])

    def test_valid_final_line_without_newline_is_preserved_when_appending(self):
        self.write_records([], tail=json.dumps(COMPLETED))
        self.run_grid(FakeClient([FakeResponse("N/A")]), grid=GRID.assign(doc_idx="2"))
        self.assertEqual(len(read_checkpoint(self.path)), 2)

    def test_checkpoint_and_settings_never_render_api_key(self):
        self.run_grid(FakeClient([ValueError("secret was rejected " + "x" * 600)]))
        self.assertNotIn("secret", repr(SETTINGS))
        self.assertNotIn("secret", self.path.read_text(encoding="utf-8"))
        record, = read_checkpoint(self.path)
        self.assertTrue(record["error"].startswith("ValueError: [REDACTED] was rejected"))
        self.assertLessEqual(len(record["error"]), 500)

    def test_progress_reports_identity_status_spend_path_without_payload(self):
        messages = []
        self.run_grid(FakeClient([ValueError("secret was rejected")]), progress=messages.append)
        self.assertEqual(len(messages), 1)
        message, = messages
        for expected in ("1/1", "hpi", "symptoms", "api_failed", "0.000000", str(self.path)):
            self.assertIn(expected, message)
        for excluded in ("secret", "Fatigue", "Paper task prompt", "ValueError", PAPER_PREAMBLE):
            self.assertNotIn(excluded, message)

    def test_empty_grid_sends_nothing_and_creates_no_checkpoint(self):
        client = FakeClient([])
        result = self.run_grid(client, grid=GRID.iloc[:0])
        self.assertTrue(result.empty)
        self.assertEqual(client.responses.calls, [])
        self.assertFalse(self.path.exists())


class PaperCheckpointHelpersTests(unittest.TestCase):
    def test_cost_projection_uses_every_request_and_fixed_ceiling(self):
        one = project_grid_cost(GRID)
        self.assertAlmostEqual(one, 0.083064)
        self.assertAlmostEqual(project_grid_cost(pd.concat([GRID, GRID])), 0.166128)
        self.assertEqual(project_grid_cost(GRID.iloc[:0]), 0.0)

    def test_cost_projection_includes_instructions_and_rounds_up_input_tokens(self):
        grid = GRID.assign(instructions="ab", request_input="cde")
        # Five characters round up to two tokens, plus 256 envelope tokens.
        self.assertAlmostEqual(project_grid_cost(grid), 0.082952)

    def test_completed_keys_require_completed_status_and_full_identity(self):
        records = [
            COMPLETED,
            {**COMPLETED, "status": "api_failed", "doc_idx": "2"},
            {**COMPLETED, "status": "spend_cap_reached", "doc_idx": "3"},
            {**COMPLETED, "status": "skipped_on_resume", "doc_idx": "4"},
            {"doc_idx": "5", "status": "completed"},
        ]
        self.assertEqual(completed_keys(iter(records)), {("1", "hpi", "symptoms", "gpt-5.6-sol")})

    def test_checkpoint_spend_sums_valid_numeric_costs_without_deduplication(self):
        records = [
            {"cost": 0.25}, {"cost": 0.25}, {"cost": 1},
            {"cost": None}, {"cost": "1"}, {"cost": -1}, {"cost": True},
            {"cost": float("nan")}, {"cost": float("inf")}, {},
        ]
        self.assertEqual(checkpoint_spend(iter(records)), 1.5)

    def test_smoke_gate_requires_all_first_eight_completed_keys(self):
        grid = pd.concat([GRID.assign(doc_idx=str(index)) for index in range(9)], ignore_index=True)
        records = [{**COMPLETED, "doc_idx": str(index)} for index in range(8)]
        self.assertTrue(smoke_is_complete(grid, iter(records), "gpt-5.6-sol"))
        self.assertFalse(smoke_is_complete(grid, records[:-1], "gpt-5.6-sol"))
        self.assertFalse(smoke_is_complete(grid, records, "other-model"))
        for status in ("api_failed", "spend_cap_reached", "skipped_on_resume"):
            with self.subTest(status=status):
                self.assertFalse(smoke_is_complete(
                    grid, [*records[:-1], {**records[-1], "status": status}], "gpt-5.6-sol",
                ))

    def test_missing_checkpoint_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(read_checkpoint(Path(directory) / "absent.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
