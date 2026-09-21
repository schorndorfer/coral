import contextlib
import functools
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from coral.paper_replication import build_request_grid, load_source, score_completed_records
from tests.test_paper_replication_runner import FakeClient, FakeResponse
from tests.test_paper_replication_scoring import FakeMetrics


NOTEBOOK = Path("notebooks/evaluate_gpt56_sol_paper_replication.py")
PREFIX = "gpt56_sol_paper_replication"
ENVIRONMENT = {
    "AZURE_OPENAI_API_KEY": "test-secret-never-display",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
}


class PaperReplicationNotebookTests(unittest.TestCase):
    def test_declares_required_runtime_and_azure_configuration(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        for dependency in ('"marimo"', '"openai"', '"pandas"', '"evaluate"', '"torch"'):
            self.assertIn(dependency, text)
        self.assertIn("AZURE_OPENAI_API_KEY", text)
        self.assertIn("AZURE_OPENAI_ENDPOINT", text)
        self.assertIn("AZURE_OPENAI_DEPLOYMENT", text)
        self.assertNotIn("OPENAI_API_KEY", text.replace("AZURE_OPENAI_API_KEY", ""))

    def test_full_run_has_exact_confirmation_and_isolated_prefix(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertIn("RUN 1120", text)
        self.assertIn("CORAL_PAPER_REPLICATION_CONFIRM", text)
        self.assertIn(PREFIX, text)
        self.assertIn("smoke_is_complete", text)
        self.assertNotIn('"output/gpt56_sol_azure', text)

    def test_protocol_controls_are_not_user_adjustable(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("Reasoning effort", text)
        self.assertNotIn("Max output tokens", text)
        self.assertIn("4,096", text)
        self.assertIn("low reasoning effort", text)

    def test_notebook_does_not_render_individual_outputs(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("run_results.to_string", text)
        self.assertNotIn("mo.ui.table(run_results", text)
        self.assertNotIn("coral.benchmarking.evaluate_model", text)
        self.assertNotIn("coral.azure_evaluation", text)
        self.assertIn("topline", text)


class PaperReplicationNotebookExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path("data/coral_inference.csv").resolve()
        cls.grid = build_request_grid(load_source(cls.source_path))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output_path = Path(self.directory.name)
        self.checkpoint_path = self.output_path / f"{PREFIX}.jsonl"
        self.topline_path = self.output_path / f"{PREFIX}_topline.csv"
        self.smoke_marker_path = self.output_path / f"{PREFIX}_smoke.json"

    def smoke_records(self):
        return [
            {
                "doc_idx": str(row.doc_idx), "section_name": row.section_name,
                "task": row.task, "model": "gpt-5.6-sol", "status": "completed",
                "output_text": "test-output-never-display", "cost": 0.0,
            }
            for row in self.grid.head(8).itertuples(index=False)
        ]

    def write_checkpoint(self, records):
        self.checkpoint_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def run_notebook(
        self, env=None, client=None, allow_scoring=False, ui=None,
        score_error=None, artifact_error=None,
    ):
        notebook_app = runpy.run_path(str(NOTEBOOK.resolve()))["app"]
        output = io.StringIO()
        offline_score = functools.partial(score_completed_records, metrics=FakeMetrics())
        injected_definitions = {
            "data_path": self.source_path,
            "output_path": self.output_path,
            "checkpoint_path": self.checkpoint_path,
            "topline_path": self.topline_path,
            "smoke_marker_path": self.smoke_marker_path,
        }
        if ui is not None:
            injected_definitions.update({
                "full_button": SimpleNamespace(value=ui.get("full", False)),
                "full_confirmation": SimpleNamespace(value=ui.get("confirmation", "")),
                "smoke_button": SimpleNamespace(value=ui.get("smoke", False)),
                "spend_cap": SimpleNamespace(value=ui.get("cap", 100)),
            })
        with (
            patch.dict(os.environ, ENVIRONMENT | (env or {}), clear=True),
            patch("coral.paper_replication.create_azure_client", side_effect=(
                (lambda settings: client) if client is not None
                else AssertionError("Unauthorized client construction")
            )),
            patch("coral.paper_replication.score_completed_records", side_effect=(
                score_error if score_error is not None else offline_score if allow_scoring
                else AssertionError("Inert/rejected action must not load metrics")
            )) as score_call,
            patch("coral.paper_replication.write_artifacts", side_effect=artifact_error)
            if artifact_error is not None else contextlib.nullcontext(),
            patch("marimo.app_meta", return_value=SimpleNamespace(mode="edit"))
            if ui is not None else contextlib.nullcontext(),
            contextlib.redirect_stdout(output),
        ):
            rendered, definitions = notebook_app.run(defs=injected_definitions)
        if not allow_scoring and score_error is None:
            score_call.assert_not_called()
        visible = output.getvalue() + "\n".join(
            getattr(item, "text", str(item)) for item in rendered
        )
        self.assertNotIn("test-secret-never-display", visible)
        self.assertNotIn("test-output-never-display", visible)
        self.assertNotIn(str(self.grid.iloc[0].section_text), visible)
        return output.getvalue(), rendered, definitions

    def test_default_script_is_inert_even_with_azure_credentials(self):
        output, _, _ = self.run_notebook()
        self.assertIn("inert", output.lower())
        self.assertEqual(list(self.output_path.iterdir()), [])

    def test_inert_with_checkpoint_does_not_rescore_or_rewrite(self):
        self.write_checkpoint(self.smoke_records())
        before = self.checkpoint_path.read_bytes()
        self.run_notebook()
        self.assertEqual(self.checkpoint_path.read_bytes(), before)
        self.assertFalse(self.topline_path.exists())

    def test_unknown_script_action_is_inert(self):
        output, _, _ = self.run_notebook({"CORAL_PAPER_REPLICATION_RUN": "1"})
        self.assertIn("inert", output.lower())
        self.assertFalse(self.checkpoint_path.exists())

    def test_interactive_open_ignores_script_run_environment(self):
        output, _, _ = self.run_notebook({"CORAL_PAPER_REPLICATION_RUN": "smoke"}, ui={})
        self.assertNotIn("Starting", output)
        self.assertEqual(list(self.output_path.iterdir()), [])

    def test_interactive_full_rechecks_confirmation_and_smoke(self):
        for confirmation in ("RUN 1120 ", "RUN 1120"):
            with self.subTest(confirmation=confirmation):
                _, rendered, _ = self.run_notebook(ui={"full": True, "confirmation": confirmation})
                self.assertIn("Rejected", "\n".join(
                    getattr(item, "text", str(item)) for item in rendered
                ))
        self.assertEqual(list(self.output_path.iterdir()), [])

    def test_interactive_smoke_honors_zero_cap_and_sends_no_requests(self):
        client = FakeClient([])
        output, _, _ = self.run_notebook(client=client, ui={"smoke": True, "cap": 0})
        self.assertIn("spend_cap_reached", output)
        self.assertEqual(client.responses.calls, [])
        record = json.loads(self.checkpoint_path.read_text())
        self.assertEqual(record["status"], "spend_cap_reached")
        self.assertFalse(self.topline_path.exists())

    def test_inert_displays_saved_topline_without_loading_metrics(self):
        pd.DataFrame([{
            "metric": "BLEU-4", "gpt56_sol": 0.5, "paper_gpt4": 0.73,
            "difference_vs_paper": -0.23,
        }]).to_csv(self.topline_path, index=False)
        before = self.topline_path.read_bytes()
        _, rendered, _ = self.run_notebook()
        tables = [item for item in rendered if type(item).__name__ == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].data.iloc[0].gpt56_sol, 0.5)
        self.assertEqual(self.topline_path.read_bytes(), before)
        self.assertFalse(self.checkpoint_path.exists())

    def test_full_requires_exact_confirmation_even_with_completed_smoke(self):
        self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"},
            FakeClient([FakeResponse("test-output-never-display") for _ in range(8)]),
            allow_scoring=True,
        )
        self.assertTrue(self.smoke_marker_path.exists())
        before = self.topline_path.read_bytes()
        for confirmation in ("", "run 1120", "RUN 1120 "):
            with self.subTest(confirmation=confirmation):
                output, _, _ = self.run_notebook({
                    "CORAL_PAPER_REPLICATION_RUN": "full",
                    "CORAL_PAPER_REPLICATION_CONFIRM": confirmation,
                })
                self.assertIn("Rejected", output)
                self.assertIn("RUN 1120", output)
        self.assertEqual(self.topline_path.read_bytes(), before)

    def test_full_requires_every_isolated_smoke_key_for_current_deployment(self):
        records = self.smoke_records()
        variants = [
            [], records[:7],
            [{**record, "model": "other-deployment"} for record in records],
            [{**record, "status": "api_failed"} for record in records],
            [records[0]] * 8,
        ]
        for variant in variants:
            with self.subTest(records=variant):
                self.write_checkpoint(variant)
                output, _, _ = self.run_notebook({
                    "CORAL_PAPER_REPLICATION_RUN": "full",
                    "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
                })
                self.assertIn("Rejected", output)
                self.assertIn("eight", output)
        self.assertFalse(self.topline_path.exists())

    def test_smoke_missing_configuration_is_rejected_without_client(self):
        output, _, _ = self.run_notebook({
            "CORAL_PAPER_REPLICATION_RUN": "smoke", "AZURE_OPENAI_API_KEY": "",
        })
        self.assertIn("Rejected", output)
        self.assertFalse(self.checkpoint_path.exists())

    def test_smoke_runs_eight_requests_and_renders_only_topline(self):
        client = FakeClient([FakeResponse("test-output-never-display") for _ in range(8)])
        output, rendered, _ = self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"}, client, allow_scoring=True,
        )
        self.assertEqual(len(client.responses.calls), 8)
        self.assertEqual(len(self.checkpoint_path.read_text().splitlines()), 8)
        self.assertEqual(len([line for line in output.splitlines() if line.startswith("Starting ")]), 9)
        self.assertEqual(len([line for line in output.splitlines() if line.startswith("Finished ")]), 8)
        self.assertEqual(len([line for line in output.splitlines() if line.startswith("Completed ")]), 1)
        self.assertEqual(pd.read_csv(self.topline_path).columns.tolist(), [
            "metric", "gpt56_sol", "paper_gpt4", "difference_vs_paper",
        ])
        tables = [item for item in rendered if type(item).__name__ == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].data.columns.tolist(), [
            "metric", "gpt56_sol", "paper_gpt4", "difference_vs_paper",
        ])

    def test_confirmed_full_resumes_smoke_and_requests_remaining_1112(self):
        self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"},
            FakeClient([FakeResponse("test-output-never-display") for _ in range(8)]),
            allow_scoring=True,
        )
        self.assertTrue(self.smoke_marker_path.exists())
        client = FakeClient([FakeResponse("test-output-never-display") for _ in range(1112)])
        _, _, full_definitions = self.run_notebook({
            "CORAL_PAPER_REPLICATION_RUN": "full",
            "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
        }, client, allow_scoring=True)
        self.assertEqual(len(client.responses.calls), 1112)
        self.assertEqual(len(self.checkpoint_path.read_text().splitlines()), 1120)
        self.assertEqual(len(pd.read_csv(self.topline_path)), 3)
        self.assertIsNotNone(full_definitions["scores"])
        self.assertEqual(len(pd.read_csv(self.output_path / f"{PREFIX}_outputs.csv")), 1120)
        resumed_client = FakeClient([])
        self.run_notebook({
            "CORAL_PAPER_REPLICATION_RUN": "full",
            "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
        }, resumed_client, allow_scoring=True)
        self.assertEqual(resumed_client.responses.calls, [])

    def test_completed_smoke_without_scoring_marker_cannot_unlock_full(self):
        self.write_checkpoint(self.smoke_records())
        for ui in (None, {"full": True, "confirmation": "RUN 1120"}):
            with self.subTest(ui=ui):
                output, rendered, _ = self.run_notebook({
                    "CORAL_PAPER_REPLICATION_RUN": "full",
                    "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
                }, ui=ui)
                visible = output + "\n".join(getattr(item, "text", str(item)) for item in rendered)
                self.assertIn("Rejected", visible)
                self.assertIn("scored smoke", visible)

    def test_successful_smoke_writes_bound_marker_and_score_snapshot(self):
        self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"},
            FakeClient([FakeResponse("test-output-never-display") for _ in range(8)]),
            allow_scoring=True,
        )
        marker = json.loads(self.smoke_marker_path.read_text())
        self.assertEqual(marker["deployment"], "gpt-5.6-sol")
        self.assertEqual(len(marker["smoke_keys"]), 8)
        self.assertEqual(len(marker["source_sha256"]), 64)
        self.assertEqual(len(marker["protocol_sha256"]), 64)
        self.assertEqual(self.smoke_marker_path.with_suffix(".csv").read_bytes(), self.topline_path.read_bytes())
        self.assertNotIn("test-output-never-display", self.smoke_marker_path.read_text())

    def test_scoring_or_artifact_failure_leaves_no_valid_smoke_marker(self):
        for failure in ("scoring", "artifact"):
            with self.subTest(failure=failure):
                self.write_checkpoint(self.smoke_records())
                output, rendered, _ = self.run_notebook(
                    {"CORAL_PAPER_REPLICATION_RUN": "smoke"}, FakeClient([]), allow_scoring=True,
                    score_error=ValueError("test-output-never-display") if failure == "scoring" else None,
                    artifact_error=OSError("test-output-never-display") if failure == "artifact" else None,
                )
                visible = output + "\n".join(getattr(item, "text", str(item)) for item in rendered)
                self.assertIn("Scoring or artifact writing failed", visible)
                self.assertFalse(self.smoke_marker_path.exists())
                rejection, _, _ = self.run_notebook({
                    "CORAL_PAPER_REPLICATION_RUN": "full",
                    "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
                })
                self.assertIn("Rejected", rejection)

    def test_failed_smoke_rescoring_invalidates_prior_success_marker(self):
        self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"},
            FakeClient([FakeResponse("test-output-never-display") for _ in range(8)]),
            allow_scoring=True,
        )
        self.assertTrue(self.smoke_marker_path.exists())
        self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"}, FakeClient([]), allow_scoring=True,
            score_error=ValueError("test-output-never-display"),
        )
        self.assertFalse(self.smoke_marker_path.exists())

    def test_stale_marker_identity_or_score_snapshot_cannot_unlock_full(self):
        self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"},
            FakeClient([FakeResponse("test-output-never-display") for _ in range(8)]),
            allow_scoring=True,
        )
        marker = json.loads(self.smoke_marker_path.read_text())
        for field, value in (
            ("deployment", "other-deployment"), ("smoke_keys", marker["smoke_keys"][:7]),
            ("source_sha256", "0" * 64), ("protocol_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                self.smoke_marker_path.write_text(json.dumps({**marker, field: value}))
                output, _, _ = self.run_notebook({
                    "CORAL_PAPER_REPLICATION_RUN": "full",
                    "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
                })
                self.assertIn("Rejected", output)
        self.smoke_marker_path.write_text(json.dumps(marker))
        self.smoke_marker_path.with_suffix(".csv").write_text("stale scores")
        output, _, _ = self.run_notebook({
            "CORAL_PAPER_REPLICATION_RUN": "full", "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
        })
        self.assertIn("Rejected", output)

    def test_changed_smoke_output_or_source_invalidates_marker(self):
        self.run_notebook(
            {"CORAL_PAPER_REPLICATION_RUN": "smoke"},
            FakeClient([FakeResponse("test-output-never-display") for _ in range(8)]),
            allow_scoring=True,
        )
        original_checkpoint = self.checkpoint_path.read_text()
        records = [json.loads(line) for line in original_checkpoint.splitlines()]
        records[0]["output_text"] = "changed output"
        self.write_checkpoint(records)
        output, _, _ = self.run_notebook({
            "CORAL_PAPER_REPLICATION_RUN": "full", "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
        })
        self.assertIn("Rejected", output)
        self.checkpoint_path.write_text(original_checkpoint)
        source = load_source(self.source_path)
        source.loc[source.index[0], "annotation_set"] = ""
        self.source_path = self.output_path / "changed_source.csv"
        source.to_csv(self.source_path, index=False)
        output, _, _ = self.run_notebook({
            "CORAL_PAPER_REPLICATION_RUN": "full", "CORAL_PAPER_REPLICATION_CONFIRM": "RUN 1120",
        })
        self.assertIn("Rejected", output)

    def test_custom_deployment_smoke_retains_identity_and_produces_topline(self):
        client = FakeClient([FakeResponse("test-output-never-display") for _ in range(8)])
        self.run_notebook({
            "CORAL_PAPER_REPLICATION_RUN": "smoke",
            "AZURE_OPENAI_DEPLOYMENT": "sol-production",
        }, client, allow_scoring=True)
        self.assertTrue(pd.read_csv(self.topline_path).gpt56_sol.notna().all())
        records = [json.loads(line) for line in self.checkpoint_path.read_text().splitlines()]
        self.assertEqual({record["model"] for record in records}, {"sol-production"})
