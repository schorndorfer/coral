import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coral import CancerDiagnosis, PrescribedMedEnt, StageEnt, SymptomEnt
from coral.paper_replication.scoring import (
    format_relations,
    score_completed_records,
    write_artifacts,
)


class PaperRelationFormattingTests(unittest.TestCase):
    def test_formats_relations_like_paper_and_omits_additional_testing(self):
        relations = format_relations([
            StageEnt("II", {"today"}, {"PET"}),
            SymptomEnt("Fatigue", {"Today"}),
        ])
        self.assertEqual(relations["Stage Datetime"], {"II today"})
        self.assertEqual(relations["Symptom Datetime"], {"Fatigue Today"})
        self.assertNotIn("Stage AdditionalTesting", relations)

    def test_formats_scalar_and_set_medication_fields(self):
        relations = format_relations([
            PrescribedMedEnt(
                "Drug", {"start"}, {"end"}, {"cancer"}, "ongoing",
                {"rash"}, {"swelling"},
            )
        ])
        self.assertEqual(relations["MedicationName Continuity"], {"Drug ongoing"})
        self.assertEqual(relations["MedicationName Reason"], {"Drug cancer"})

    def test_rejects_tuple_sets_that_produce_no_relations(self):
        with self.assertRaisesRegex(ValueError, "no relations"):
            format_relations([CancerDiagnosis({"today"})])


class FakeMetrics:
    def compute_bleu_score(self, preds, references, max_n, smooth):
        return {"bleu": float(preds[0] in references[0])}

    def compute_rouge_score(self, preds, references, rouge_types):
        return {"rouge1": float(preds[0] == references[0])}

    def compute_em_over_multiset_prec_recall_f1(self, outputs, annotations):
        output_set, annotation_set = set(outputs), set(annotations)
        true_positive = len(output_set & annotation_set)
        precision = true_positive / len(output_set)
        recall = true_positive / len(annotation_set)
        f1 = 0.0 if precision == recall == 0.0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1


SOURCE = pd.DataFrame([{
    "doc_idx": "1",
    "section_name": "hpi",
    "section_text": "Fatigue today.",
    "task": "symptoms",
    "annotation_set": "SymptomEnt(Symptom='fatigue', Datetime={'today'})\n",
}])

COMPLETED = {
    "doc_idx": "1", "section_name": "hpi", "task": "symptoms",
    "model": "gpt-5.6-sol", "status": "completed",
    "output_text": "SymptomEnt(Symptom='fatigue', Datetime={'today'})",
    "input_tokens": 100, "output_tokens": 20, "cost": 0.0008,
    "elapsed_seconds": 1.0,
}


class PaperScoringTests(unittest.TestCase):
    def test_custom_deployment_alias_gets_topline_scores(self):
        custom = {**COMPLETED, "model": "sol-production"}
        scores = score_completed_records(SOURCE, [custom], FakeMetrics(), model="sol-production")
        self.assertEqual(scores.topline.gpt56_sol.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(scores.outputs.model.tolist(), ["sol-production"])
        self.assertEqual(set(scores.relations.model), {"sol-production"})

    def test_sole_completed_model_is_inferred_without_literal_deployment(self):
        custom = {**COMPLETED, "model": "sol-production"}
        scores = score_completed_records(SOURCE, [custom], FakeMetrics())
        self.assertEqual(scores.topline.gpt56_sol.tolist(), [1.0, 1.0, 1.0])

    def test_multiple_completed_models_require_explicit_selection(self):
        custom = {**COMPLETED, "model": "sol-production", "output_text": "invalid"}
        with self.assertRaisesRegex(ValueError, "exactly one completed model"):
            score_completed_records(SOURCE, [COMPLETED, custom], FakeMetrics())
        scores = score_completed_records(
            SOURCE, [COMPLETED, custom], FakeMetrics(), model="sol-production",
        )
        self.assertEqual(scores.topline.gpt56_sol.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(scores.outputs.model.tolist(), ["sol-production"])

    def test_empty_records_require_model_but_explicit_selection_stays_empty(self):
        with self.assertRaisesRegex(ValueError, "exactly one completed model"):
            score_completed_records(SOURCE, [], FakeMetrics())
        scores = score_completed_records(SOURCE, [], FakeMetrics(), model="sol-production")
        self.assertTrue(scores.outputs.empty)
        self.assertTrue(scores.topline.gpt56_sol.isna().all())

    def test_scores_only_completed_api_outputs(self):
        failed = {**COMPLETED, "doc_idx": "2", "status": "api_failed", "output_text": None}
        scores = score_completed_records(SOURCE, [COMPLETED, failed], FakeMetrics())
        self.assertEqual(len(scores.outputs), 1)
        self.assertEqual(len(scores.instances), 1)
        self.assertEqual(scores.instances.iloc[0].em_f1, 1.0)

    def test_malformed_completed_output_is_scored_as_paper_default(self):
        malformed = {**COMPLETED, "output_text": "not a named tuple"}
        scores = score_completed_records(SOURCE, [malformed], FakeMetrics())
        self.assertEqual(scores.instances.iloc[0].em_f1, 0.0)

    def test_missing_annotation_uses_default_reference(self):
        no_annotation_source = SOURCE.iloc[0:0].copy()
        unknown = {
            **COMPLETED,
            "output_text": "SymptomEnt(Symptom='unknown', Datetime={'unknown'})",
        }
        scores = score_completed_records(no_annotation_source, [unknown], FakeMetrics())
        self.assertEqual(scores.instances.iloc[0].em_f1, 1.0)

    def test_rejects_duplicate_matching_annotation_rows(self):
        duplicate_source = pd.concat([SOURCE, SOURCE], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate source annotations"):
            score_completed_records(duplicate_source, [COMPLETED], FakeMetrics())

    def test_topline_is_unweighted_mean_of_relation_aggregates(self):
        second = {
            **COMPLETED,
            "doc_idx": "2",
            "output_text": "SymptomEnt(Symptom='pain', Datetime={'today'})",
        }
        source = pd.concat([
            SOURCE,
            pd.DataFrame([{**SOURCE.iloc[0].to_dict(), "doc_idx": "2"}]),
        ], ignore_index=True)
        scores = score_completed_records(source, [COMPLETED, second], FakeMetrics())
        self.assertEqual(scores.topline.metric.tolist(), ["BLEU-4", "ROUGE-1", "EM F1"])
        self.assertEqual(scores.topline.paper_gpt4.tolist(), [0.73, 0.72, 0.51])
        self.assertEqual(scores.topline.gpt56_sol.tolist(), [0.5, 0.5, 0.5])

    def test_writes_exact_isolated_artifact_names(self):
        scores = score_completed_records(SOURCE, [COMPLETED], FakeMetrics())
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifacts(scores, Path(directory))
            self.assertEqual(paths.outputs.name, "gpt56_sol_paper_replication_outputs.csv")
            self.assertEqual(paths.instances.name, "gpt56_sol_paper_replication_instance_scores.csv")
            self.assertEqual(paths.relations.name, "gpt56_sol_paper_replication_relation_scores.csv")
            self.assertEqual(paths.topline.name, "gpt56_sol_paper_replication_topline.csv")
            for path in (paths.outputs, paths.instances, paths.relations, paths.topline):
                self.assertTrue(path.exists())

    def test_outputs_csv_contains_raw_and_json_parsed_output(self):
        scores = score_completed_records(SOURCE, [COMPLETED], FakeMetrics())
        self.assertIn("output_text", scores.outputs)
        self.assertIn("parsed_output_json", scores.outputs)
        self.assertNotIn("api_key", scores.outputs)
