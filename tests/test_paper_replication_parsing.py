import json
import unittest

from coral import CancerDiagnosis, RadTest, SymptomEnt, task_to_default_tuple_dict
from coral.paper_replication.parsing import (
    parse_annotation_set,
    parse_namedtuple_expression,
    parse_paper_output,
    serialize_parsed_tuples,
)


class PaperParserTests(unittest.TestCase):
    def test_malformed_field_shapes_default_and_serialize_safely(self):
        malformed = (
            "SymptomEnt('fatigue', None)",
            "SymptomEnt('fatigue', {1, 'today'})",
            "SymptomEnt('fatigue', {b'today'})",
            "SymptomEnt(b'fatigue', {'today'})",
            "SymptomEnt(1, {'today'})",
            "SymptomEnt('fatigue', 'today')",
            "SymptomEnt('fatigue', [['today']])",
            "SymptomEnt('fatigue', {'date': None})",
            "SymptomEnt('fatigue', {'date': ['today']})",
            "SymptomEnt('fatigue', {b'date': 'today'})",
            "SymptomEnt('fatigue', {1: 'today', 'date': 'yesterday'})",
        )
        for output in malformed:
            with self.subTest(output=output):
                parsed = parse_paper_output(output, "symptoms")
                self.assertEqual(parsed, [task_to_default_tuple_dict["symptoms"]])
                self.assertEqual(json.loads(serialize_parsed_tuples(parsed))[0]["fields"], {
                    "Symptom": "unknown", "Datetime": ["unknown"],
                })

    def test_malformed_annotation_field_shapes_raise_contextual_errors(self):
        for relation in ("None", "{1, 'today'}", "{b'today'}", "[['today']]"):
            with self.subTest(relation=relation), self.assertRaisesRegex(
                ValueError, "invalid annotation for task symptoms on line 1"
            ):
                parse_annotation_set(f"SymptomEnt('fatigue', {relation})", "symptoms")

    def test_diagnosis_datetime_requires_string_collection(self):
        for value in ("None", "'today'", "{1}", "{b'today'}"):
            with self.subTest(value=value):
                self.assertEqual(
                    parse_paper_output(f"CancerDiagnosis({value})", "symptoms_at_diagnosis"),
                    [task_to_default_tuple_dict["symptoms_at_diagnosis"]],
                )

    def test_secondary_scalar_fields_require_strings(self):
        for output, task in (
            ("FutureMedEnt('drug', {'planned'}, {'rash'})", "future_med_consideration_ae"),
            (
                "PrescribedMedEnt('drug', {'today'}, {'unknown'}, {'cancer'}, None, {'rash'}, {'unknown'})",
                "prescribed_med_begin_end_reason_continuity_ae",
            ),
        ):
            with self.subTest(task=task):
                self.assertEqual(parse_paper_output(output, task), [task_to_default_tuple_dict[task]])

    def test_preserves_diagnosis_constructor_only_for_symptoms_at_diagnosis(self):
        diagnosis = "CancerDiagnosis(Datetime={'today'})"
        self.assertEqual(
            parse_namedtuple_expression(diagnosis, "symptoms_at_diagnosis"),
            CancerDiagnosis({"today"}),
        )
        for task in ("symptoms", "symptoms_due_to_cancer", "histology_datetime"):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "constructor"):
                parse_namedtuple_expression(diagnosis, task)

    def test_parses_keyword_and_positional_named_tuples(self):
        keyword = parse_namedtuple_expression(
            "SymptomEnt(Symptom='fatigue', Datetime={'today'})", "symptoms"
        )
        positional = parse_namedtuple_expression(
            "SymptomEnt('fatigue', {'today'})", "symptoms"
        )
        self.assertEqual(keyword, SymptomEnt("fatigue", {"today"}))
        self.assertEqual(positional, keyword)

    def test_splits_paper_style_space_separated_and_newline_outputs(self):
        output = (
            "SymptomEnt(Symptom='fatigue', Datetime={'today'}) "
            "SymptomEnt(Symptom='pain', Datetime={'yesterday'})\nN/A"
        )
        self.assertEqual(
            parse_paper_output(output, "symptoms"),
            [
                SymptomEnt("fatigue", {"today"}),
                SymptomEnt("pain", {"yesterday"}),
                task_to_default_tuple_dict["symptoms"],
            ],
        )

    def test_empty_nonanswer_and_malformed_text_default(self):
        for output in ("", "N/A", "no symptoms", "none reported", "unknown", "bad("):
            with self.subTest(output=output):
                self.assertEqual(
                    parse_paper_output(output, "symptoms"),
                    [task_to_default_tuple_dict["symptoms"]],
                )

    def test_each_malformed_line_gets_a_default(self):
        parsed = parse_paper_output("bad(\nN/A", "symptoms")
        self.assertEqual(parsed, [task_to_default_tuple_dict["symptoms"]] * 2)

    def test_annotation_lines_use_the_same_safe_constructor_parser(self):
        parsed = parse_annotation_set(
            "RadTest(RadiologyTest='CT', Datetime={'today'}, Site={'chest'}, "
            "Reason={'staging'}, Result={'stable'})\n",
            "radtest_datetime_site_reason_result",
        )
        self.assertEqual(
            parsed,
            [RadTest("CT", {"today"}, {"chest"}, {"staging"}, {"stable"})],
        )

    def test_every_paper_task_accepts_its_default_constructor(self):
        for task, default in task_to_default_tuple_dict.items():
            with self.subTest(task=task):
                self.assertEqual(parse_paper_output(repr(default), task), [default])

    def test_serialization_is_json_and_stable_for_sets(self):
        rendered = serialize_parsed_tuples([SymptomEnt("fatigue", {"today", "yesterday"})])
        self.assertEqual(
            json.loads(rendered),
            [{"constructor": "SymptomEnt", "fields": {
                "Symptom": "fatigue", "Datetime": ["today", "yesterday"]
            }}],
        )

    def test_rejects_wrong_constructor_for_task(self):
        with self.assertRaisesRegex(ValueError, "constructor"):
            parse_namedtuple_expression(
                "RadTest('CT', {'today'}, {'chest'}, {'staging'}, {'stable'})",
                "symptoms",
            )

    def test_duplicate_keyword_model_output_defaults(self):
        output = (
            "SymptomEnt(Symptom='first', Symptom='second', "
            "Datetime={'today'})"
        )
        self.assertEqual(
            parse_paper_output(output, "symptoms"),
            [task_to_default_tuple_dict["symptoms"]],
        )

    def test_duplicate_keyword_annotation_raises_contextually(self):
        annotation = (
            "SymptomEnt(Symptom='first', Symptom='second', "
            "Datetime={'today'})"
        )
        with self.assertRaisesRegex(ValueError, "invalid annotation"):
            parse_annotation_set(annotation, "symptoms")

    def test_arbitrary_python_is_never_executed(self):
        payloads = (
            "__import__('os').system('false')",
            "SymptomEnt(Symptom=open('/tmp/x'), Datetime={'today'})",
            "SymptomEnt(**{'Symptom': 'fatigue', 'Datetime': {'today'}})",
            "(lambda: 1)()",
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(
                    parse_paper_output(payload, "symptoms"),
                    [task_to_default_tuple_dict["symptoms"]],
                )
