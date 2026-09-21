import json
import unittest

from coral import RadTest, SymptomEnt, task_to_default_tuple_dict
from coral.paper_replication.parsing import (
    parse_annotation_set,
    parse_namedtuple_expression,
    parse_paper_output,
    serialize_parsed_tuples,
)


class PaperParserTests(unittest.TestCase):
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
