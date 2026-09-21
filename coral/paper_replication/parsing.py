"""Safe parsing and serialization for paper-replication named tuples."""

import ast
import json
import re

from coral import task_to_default_tuple_dict

from .protocol import TASK_ORDER


TASK_CONSTRUCTORS = {
    task: type(default)
    for task, default in task_to_default_tuple_dict.items()
    if task in TASK_ORDER
}
NON_ANSWERS = ("no ", "none ")


def _constructor_for_task(task: str) -> type[tuple]:
    try:
        return TASK_CONSTRUCTORS[task]
    except KeyError as error:
        raise ValueError(f"unknown paper task: {task}") from error


def parse_namedtuple_expression(source: str, task: str) -> tuple:
    """Parse one allowlisted named-tuple call without executing source text."""
    constructor = _constructor_for_task(task)
    try:
        expression = ast.parse(source, mode="eval")
    except (SyntaxError, TypeError) as error:
        raise ValueError("invalid named-tuple expression") from error

    call = expression.body
    if not isinstance(call, ast.Call):
        raise ValueError("named-tuple expression must be a constructor call")
    if not isinstance(call.func, ast.Name) or call.func.id != constructor.__name__:
        raise ValueError(f"unexpected constructor for task {task}")
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise ValueError("starred positional arguments are not allowed")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("keyword unpacking is not allowed")

    try:
        arguments = [ast.literal_eval(argument) for argument in call.args]
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        }
    except (ValueError, TypeError, MemoryError, RecursionError) as error:
        raise ValueError("named-tuple arguments must be literals") from error

    try:
        return constructor(*arguments, **keywords)
    except TypeError as error:
        raise ValueError(f"invalid {constructor.__name__} fields") from error


def _paper_output_text(output: object) -> str:
    if not isinstance(output, str):
        return ""
    return output.replace("o'clock", "o clock")


def _is_nonanswer(line: str) -> bool:
    normalized = line.lower()
    return (
        not normalized
        or normalized in {"n/a", "unknown"}
        or normalized.startswith(NON_ANSWERS)
    )


def parse_paper_output(output: object, task: str) -> list[tuple]:
    """Parse paper model output, using the task default for every bad line."""
    _constructor_for_task(task)
    text = _paper_output_text(output)
    text = re.sub(r"(\))\s([A-Z]+)", r"\1\n\2", text).strip()
    if not text:
        return [task_to_default_tuple_dict[task]]

    parsed = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _is_nonanswer(line):
            parsed.append(task_to_default_tuple_dict[task])
            continue
        try:
            parsed.append(parse_namedtuple_expression(line, task))
        except ValueError:
            parsed.append(task_to_default_tuple_dict[task])
    return parsed


def parse_annotation_set(annotation_set: object, task: str) -> list[tuple]:
    """Parse trusted-format annotation lines while preserving malformed-gold errors."""
    _constructor_for_task(task)
    if not isinstance(annotation_set, str) or not annotation_set.strip():
        return [task_to_default_tuple_dict[task]]

    parsed = []
    for line_number, raw_line in enumerate(annotation_set.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed.append(parse_namedtuple_expression(line, task))
        except ValueError as error:
            raise ValueError(
                f"invalid annotation for task {task} on line {line_number}: {error}"
            ) from error
    return parsed or [task_to_default_tuple_dict[task]]


def _json_value(value: object) -> object:
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(value[key]) for key in sorted(value)}
    return value


def serialize_parsed_tuples(values: list[tuple]) -> str:
    """Serialize named tuples as deterministic JSON records for CSV artifacts."""
    records = []
    for value in values:
        if not hasattr(value, "_fields"):
            raise ValueError("parsed values must be named tuples")
        records.append(
            {
                "constructor": type(value).__name__,
                "fields": {
                    field: _json_value(getattr(value, field)) for field in value._fields
                },
            }
        )
    return json.dumps(records, sort_keys=True)
