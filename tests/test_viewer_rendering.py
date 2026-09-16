from __future__ import annotations

import unittest
from html.parser import HTMLParser

from coral.viewer.data import Span, ViewerEntity
from coral.viewer.rendering import entity_color, format_offsets, render_highlighted_text


class _TextCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str):
        self.parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self.parts)


class HighlightRenderingTests(unittest.TestCase):
    def test_escapes_note_text_and_marks_a_single_annotation(self):
        note = "Before <script> & Drug\n"
        entity = ViewerEntity(
            id="T1",
            type="MedicationName",
            spans=(Span(18, 22),),
            text="Drug",
            auxiliary=False,
        )

        rendered = render_highlighted_text(note, (entity,))

        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("white-space: pre-wrap", rendered)
        self.assertIn('data-entity-ids="T1"', rendered)
        self.assertIn('data-entity-types="MedicationName"', rendered)

    def test_entity_colors_are_stable_and_type_specific(self):
        self.assertEqual(entity_color("MedicationName"), entity_color("MedicationName"))
        self.assertNotEqual(entity_color("MedicationName"), entity_color("Datetime"))

    def test_overlap_uses_a_shared_striped_fragment_without_losing_text(self):
        note = "ABCDE"
        left = self.entity("T1", "Problem", (Span(0, 3),))
        right = self.entity("T2", "Treatment", (Span(2, 5),))

        rendered = render_highlighted_text(note, (left, right))

        self.assertIn('data-entity-ids="T1 T2"', rendered)
        self.assertIn('data-entity-types="Problem Treatment"', rendered)
        self.assertIn("linear-gradient", rendered)
        self.assert_rendered_text(note, rendered)

    def test_discontinuous_entity_marks_each_span_without_losing_text(self):
        note = "A B C"
        entity = self.entity("T3", "ClinicalCondition", (Span(0, 1), Span(4, 5)))

        rendered = render_highlighted_text(note, (entity,))

        self.assertEqual(rendered.count('data-entity-ids="T3"'), 2)
        self.assert_rendered_text(note, rendered)

    def test_adjacent_spans_preserve_every_character_once(self):
        note = "ABCD"
        first = self.entity("T4", "Test", (Span(0, 2),))
        second = self.entity("T5", "Datetime", (Span(2, 4),))

        rendered = render_highlighted_text(note, (first, second))

        self.assertIn('data-entity-ids="T4"', rendered)
        self.assertIn('data-entity-ids="T5"', rendered)
        self.assert_rendered_text(note, rendered)

    def test_selected_entity_is_emphasized_in_every_fragment(self):
        note = "A B C"
        entity = self.entity("T6", "MedicationName", (Span(0, 1), Span(4, 5)))

        rendered = render_highlighted_text(note, (entity,), selected_entity_id="T6")

        self.assertEqual(rendered.count('data-selected="true"'), 2)
        self.assertIn("outline:", rendered)
        self.assert_rendered_text(note, rendered)

    def test_empty_entities_return_one_escaped_note_container(self):
        note = "Plain & <note>"

        rendered = render_highlighted_text(note, ())

        self.assertEqual(
            rendered,
            '<div class="coral-note" style="white-space: pre-wrap; overflow-wrap: anywhere">'
            "Plain &amp; &lt;note&gt;</div>",
        )
        self.assert_rendered_text(note, rendered)

    def test_markdown_syntax_after_blank_lines_remains_inside_html_note(self):
        note = (
            "Paragraph one.\n\n"
            "# Literal heading\n\n"
            "- literal list item\n\n"
            "*literal emphasis*\n\n"
            "![literal image](https://invalid.example/image.png)"
        )

        rendered = render_highlighted_text(note, ())

        self.assertNotIn("\n", rendered)
        self.assertEqual(rendered.count('<div class="coral-note"'), 1)
        self.assert_rendered_text(note, rendered)

    def test_entity_metadata_is_escaped_in_all_html_attributes(self):
        entity = self.entity(
            'T1" onmouseover="synthetic()',
            '<img src=x onerror="synthetic()">',
            (Span(0, 1),),
        )

        rendered = render_highlighted_text("X", (entity,))

        self.assertIn(
            'data-entity-ids="T1&quot; onmouseover=&quot;synthetic()"',
            rendered,
        )
        self.assertIn(
            'data-entity-types="&lt;img src=x onerror=&quot;synthetic()&quot;&gt;"',
            rendered,
        )
        self.assertIn(
            'title="&lt;img src=x onerror=&quot;synthetic()&quot;&gt; '
            '(T1&quot; onmouseover=&quot;synthetic())"',
            rendered,
        )
        self.assertIn(
            'aria-label="&lt;img src=x onerror=&quot;synthetic()&quot;&gt; '
            '(T1&quot; onmouseover=&quot;synthetic())"',
            rendered,
        )
        self.assertNotIn("<img", rendered)

    def test_formats_all_entity_offsets_for_display(self):
        entity = self.entity("T7", "Problem", (Span(1, 3), Span(5, 8)))

        self.assertEqual(format_offsets(entity), "1-3; 5-8")

    def entity(self, identifier: str, entity_type: str, spans: tuple[Span, ...]) -> ViewerEntity:
        return ViewerEntity(identifier, entity_type, spans, "synthetic", False)

    def assert_rendered_text(self, expected: str, rendered: str):
        parser = _TextCollector()
        parser.feed(rendered)
        parser.close()
        self.assertEqual(parser.text, expected)
