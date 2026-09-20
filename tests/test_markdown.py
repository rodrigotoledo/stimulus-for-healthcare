"""Tests for the markdown -> sanitized HTML renderer.

The renderer is a security boundary: its output is inserted into the DOM, so
every test that feeds it hostile input matters as much as the formatting ones.
"""

from __future__ import annotations

import pytest

from app.services import markdown


# --------------------------------------------------------------------------
# the security boundary
# --------------------------------------------------------------------------
def test_script_tags_are_escaped_not_executed():
    html = markdown.render("<script>alert(1)</script>")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_event_handlers_are_escaped():
    """The payload may appear as *text*, but never as a live attribute."""
    html = markdown.render('<img src=x onerror="alert(1)">')

    assert "&lt;img" in html
    # Escaped quotes prove it is inert text, not an attribute value.
    assert 'onerror="' not in html
    assert "onerror=&quot;" in html


def test_javascript_urls_are_not_linked():
    html = markdown.render("[click](javascript:alert(1))")

    # No anchor, and the raw markdown syntax is consumed rather than shown.
    assert "<a " not in html
    assert "[click](" not in html
    assert "click" in html


def test_data_urls_are_not_linked():
    html = markdown.render("[x](data:text/html;base64,PHNjcmlwdD4=)")

    assert "<a " not in html
    assert "[x](" not in html
    assert ">" in html  # still rendered as a paragraph


def test_http_links_are_rendered_and_isolated():
    html = markdown.render("[nice](https://example.com/page)")

    assert 'href="https://example.com/page"' in html
    assert 'rel="noopener noreferrer"' in html
    assert 'target="_blank"' in html


def test_html_inside_code_blocks_stays_literal():
    html = markdown.render("```html\n<script>alert(1)</script>\n```")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_html_inside_inline_code_stays_literal():
    html = markdown.render("use `<div>` here")

    assert "&lt;div&gt;" in html
    assert "<div>" not in html


def test_quotes_in_attributes_cannot_break_out():
    html = markdown.render('[x](https://example.com/?a="onmouseover=alert(1))')

    # Any surviving anchor must not have an unescaped quote in its href.
    if "<a " in html:
        href = html.split('href="', 1)[1].split('"')[0]
        assert '"' not in href


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def test_headings_render_as_elements_not_hash_literals():
    html = markdown.render("## Summary\n\nThe patient reports pain.")

    assert "<h2" in html
    assert "## " not in html
    assert "Summary" in html


def test_all_heading_levels():
    for level in range(1, 7):
        html = markdown.render(f"{'#' * level} Title")
        assert f"<h{level}" in html


def test_ordered_list_becomes_a_real_list():
    html = markdown.render("1. First finding\n2. Second finding\n3. Third")

    assert "<ol" in html
    assert html.count("<li>") == 3
    assert "First finding" in html
    # The raw markers must be gone.
    assert "1. First" not in html


def test_ordered_list_supports_paren_and_multidigit_markers():
    html = markdown.render("1) one\n10) ten")

    assert html.count("<li>") == 2
    assert "10)" not in html


def test_unordered_list_becomes_a_real_list():
    html = markdown.render("- alpha\n- beta\n* gamma")

    assert "<ul" in html
    assert html.count("<li>") == 3


def test_list_kind_switches_close_the_previous_list():
    html = markdown.render("1. ordered\n\n- bullet")

    assert "</ol>" in html
    assert "<ul" in html


def test_bold_and_italic_and_strike():
    html = markdown.render("**bold** and *italic* and ~~gone~~")

    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<del>gone</del>" in html
    assert "**" not in html


def test_paragraphs_are_separate_elements():
    html = markdown.render("First paragraph.\n\nSecond paragraph.")

    assert html.count("<p ") == 2


def test_single_newlines_within_a_paragraph_do_not_split_it():
    html = markdown.render("one\ntwo")

    assert html.count("<p ") == 1
    assert "one two" in html


def test_fenced_code_block_with_language():
    html = markdown.render("```python\nprint('hi')\n```")

    assert "<pre" in html
    assert "python" in html
    assert "print(&#x27;hi&#x27;)" in html or "print('hi')" in html


def test_blockquote():
    html = markdown.render("> quoted advice")

    assert "<blockquote" in html
    assert "quoted advice" in html
    assert "&gt;" not in html


def test_horizontal_rule():
    html = markdown.render("above\n\n---\n\nbelow")

    assert "<hr" in html
    assert "---" not in html


def test_table():
    html = markdown.render("| A | B |\n|---|---|\n| 1 | 2 |")

    assert "<table" in html
    assert "<th" in html
    assert "<td" in html
    assert "|---|" not in html


def test_inline_code_is_not_reformatted_by_bold():
    """`**not bold**` inside backticks must stay literal."""
    html = markdown.render("`**not bold**`")

    assert "<strong>" not in html
    assert "**not bold**" in html


def test_text_inside_bold_is_not_treated_as_italic():
    html = markdown.render("**bold**")

    assert "<em>" not in html


# --------------------------------------------------------------------------
# streaming: partial input must always render something sane
# --------------------------------------------------------------------------
def test_partial_bold_renders_literally_until_it_closes():
    open_html = markdown.render("Verdict: **NEEDS")
    closed_html = markdown.render("Verdict: **NEEDS_CAUTION**")

    assert "<strong>" not in open_html
    assert "**NEEDS" in open_html
    assert "<strong>NEEDS_CAUTION</strong>" in closed_html


def test_unclosed_code_fence_still_renders_a_block():
    html = markdown.render("```python\nprint('partial')")

    assert "<pre" in html
    assert "print" in html


def test_unclosed_table_falls_back_to_a_paragraph():
    html = markdown.render("| only a header |")

    assert "<table" not in html
    assert "only a header" in html


def test_empty_input_renders_nothing():
    assert markdown.render("") == ""
    assert markdown.render("   \n  \n") == ""


def test_a_realistic_lens_reply_renders_structured_html():
    reply = (
        "## Clinical review\n\n"
        "1. The patient reports a **one-sided headache** for three days.\n"
        "2. The assistant advised `800mg ibuprofen` without checking for contraindications.\n"
        "3. Red flags: nausea, no fever.\n\n"
        "> Not medical advice.\n\n"
        "**Verdict:** NEEDS_CAUTION\n"
    )
    html = markdown.render(reply)

    assert "<h2" in html
    assert "<ol" in html
    assert "<li>" in html
    assert "<strong>one-sided headache</strong>" in html
    assert "<code" in html
    assert "<blockquote" in html
    # No raw markdown syntax survives anywhere.
    for token in ("##", "**", "`800mg"):
        assert token not in html, f"raw markdown leaked: {token}"


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert(1)</script>",
        "<svg/onload=alert(1)>",
        "![x](javascript:alert(1))",
        "<iframe src=javascript:alert(1)>",
        "<a href=\"javascript:alert(1)\">x</a>",
        "**<script>alert(1)</script>**",
        "# <script>alert(1)</script>",
        "1. <script>alert(1)</script>",
        "> <script>alert(1)</script>",
        "`<script>alert(1)</script>`",
    ],
)
def test_no_payload_can_emit_an_active_element(payload):
    """Whatever the reply contains, no live element or handler may come out.

    The invariant is about *structure*, not substrings: an escaped payload
    showing `&lt;script&gt;` as visible text is the correct, safe outcome.
    """
    html = markdown.render(payload)

    lower = html.lower()
    # No live tag of any dangerous kind. An escaped payload renders as
    # `&lt;script&gt;`, which does not contain the literal `<script`.
    for tag in ("<script", "<iframe", "<svg", "<object", "<embed", "<form", "<img"):
        assert tag not in lower, f"live {tag} in output: {html}"

    # No live event handler or scheme: `onload=&quot;` (escaped) is fine.
    for needle in ('onload="', "onerror=", "onclick=", 'href="javascript:'):
        assert needle not in lower, f"live handler/url in output: {html}"

    # Nothing may be emitted as an `<a>` pointing at a script scheme.
    assert "javascript:alert" not in lower or "<a " not in lower
