"""Guard the UI language: the served page must not leak English labels.

The manual label sweep that found `Stop`, `Go`, `no models found` and
`turn`/`msg` was a one-off script, so nothing stopped the next template edit
from reintroducing English. This turns it into a test.

It enumerates the strings the page actually *renders* (buttons, labels,
headings, options, placeholders) rather than grepping templates, because that
is what caught the leftovers a file-by-file search missed.
"""

from __future__ import annotations

import re

import pytest

from app.lenses import lens_catalog
from app.templating import ROLE_LABELS

# Words that must not appear in the pt-BR UI. Kept to words that are
# unambiguously English UI chrome — never dataset ids, model names, config
# names, dataset notes or transcript content, which are legitimately English.
FORBIDDEN_WORDS = [
    "Stop",
    "Prev",
    "Next",
    "Go",
    "Search",
    "Dataset",
    "Datasets",
    "Dialogs",
    "Analysis",
    "Transcript",
    "Temperature",
    "Config",
    "Split",
    "models",
    "turn",
    "msg",
    "not stated",
    "Research tool",
    "Curated",
    "Browse",
    "Select",
]

# Substrings exempted because they belong to identifiers or upstream data
# rather than UI copy.
EXEMPT_PATTERNS = [
    r"data-[a-z-]+=\"[^\"]*\"",   # Stimulus attributes stay English
    r"[\w.-]+/[\w.-]+",           # owner/dataset ids
    r"\b[\w.-]+:[\w.-]+\b",       # model tags such as gemma4:latest
    r"\b(?:text/event-stream|application/json|AbortError)\b",
]


def _strip_exempt(text: str) -> str:
    """Remove attribute values but keep the enclosing tags readable."""
    for pattern in EXEMPT_PATTERNS:
        text = re.sub(pattern, " ", text)
    return text


def _visible_strings(html: str) -> list[str]:
    """Text a user can read: element bodies, placeholders, titles, labels."""
    found = re.findall(r">([^<>]+)<", html)
    found += re.findall(r'placeholder="([^"]+)"', html)
    found += re.findall(r'title="([^"]+)"', html)
    found += re.findall(r"<label[^>]*>([^<]+)</label>", html)
    return [text.strip() for text in found if text.strip()]


def _catalog_notes() -> set[str]:
    """Dataset notes are English prose from upstream — not UI copy."""
    from app.catalog import CURATED_DATASETS

    return {entry["note"] for entry in CURATED_DATASETS if entry.get("note")}


def _visible_strings_excluding_data(html: str) -> list[str]:
    notes = _catalog_notes()
    return [
        text
        for text in _visible_strings(html)
        if not any(text.startswith(note[:40]) for note in notes)
    ]


def test_index_page_has_no_english_ui_strings(ui):
    client, _, _ = ui
    body = _strip_exempt(client.get("/").text)

    leaked = {}
    for text in _visible_strings_excluding_data(body):
        for word in FORBIDDEN_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                leaked.setdefault(word, set()).add(text[:60])

    assert not leaked, f"English left in the pt-BR UI: {leaked}"


def test_curated_fragment_has_no_english_ui_strings(ui):
    client, _, _ = ui
    body = _strip_exempt(client.get("/fragments/datasets/curated").text)

    leaked = [text for text in _visible_strings(body) if "curated" in text.lower()]
    assert not leaked, f"untranslated badge text: {leaked}"


def test_splits_fragment_labels_are_portuguese(ui):
    client, _, _ = ui
    body = client.get("/fragments/datasets/some/dataset/splits").text

    assert "Configuração" in body
    assert "Divisão" in body
    assert ">Config<" not in body
    assert ">Split<" not in body


def test_rows_fragment_counters_are_portuguese(ui):
    client, _, _ = ui
    body = client.get(
        "/fragments/datasets/ruslanmv/ai-medical-chatbot/rows",
        params={"config": "default", "split": "train"},
    ).text

    assert "turno" in body
    assert "mensagens" in body
    # The old English counters must be gone.
    assert re.search(r"\bturn\b(?!o)", body) is None
    assert re.search(r"\bmsg\b", body) is None


def test_pagination_controls_are_portuguese(ui):
    client, _, _ = ui
    body = client.get(
        "/fragments/datasets/ruslanmv/ai-medical-chatbot/rows",
        params={"config": "default", "split": "train"},
    ).text

    assert "Anterior" in body
    assert "Próximo" in body
    assert ">Prev<" not in body and "Prev" not in body
    assert "Next" not in body


def test_english_lens_titles_are_not_offered_by_default(ui):
    """The picker is server-rendered in pt-BR, so no English title may appear."""
    client, _, _ = ui
    body = client.get("/").text

    # Scope to the lens <select>: dataset notes elsewhere are English prose
    # describing *how* to use a lens, not a UI label.
    picker = body.split('id="lens-select"', 1)[1].split("</select>", 1)[0]
    options = re.findall(r">([^<>]+)<", picker)

    pt_titles = {entry["title"] for entry in lens_catalog("pt-BR")}
    rendered = {text.strip() for text in options if text.strip()}

    for entry in lens_catalog("en"):
        if entry["title"] in pt_titles:
            continue  # identical in both, e.g. the SOAP acronym
        assert entry["title"] not in rendered, f"English lens title offered: {entry['title']}"

    for title in pt_titles:
        assert title in rendered, f"missing pt-BR lens title: {title}"


def test_role_labels_are_portuguese():
    assert ROLE_LABELS["user"] == "Paciente / Usuário"
    assert ROLE_LABELS["assistant"] == "Profissional / Assistente"
    assert ROLE_LABELS["reasoning"] == "Trace de raciocínio"


def test_language_picker_offers_portuguese_first(ui):
    client, _, _ = ui
    body = client.get("/").text

    assert "Português (Brasil)" in body
    assert "English" in body
    # pt-BR must be the pre-selected option.
    assert re.search(r'value="pt-BR"\s+selected', body)


def test_no_english_in_the_analysis_panel(ui):
    client, _, _ = ui
    # Not stripped: the split key is a data-controller attribute.
    body = client.get("/").text

    panel = body.split('data-controller="analysis-stream"', 1)[1]
    leaked = {}
    for text in _visible_strings(panel):
        for word in ("Analyze", "Stop", "Select a dialog", "no models found"):
            if word.lower() in text.lower():
                leaked.setdefault(word, set()).add(text[:60])

    assert not leaked, f"English left in the analysis panel: {leaked}"
