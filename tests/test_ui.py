"""Tests for the server-rendered UI: pages, fragments and HTML-over-SSE.

These cover the Stimulus surface. The JSON API has its own suite in
`test_api.py`, and the two share the same service layer.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.config import Settings, get_settings
from app.main import app
from app.templating import first_user_content

from test_api import FakeHF, FakeOllama


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
def test_index_renders_the_three_columns(ui):
    client, _, _ = ui
    response = client.get("/")

    assert response.status_code == 200
    body = response.text
    assert "Seletor de Diálogos HuggingFace" in body
    # All four Stimulus controllers are mounted in the markup.
    for controller in (
        "dataset-browser",
        "dialog-explorer",
        "analysis-stream",
        "status",
    ):
        assert f'data-controller="{controller}"' in body


def test_index_ships_the_importmap_and_app_module(ui):
    client, _, _ = ui
    body = client.get("/").text

    assert 'type="importmap"' in body
    assert "cdn.jsdelivr.net/npm/@hotwired/stimulus" in body
    assert "/static/js/app.js" in body


def test_index_lists_curated_datasets_server_side(ui):
    """The first paint is server-rendered — no JS needed to see the list."""
    client, _, _ = ui
    body = client.get("/").text

    assert 'data-dataset-id="lavita/medical-qa-datasets"' in body
    assert body.count("data-dataset-id=") == 8


def test_index_carries_url_templates_for_the_controllers(ui):
    client, _, _ = ui
    body = client.get("/").text

    # The dialog column exposes templates the controller fills with the id.
    assert "__DATASET__" in body


def test_index_shows_ollama_state(ui):
    client, _, _ = ui
    body = client.get("/").text

    assert "modelos" in body  # badge reads "N modelos"


def test_index_degrades_when_ollama_is_down(ui):
    client, _, _ = ui
    app.dependency_overrides[routes.ollama_client] = lambda: FakeOllama(fail=True)
    try:
        body = client.get("/").text
    finally:
        app.dependency_overrides.pop(routes.ollama_client, None)

    assert "Ollama offline" in body


def test_index_prefills_model_and_lens_pickers(ui):
    client, _, _ = ui
    body = client.get("/").text

    assert 'value="MedGemma:4b" selected' in body
    assert 'value="clinical_review" selected' in body


# --------------------------------------------------------------------------
# fragments
# --------------------------------------------------------------------------
def test_health_fragment(ui):
    client, _, _ = ui
    body = client.get("/fragments/health").text

    assert "modelos" in body  # badge reads "N modelos"
    assert "10 modelos" in body or "1 modelos" in body


def test_curated_fragment_lists_datasets(ui):
    client, _, _ = ui
    body = client.get("/fragments/datasets/curated").text

    assert body.count("data-dataset-id=") == 8
    assert 'data-dataset-id="heliosbrahma/mental_health_chatbot_dataset"' in body


def test_search_fragment_marks_curated_hits(ui):
    client, fake_hf, _ = ui
    body = client.get("/fragments/datasets/search", params={"q": "medical"}).text

    assert fake_hf.calls[0] == ("search", "medical", 25)
    assert "some/medical-dialogues" in body


def test_search_fragment_reports_huggingface_errors(ui):
    client, fake_hf, _ = ui
    from app.services.huggingface import HFDatasetsError

    fake_hf.fail_with = HFDatasetsError("dataset is gated", status_code=401)
    response = client.get("/fragments/datasets/search", params={"q": "x"})

    assert response.status_code == 401
    assert "dataset is gated" in response.text


def test_splits_fragment_renders_the_pickers(ui):
    client, _, _ = ui
    body = client.get("/fragments/datasets/some/dataset/splits").text

    assert 'id="config-select"' in body
    assert 'id="split-select"' in body
    assert "train" in body


def test_rows_fragment_renders_cards_and_transcript(ui):
    client, _, _ = ui
    body = client.get(
        "/fragments/datasets/ruslanmv/ai-medical-chatbot/rows",
        params={"config": "default", "split": "train", "length": 2, "selected": 0},
    ).text

    assert body.count("data-dialog-index=") == 2
    # The selected transcript renders as chat bubbles with role labels.
    assert "Paciente / Usuário" in body
    assert "Profissional / Assistente" in body
    assert "I have a headache." in body


def test_rows_fragment_survives_pydantic_messages(ui):
    """Regression: the preview filter must accept Message models, not just dicts."""
    client, _, _ = ui
    response = client.get(
        "/fragments/datasets/ruslanmv/ai-medical-chatbot/rows",
        params={"config": "default", "split": "train"},
    )

    assert response.status_code == 200
    assert "TypeError" not in response.text
    assert "AttributeError" not in response.text


def test_rows_fragment_reports_huggingface_failure_as_markup(ui):
    client, fake_hf, _ = ui
    from app.services.huggingface import HFDatasetsError

    fake_hf.fail_with = HFDatasetsError("nope", status_code=404)
    response = client.get("/fragments/datasets/some/dataset/rows")

    assert response.status_code == 404
    assert "alert-error" in response.text


# --------------------------------------------------------------------------
# analysis over SSE, as HTML fragments
# --------------------------------------------------------------------------
def test_analyze_events_streams_html_fragments(ui):
    client, _, _ = ui
    response = client.post(
        "/fragments/analyze",
        json={
            "model": "MedGemma:4b",
            "lens": "risk_audit",
            "messages": [{"role": "user", "content": "I have chest pain."}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    assert "event: fragment" in text
    assert "event: meta" in text
    assert "event: done" in text


def test_analyze_events_never_emit_a_bare_newline_in_a_frame(ui):
    """Each fragment frame must be exactly one `data:` line, or framing breaks."""
    client, _, _ = ui
    response = client.post(
        "/fragments/analyze",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    lines = response.text.split("\n")
    for position, line in enumerate(lines):
        if line == "event: fragment":
            assert lines[position + 1].startswith("data: ")
            # The next line after the payload must be blank (frame separator).
            assert lines[position + 2] == ""


def _fragment_payloads(response_text: str) -> list[str]:
    """Every `fragment` payload, in order, as the browser would receive them."""
    payloads: list[str] = []
    lines = response_text.split("\n")
    for position, line in enumerate(lines):
        if line == "event: fragment":
            payloads.append(lines[position + 1][len("data: ") :])
    return payloads


def test_last_fragment_is_the_whole_reply_not_a_delta(ui):
    """The client REPLACES the panel, so the final payload must be complete.

    A regression here is invisible in a single-frame test and shows up only as
    the panel displaying the last chunk of a long analysis.
    """
    client, _, fake_ollama = ui
    fake_ollama.chunks = [
        {"message": {"content": "First part. "}, "done": False},
        {"message": {"content": "Second part. "}, "done": False},
        {"message": {"content": "Third part."}, "done": False},
        {"done": True, "eval_count": 3, "total_duration": 1},
    ]

    response = client.post(
        "/fragments/analyze",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    payloads = _fragment_payloads(response.text)
    assert payloads, "expected at least one fragment"

    final = payloads[-1].replace("\\n", "")
    assert "First part." in final
    assert "Second part." in final
    assert "Third part." in final


def test_fragments_are_monotonically_growing(ui):
    """Each payload must contain everything the previous one did."""
    client, _, fake_ollama = ui
    # Long enough to cross the flush threshold several times.
    chunk = "palavra " * 20
    fake_ollama.chunks = [
        {"message": {"content": f"parte{i} {chunk}"}, "done": False} for i in range(6)
    ] + [{"done": True, "eval_count": 6, "total_duration": 1}]

    response = client.post(
        "/fragments/analyze",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    payloads = [p.replace("\\n", "") for p in _fragment_payloads(response.text)]
    assert len(payloads) >= 2, "expected several flushes for a long reply"

    for earlier, later in zip(payloads, payloads[1:]):
        earlier_text = re.sub(r"<[^>]+>", "", earlier)
        later_text = re.sub(r"<[^>]+>", "", later)
        assert earlier_text.strip() in later_text, (
            "a later fragment dropped earlier content — the panel would lose text"
        )


def test_rendered_reply_contains_real_html_elements(ui):
    """Markdown must arrive as elements, not as escaped raw syntax."""
    client, _, fake_ollama = ui
    fake_ollama.chunks = [
        {"message": {"content": "## Summary\n\n"}, "done": False},
        {"message": {"content": "1. First finding\n"}, "done": False},
        {"message": {"content": "2. **Second finding**\n"}, "done": False},
        {"done": True, "eval_count": 3, "total_duration": 1},
    ]

    response = client.post(
        "/fragments/analyze",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    payloads = _fragment_payloads(response.text)
    final = payloads[-1].replace("\\n", "")

    assert "<h2" in final
    assert "<ol" in final
    assert "<li>" in final
    assert "<strong>Second finding</strong>" in final
    # No raw markdown syntax leaked into the HTML.
    assert "##" not in final
    assert "**" not in final


def test_analyze_events_escapes_html_in_model_output(ui):
    """Model text is untrusted: it must arrive escaped, never as live markup."""
    client, _, fake_ollama = ui
    fake_ollama.chunks = [
        {"message": {"content": "<script>alert(1)</script>"}, "done": False},
        {"done": True, "eval_count": 1, "total_duration": 1},
    ]

    response = client.post(
        "/fragments/analyze",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert "&lt;script&gt;" in response.text
    assert "<script>alert(1)</script>" not in response.text


def test_analyze_events_requires_a_transcript(ui):
    client, _, _ = ui
    assert client.post("/fragments/analyze", json={}).status_code == 422


def test_analyze_events_reports_model_errors(ui):
    client, _, fake_ollama = ui

    async def boom(*args, **kwargs):
        yield {"error": "model 'nope' not found"}

    fake_ollama.chat_stream = boom
    response = client.post(
        "/fragments/analyze",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert "event: error" in response.text
    assert "not found" in response.text


# --------------------------------------------------------------------------
# template filter
# --------------------------------------------------------------------------
def test_first_user_content_prefers_the_user_turn():
    messages = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "the real question"},
    ]
    assert first_user_content(messages) == "the real question"


def test_first_user_content_accepts_pydantic_messages():
    from app.schemas import Message

    messages = [Message(role="user", content="a pydantic question")]
    assert first_user_content(messages) == "a pydantic question"


def test_first_user_content_handles_empty_and_truncates():
    assert first_user_content([]) == ""
    assert first_user_content([{"role": "user", "content": "x" * 500}]).endswith("…")


def test_first_user_content_collapses_whitespace():
    messages = [{"role": "user", "content": "line one\n\n  line two"}]
    assert first_user_content(messages) == "line one line two"
