"""Browser tests for the Stimulus UI (the one gap the curl checks left open).

These drive a real Chromium against a real uvicorn process, so they exercise
what nothing else did: Stimulus wiring up, `window` events crossing columns,
DOM swaps landing, and fragments appending as the SSE stream arrives.

HuggingFace and Ollama are stubbed at the HTTP boundary with a tiny in-process
ASGI app, so the tests are hermetic and fast but still go through the browser's
own fetch and event-source plumbing.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("playwright", reason="playwright not installed")

from playwright.sync_api import expect  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import Response  # noqa: E402

from app.api import routes  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.main import app as real_app  # noqa: E402


# --------------------------------------------------------------------------
# stub upstream services
# --------------------------------------------------------------------------
HF_ROWS = [
    {"Patient": "I have had a headache for three days.", "Doctor": "Since when?"},
    {"Patient": "Three days now.", "Doctor": "Any fever or neck stiffness?"},
]

SPLITS = {
    "splits": [
        {"config": "default", "split": "train", "num_rows": 172},
        {"config": "default", "split": "test", "num_rows": 20},
    ],
    "pending": [],
    "failed": [],
}


class StubHF:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def search_datasets(self, query="", limit=25):
        self.calls.append(("search", query))
        return [
            {
                "id": "stub/medical-dialogues",
                "downloads": 4242,
                "likes": 7,
                "tags": ["medical"],
                "gated": False,
            }
        ]

    async def list_splits(self, dataset):
        self.calls.append(("splits", dataset))
        return SPLITS

    async def fetch_rows(self, dataset, config, split, offset=0, length=10):
        self.calls.append(("rows", dataset, config, split, offset, length))
        return {
            "rows": [{"row": row} for row in HF_ROWS[offset : offset + length]],
            "num_rows_total": 172,
            "truncated": False,
        }


class StubOllama:
    """Streams a few tokens, then done — enough to prove fragments append."""

    async def list_models(self):
        return [
            {
                "name": "StubModel:1b",
                "size": 1,
                "family": "stub",
                "parameter_size": "1B",
                "quantization_level": "F16",
                "capabilities": ["completion"],
                "modified_at": "2026-01-01T00:00:00Z",
            }
        ]

    async def chat_stream(self, model, messages, options=None, keep_alive=None):
        for token in ("Verdict: ", "NEEDS_", "CAUTION"):
            yield {"message": {"content": token}, "done": False}
        yield {"done": True, "eval_count": 3, "total_duration": 1_000_000_000}


# --------------------------------------------------------------------------
# server fixture
# --------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Run the real app under uvicorn with HF/Ollama stubbed, in a thread."""
    stub_hf = StubHF()

    real_app.dependency_overrides[routes.hf_client] = lambda: stub_hf
    real_app.dependency_overrides[routes.ollama_client] = lambda: StubOllama()
    real_app.dependency_overrides[get_settings] = lambda: Settings(
        ollama_default_model="StubModel:1b"
    )

    port = _free_port()
    config = uvicorn.Config(
        real_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    import urllib.request

    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("uvicorn did not start")

    yield base, stub_hf

    server.should_exit = True
    thread.join(timeout=10)
    real_app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def app_url(live_server):
    return live_server[0]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_stimulus_boots_and_registers_its_controllers(page, app_url):
    """Stimulus must load (from the CDN importmap) and connect every controller."""
    page.goto(app_url)

    # The controllers connect on load; window.Stimulus is exposed by app.js.
    page.wait_for_function("() => !!window.Stimulus")
    connected = page.evaluate(
        "() => window.Stimulus.router.modules.map(m => m.identifier).sort()"
    )
    assert connected == [
        "analysis-stream",
        "dataset-browser",
        "dialog-explorer",
        "status",
    ]


def test_page_renders_three_columns_without_javascript_errors(page, app_url):
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    page.goto(app_url)
    page.wait_for_function("() => !!window.Stimulus")

    expect(page.locator("h1")).to_have_text("Seletor de Diálogos HuggingFace")
    # The server-rendered curated list is in the first paint.
    expect(page.locator("[data-dataset-id]")).to_have_count(8)
    assert errors == [], f"javascript errors on load: {errors}"


def test_status_controller_shows_ollama_state(page, app_url):
    page.goto(app_url)

    badge = page.locator('[data-status-target="badge"]')
    expect(badge).to_contain_text("modelos")


def test_selecting_a_dataset_loads_splits_and_rows_in_the_browser(page, app_url):
    """The cross-column window event must actually drive the dialog column."""
    page.goto(app_url)
    page.wait_for_function("() => !!window.Stimulus")

    page.locator('[data-dataset-id="lavita/medical-qa-datasets"]').click()

    # dialog-explorer swapped in the splits fragment...
    expect(page.locator("#config-select")).to_be_visible()
    expect(page.locator("#split-select")).to_be_visible()
    expect(page.locator("#split-select")).to_contain_text("train")

    # ...and the rows fragment, which renders dialog cards.
    expect(page.locator("[data-dialog-index]")).to_have_count(2)


def test_selecting_a_dialog_renders_the_transcript_and_arms_the_button(page, app_url):
    page.goto(app_url)
    page.wait_for_function("() => !!window.Stimulus")

    page.locator('[data-dataset-id="lavita/medical-qa-datasets"]').click()
    expect(page.locator("[data-dialog-index]")).to_have_count(2)

    page.locator('[data-dialog-index="0"]').first.click()

    # The transcript comes back with the rows fragment, so the transcript is
    # the real signal that the selection round-trip finished.
    transcript = page.locator("[data-dialog-explorer-target=transcript]")
    expect(transcript).to_contain_text("Paciente / Usuário")
    expect(transcript).to_contain_text("I have had a headache for three days.")
    expect(transcript).to_contain_text("Profissional / Assistente")

    # analysis-stream received dialog:selected and enabled the run button.
    run = page.locator('[data-analysis-stream-target="runButton"]')
    expect(run).to_be_enabled()


def test_running_an_analysis_streams_fragments_into_the_dom(page, app_url):
    """The end-to-end payoff: real SSE, appended as server-rendered HTML."""
    page.goto(app_url)
    page.wait_for_function("() => !!window.Stimulus")

    page.locator('[data-dataset-id="lavita/medical-qa-datasets"]').click()
    expect(page.locator("[data-dialog-index]")).to_have_count(2)
    page.locator('[data-dialog-index="0"]').first.click()
    expect(
        page.locator("[data-dialog-explorer-target=transcript]")
    ).to_contain_text("Paciente / Usuário")

    page.locator('[data-analysis-stream-target="runButton"]').click()

    output = page.locator('[data-analysis-stream-target="output"]')
    expect(output).to_contain_text("Verdict:", timeout=15000)
    expect(output).to_contain_text("NEEDS_CAUTION", timeout=15000)

    # The final meta line arrives on the done event.
    expect(page.locator('[data-analysis-stream-target="meta"]')).to_contain_text(
        "tokens", timeout=15000
    )

def test_search_tab_swaps_in_hub_results(page, app_url):
    page.goto(app_url)
    page.wait_for_function("() => !!window.Stimulus")

    page.locator('[data-tab="search"]').click()
    search = page.locator('[data-dataset-browser-target="query"]')
    expect(search).to_be_visible()
    search.fill("medical dialogue")
    page.locator('form[data-dataset-browser-target="searchForm"] button').click()

    expect(page.locator('[data-dataset-id="stub/medical-dialogues"]')).to_be_visible()


def test_switching_datasets_twice_keeps_working(page, app_url):
    """Guards the URL-template rewriting in dialog-explorer::loadDataset."""
    page.goto(app_url)
    page.wait_for_function("() => !!window.Stimulus")

    page.locator('[data-dataset-id="lavita/medical-qa-datasets"]').click()
    expect(page.locator("[data-dialog-index]")).to_have_count(2)

    page.locator('[data-dataset-id="ruslanmv/ai-medical-chatbot"]').click()
    expect(page.locator("[data-dialog-index]")).to_have_count(2)

    page.locator('[data-dataset-id="lavita/medical-qa-datasets"]').click()
    expect(page.locator("[data-dialog-index]")).to_have_count(2)


def test_no_javascript_errors_during_a_full_interaction(page, app_url):
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    page.goto(app_url)
    page.wait_for_function("() => !!window.Stimulus")
    page.locator('[data-dataset-id="lavita/medical-qa-datasets"]').click()
    expect(page.locator("[data-dialog-index]")).to_have_count(2)
    page.locator('[data-dialog-index="0"]').first.click()
    expect(
        page.locator("[data-dialog-explorer-target=transcript]")
    ).to_contain_text("Paciente / Usuário")
    page.locator('[data-analysis-stream-target="runButton"]').click()
    expect(page.locator('[data-analysis-stream-target="output"]')).to_contain_text(
        "Verdict:", timeout=15000
    )

    assert errors == [], f"javascript errors during interaction: {errors}"
