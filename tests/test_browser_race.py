"""Browser tests for the stale-response guard and error routing in dialog-explorer.

Rapid dataset switching fires overlapping `rows` requests. Without independent
per-channel sequence numbers the loser of that race overwrites the winner with
the wrong dataset's dialogs — invisible to fragment-level tests, because every
individual request still returns 200.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
import urllib.request

import pytest

pytest.importorskip("playwright", reason="playwright not installed")

from playwright.sync_api import expect  # noqa: E402

import uvicorn  # noqa: E402

from app.api import routes  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.main import app as real_app  # noqa: E402


class SlowFirstRowsHF:
    """Delay the first rows request so a later one can overtake it."""

    def __init__(self) -> None:
        self.rows_requests = 0

    async def list_splits(self, dataset):
        return {
            "splits": [{"config": "default", "split": "train", "num_rows": 10}],
            "pending": [],
            "failed": [],
        }

    async def search_datasets(self, query="", limit=25):
        return []

    async def fetch_rows(self, dataset, config, split, offset=0, length=10):
        self.rows_requests += 1
        first = self.rows_requests == 1
        if first:
            # Outlive the request that will be fired right behind it.
            await asyncio.sleep(2.0)

        marker = "SLOWFIRST" if first else f"FAST{dataset.split('/')[0]}"
        return {
            "rows": [{"row": {"Patient": f"{marker} question", "Doctor": f"{marker} answer"}}],
            "num_rows_total": 10,
            "truncated": False,
        }


class ExplodingRowsHF(SlowFirstRowsHF):
    """Succeed for the first dataset, fail for the second."""

    async def fetch_rows(self, dataset, config, split, offset=0, length=10):
        self.rows_requests += 1
        if "ruslanmv" in dataset:
            from app.services.huggingface import HFDatasetsError

            raise HFDatasetsError("simulated upstream failure", status_code=502)
        return await super().fetch_rows(dataset, config, split, offset, length)


class StubOllama:
    async def list_models(self):
        return []

    async def chat_stream(self, *args, **kwargs):
        yield {"done": True, "eval_count": 0}


def _serve(stub):
    real_app.dependency_overrides[routes.hf_client] = lambda: stub
    real_app.dependency_overrides[routes.ollama_client] = lambda: StubOllama()
    real_app.dependency_overrides[get_settings] = lambda: Settings()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(real_app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("uvicorn did not start")
    return base, server, thread


@pytest.fixture(scope="module")
def slow_server():
    stub = SlowFirstRowsHF()
    base, server, thread = _serve(stub)
    yield base, stub
    server.should_exit = True
    thread.join(timeout=10)
    real_app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def failing_server():
    stub = ExplodingRowsHF()
    base, server, thread = _serve(stub)
    yield base, stub
    server.should_exit = True
    thread.join(timeout=10)
    real_app.dependency_overrides.clear()


def test_slow_response_cannot_clobber_a_newer_selection(page, slow_server):
    """Switch datasets twice quickly; the second selection must win the DOM."""
    base, stub = slow_server

    page.goto(base)
    page.wait_for_function("() => !!window.Stimulus")

    page.locator('[data-dataset-id="lavita/medical-qa-datasets"]').click()
    # Let the slow request get in flight, then overtake it.
    page.wait_for_timeout(150)
    page.locator('[data-dataset-id="ruslanmv/ai-medical-chatbot"]').click()

    expect(page.locator("[data-dialog-index]")).to_have_count(1)
    # Wait past the slow response so a late write would have landed.
    page.wait_for_timeout(2500)

    body = page.locator("[data-dialog-explorer-target=body]").inner_text()
    assert "FASTruslanmv" in body, f"newer selection missing from DOM: {body[:200]!r}"
    assert "SLOWFIRST" not in body, "the superseded slow response overwrote the DOM"
    assert stub.rows_requests >= 2


def test_failed_rows_request_reports_into_the_rows_region(page, failing_server):
    """An upstream failure must not be written into the splits picker region."""
    base, _ = failing_server

    page.goto(base)
    page.wait_for_function("() => !!window.Stimulus")

    page.locator('[data-dataset-id="ruslanmv/ai-medical-chatbot"]').click()

    body = page.locator("[data-dialog-explorer-target=body]")
    expect(body).to_contain_text("simulated upstream failure")

    # The pickers must still be intact and usable.
    expect(page.locator("#config-select")).to_be_visible()
    expect(page.locator("#split-select")).to_be_visible()
