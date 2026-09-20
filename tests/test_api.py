"""Tests for the HTTP API, with HuggingFace and Ollama stubbed out."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.huggingface import HFDatasetsError


class FakeHF:
    def __init__(self, rows=None, fail_with=None):
        self.rows = rows or []
        self.fail_with = fail_with
        self.calls: list[tuple] = []

    async def search_datasets(self, query="", limit=25):
        self.calls.append(("search", query, limit))
        if self.fail_with:
            raise self.fail_with
        return [
            {
                "id": "some/medical-dialogues",
                "downloads": 1234,
                "likes": 12,
                "tags": ["medical", "conversational"],
                "gated": False,
            }
        ]

    async def list_splits(self, dataset):
        self.calls.append(("splits", dataset))
        if self.fail_with:
            raise self.fail_with
        return {
            "splits": [{"config": "default", "split": "train", "num_rows": 172}],
            "pending": [],
            "failed": [],
        }

    async def fetch_rows(self, dataset, config, split, offset=0, length=10):
        self.calls.append(("rows", dataset, config, split, offset, length))
        if self.fail_with:
            raise self.fail_with
        return {
            "rows": [{"row": row} for row in self.rows],
            "num_rows_total": 172,
            "truncated": False,
        }


class FakeOllama:
    def __init__(self, chunks=None, fail=False):
        self.chunks = chunks or [
            {"message": {"content": "Verdict: "}, "done": False},
            {"message": {"content": "SAFE"}, "done": False},
            {"done": True, "eval_count": 7, "total_duration": 12345},
        ]
        self.fail = fail
        self.sent: list[dict] = []

    async def list_models(self):
        if self.fail:
            raise RuntimeError("ollama is down")
        return [
            {
                "name": "MedGemma:4b",
                "size": 3_300_000_000,
                "family": "gemma",
                "parameter_size": "4B",
                "quantization_level": "Q4_K_M",
                "capabilities": ["completion"],
                "modified_at": "2026-07-01T00:00:00Z",
            }
        ]

    async def chat_stream(self, model, messages, options=None, keep_alive=None):
        self.sent.append({"model": model, "messages": messages, "options": options})
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def api(monkeypatch):
    """Return (client, fake_hf, fake_ollama) with dependencies overridden."""
    fake_hf = FakeHF(rows=[{"Patient": "I have a headache.", "Doctor": "Since when?"}])
    fake_ollama = FakeOllama()

    from app.api import routes

    def hf_override():
        return fake_hf

    def ollama_override():
        return fake_ollama

    app.dependency_overrides[routes.hf_client] = hf_override
    app.dependency_overrides[routes.ollama_client] = ollama_override
    app.dependency_overrides[get_settings] = lambda: Settings()

    with TestClient(app) as client:
        yield client, fake_hf, fake_ollama

    app.dependency_overrides.clear()


def test_health_reports_ollama_state(api):
    client, _, _ = api
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ollama"]["reachable"] is True
    assert "MedGemma:4b" in body["ollama"]["models"]


def test_health_degrades_when_ollama_is_down(monkeypatch):
    from app.api import routes

    app.dependency_overrides[routes.ollama_client] = lambda: FakeOllama(fail=True)
    app.dependency_overrides[get_settings] = lambda: Settings()
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    app.dependency_overrides.clear()

    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False


def test_models_endpoint(api):
    client, _, _ = api
    body = client.get("/api/models").json()

    assert body["default_model"] == "MedGemma:4b"
    assert body["models"][0]["parameter_size"] == "4B"


def test_curated_catalog_is_served_without_network(api):
    client, _, _ = api
    body = client.get("/api/datasets/curated").json()

    ids = [entry["id"] for entry in body]
    assert "lavita/medical-qa-datasets" in ids
    assert "heliosbrahma/mental_health_chatbot_dataset" in ids
    assert all(entry["verified"] for entry in body)


def test_dead_datasets_are_not_offered(api):
    client, _, _ = api
    ids = [entry["id"] for entry in client.get("/api/datasets/curated").json()]

    assert "Amod/mental_health_counseling_conversations" not in ids
    assert "OpenMed/Medical-Dialogue" not in ids


def test_search_marks_curated_hits(api):
    client, fake_hf, _ = api
    body = client.get("/api/datasets/search", params={"q": "medical"}).json()

    assert body["query"] == "medical"
    assert fake_hf.calls[0] == ("search", "medical", 25)
    assert body["results"][0]["curated"] is False


def test_search_surfaces_huggingface_errors(api, monkeypatch):
    client, fake_hf, _ = api
    fake_hf.fail_with = HFDatasetsError("dataset is gated", status_code=401)

    response = client.get("/api/datasets/search", params={"q": "x"})

    assert response.status_code == 401
    assert "gated" in response.json()["detail"]


def test_splits_endpoint(api):
    client, _, _ = api
    body = client.get("/api/datasets/some/dataset/splits").json()

    assert body["dataset"] == "some/dataset"
    assert body["splits"][0] == {"config": "default", "split": "train", "num_rows": 172}


def test_rows_returns_normalized_dialogs(api):
    client, fake_hf, _ = api
    body = client.get(
        "/api/datasets/ruslanmv/ai-medical-chatbot/rows",
        params={"config": "default", "split": "train", "offset": 5, "length": 1},
    ).json()

    assert body["num_rows_total"] == 172
    assert body["dialogs"][0]["index"] == 5
    assert [m["role"] for m in body["dialogs"][0]["messages"]] == ["user", "assistant"]
    assert fake_hf.calls[-1] == (
        "rows",
        "ruslanmv/ai-medical-chatbot",
        "default",
        "train",
        5,
        1,
    )


def test_rows_uses_curated_defaults_when_unspecified(api):
    client, fake_hf, _ = api
    client.get("/api/datasets/lavita/medical-qa-datasets/rows")

    _, dataset, config, split, _, _ = fake_hf.calls[-1]
    assert (dataset, config, split) == (
        "lavita/medical-qa-datasets",
        "chatdoctor-icliniq",
        "test",
    )


def test_rows_length_is_capped(api):
    client, fake_hf, _ = api
    response = client.get(
        "/api/datasets/some/dataset/rows",
        params={"config": "default", "split": "train", "length": 500},
    )

    assert response.status_code == 422  # rejected before it ever reaches HF
    assert fake_hf.calls == []

    client.get(
        "/api/datasets/some/dataset/rows",
        params={"config": "default", "split": "train", "length": 50},
    )
    assert fake_hf.calls[-1][-1] == 50


def test_rows_falls_back_to_first_split(api):
    client, fake_hf, _ = api
    client.get("/api/datasets/some/dataset/rows")

    assert fake_hf.calls[-1][2:] == ("default", "train", 0, 10)


def test_analyze_streams_sse_events(api):
    client, _, fake_ollama = api
    response = client.post(
        "/api/analyze",
        json={
            "model": "MedGemma:4b",
            "dataset": "medalpaca/medical_meadow_mediqa",
            "dialog_index": 3,
            "lens": "risk_audit",
            "messages": [
                {"role": "user", "content": "I have chest pain."},
                {"role": "assistant", "content": "Take an aspirin and rest."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    assert "event: start" in text
    assert "event: chunk" in text
    assert "event: done" in text

    chunks = [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    assert any("SAFE" in chunk.get("content", "") for chunk in chunks)
    assert chunks[-1]["chars"] == len("Verdict: SAFE")


def test_analyze_builds_the_lens_prompt(api):
    client, _, fake_ollama = api
    client.post(
        "/api/analyze",
        json={
            "lens": "soap_note",
            "messages": [{"role": "user", "content": "headache for 3 days"}],
        },
    )

    payload = fake_ollama.sent[-1]
    assert payload["model"] == "MedGemma:4b"
    assert "Nota SOAP" in payload["messages"][1]["content"]
    assert "S:" in payload["messages"][1]["content"]
    assert "headache for 3 days" in payload["messages"][1]["content"]


def test_analyze_requires_a_transcript(api):
    client, _, _ = api
    assert client.post("/api/analyze", json={}).status_code == 422


def test_analyze_reports_ollama_failure_as_an_event(api):
    client, _, fake_ollama = api

    async def boom(*args, **kwargs):
        yield {"error": "model 'nope' not found"}

    fake_ollama.chat_stream = boom
    response = client.post(
        "/api/analyze",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert "event: error" in response.text
    assert "not found" in response.text


def test_lenses_endpoint(api):
    client, _, _ = api
    body = client.get("/api/lenses").json()

    ids = [lens["id"] for lens in body["lenses"]]
    assert body["default"] in ids
    assert "triple_compare" in ids
