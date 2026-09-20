"""Fixtures: real dataset rows captured from the datasets-server API.

Every row below was copied verbatim (truncated) from a live
``https://datasets-server.huggingface.co/rows`` response so that the
normalizer tests reflect reality rather than an invented schema.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# --- lavita/medical-qa-datasets · chatdoctor-icliniq / test (row 0) --------
LAVITA_ROW = {
    "input": (
        "Hello doctor,I had mumps five months ago and after that, I started to "
        "have an infection in my left testes."
    ),
    "answer_icliniq": (
        "Hello, Welcome to Chat Doctor forum. I can understand your concern. "
        "You had mumps and this is a viral infection known to cause an "
        "inflammation of the testis in some cases."
    ),
    "answer_chatgpt": (
        "It is possible that you still have an infection in your testes, "
        "especially if you have been experiencing pain."
    ),
    "answer_chatdoctor": (
        "1) There is no need to worry about reinfection as the infection "
        "caused by Mumps virus does not persist in the body."
    ),
}

# --- heliosbrahma/mental_health_chatbot_dataset (row 0) -------------------
HELIOS_ROW = {
    "text": (
        "<HUMAN>: What is a panic attack?\n"
        "<ASSISTANT>: Panic attacks come on suddenly and involve intense and "
        "often overwhelming fear. They're accompanied by very challenging "
        "physical symptoms, like a racing heartbeat, shortness of breath."
    ),
}

# --- ruslanmv/ai-medical-chatbot (row 0) ----------------------------------
RUSLANMV_ROW = {
    "Description": "Q. What does abutment of the nerve root mean?",
    "Patient": (
        "Hi doctor,I am just wondering what is abutting and abutment of the "
        "nerve root means in a back issue. Please explain."
    ),
    "Doctor": (
        "Hi. I have gone through your query with diligence and would like you "
        "to know that I am here to help you."
    ),
}

# --- medalpaca/medical_meadow_mediqa (row 0) ------------------------------
MEDALPACA_ROW = {
    "output": (
        "Abetalipoproteimemia, also known as Bassen-Kornzweig syndrome, is "
        "diagnosed using blood tests for Apolipoprotein B."
    ),
    "input": (
        "Bassen-Kornzweig syndrome (Exams and Tests): There may be damage to "
        "the retina of the eye (retinitis pigmentosa)."
    ),
    "instruction": (
        "abetalipoproteimemia hi, I would like to know if there is any support "
        "for those suffering with abetalipoproteinemia?"
    ),
}

# --- FreedomIntelligence/medical-o1-reasoning-SFT (row 0) -----------------
MEDICAL_O1_ROW = {
    "Question": (
        "Given the symptoms of sudden weakness in the left arm and leg, recent "
        "long-distance travel, what specific cardiac abnormality is most likely?"
    ),
    "Complex_CoT": (
        "Okay, let's see what's going on here. We've got sudden weakness in the "
        "person's left arm and leg - and that screams something neuro-related."
    ),
    "Response": (
        "The specific cardiac abnormality most likely to be found is a patent "
        "foramen ovale (PFO)."
    ),
}

# --- a ShareGPT-shaped row (not in the curated list, valid via search) ----
SHAREGPT_ROW = {
    "conversations": [
        {"from": "human", "value": "I have had a headache for three days."},
        {"from": "gpt", "value": "Is the pain one-sided or on both sides?"},
        {"from": "human", "value": "One-sided, on the right."},
        {"from": "gpt", "value": "Please seek care if it worsens suddenly."},
    ]
}


@pytest.fixture
def lavita_row() -> dict:
    return dict(LAVITA_ROW)


@pytest.fixture
def helios_row() -> dict:
    return dict(HELIOS_ROW)


@pytest.fixture
def ruslanmv_row() -> dict:
    return dict(RUSLANMV_ROW)


@pytest.fixture
def medalpaca_row() -> dict:
    return dict(MEDALPACA_ROW)


@pytest.fixture
def medical_o1_row() -> dict:
    return dict(MEDICAL_O1_ROW)


@pytest.fixture
def sharegpt_row() -> dict:
    return dict(SHAREGPT_ROW)


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """TestClient with external calls disabled by default."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def ui(monkeypatch):
    """(client, fake_hf, fake_ollama) with HuggingFace and Ollama stubbed.

    Shared by every test module that drives the HTTP surface of the UI.
    """
    from test_api import FakeHF, FakeOllama

    fake_hf = FakeHF(
        rows=[
            {"Patient": "I have a headache.", "Doctor": "Since when?"},
            {"Patient": "Three days.", "Doctor": "Any fever?"},
        ]
    )
    fake_ollama = FakeOllama(
        chunks=[
            {"message": {"content": "Verdict: "}, "done": False},
            {"message": {"content": "NEEDS_CAUTION"}, "done": False},
            {"done": True, "eval_count": 9, "total_duration": 2_000_000_000},
        ]
    )

    from app.api import routes
    from app.config import Settings, get_settings
    from app.main import app

    app.dependency_overrides[routes.hf_client] = lambda: fake_hf
    app.dependency_overrides[routes.ollama_client] = lambda: fake_ollama
    app.dependency_overrides[get_settings] = lambda: Settings()

    with TestClient(app) as client:
        yield client, fake_hf, fake_ollama

    app.dependency_overrides.clear()
