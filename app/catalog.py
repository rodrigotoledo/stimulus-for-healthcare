"""Curated catalog of HuggingFace dialog datasets.

Every entry here was probed live against
``https://datasets-server.huggingface.co`` and is known to expose the
rows endpoint without authentication. The ``default_config`` /
``default_split`` values come from that probe, and ``note`` records the
shape of the row so the normalizer decisions are auditable.

Datasets that were checked and are *not* usable (404 on the datasets
server, gated or private) are recorded in ``DEAD_DATASETS`` so nobody
re-adds them by hand.
"""

from __future__ import annotations

from typing import Any

CURATED_DATASETS: list[dict[str, Any]] = [
    {
        "id": "lavita/medical-qa-datasets",
        "note": (
            "ChatDoctor / iCliniq / HealthCareMagic. One patient question with up "
            "to three parallel answers (iCliniq, ChatGPT, ChatDoctor) — use the "
            "'Compare answers' lens."
        ),
        "default_config": "chatdoctor-icliniq",
        "default_split": "test",
        "tags": ["medical", "multi-turn-pairs", "multi-answer"],
    },
    {
        "id": "ruslanmv/ai-medical-chatbot",
        "note": (
            "256k patient/doctor exchanges with a short 'Description' title. "
            "Columns: Description, Patient, Doctor."
        ),
        "default_config": "default",
        "default_split": "train",
        "tags": ["medical", "dialogue"],
    },
    {
        "id": "medalpaca/medical_meadow_mediqa",
        "note": "MEDIQA clinical Q&A. Columns: instruction, input, output.",
        "default_config": "default",
        "default_split": "train",
        "tags": ["medical", "qa"],
    },
    {
        "id": "keivalya/MedQuad-MedicalQnADataset",
        "note": "MedQuAD medical question/answer pairs from NIH sources.",
        "default_config": "default",
        "default_split": "train",
        "tags": ["medical", "qa"],
    },
    {
        "id": "BI55/MedText",
        "note": "Short clinical vignettes paired with an assessment.",
        "default_config": "default",
        "default_split": "train",
        "tags": ["medical", "vignette"],
    },
    {
        "id": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "note": (
            "Question, Complex_CoT (reasoning trace) and Response. Use the "
            "'Reasoning check' lens to audit the trace against the answer."
        ),
        "default_config": "en",
        "default_split": "train",
        "tags": ["medical", "reasoning"],
    },
    {
        "id": "heliosbrahma/mental_health_chatbot_dataset",
        "note": "Mental-health support conversations tagged <HUMAN>/<ASSISTANT>.",
        "default_config": "default",
        "default_split": "train",
        "tags": ["mental-health", "tagged-text"],
    },
    {
        "id": "HPAI-BSC/CareQA",
        "note": "CareQA multiple-choice and open clinical questions (en/es).",
        "default_config": "CareQA_en_open",
        "default_split": "test",
        "tags": ["medical", "qa", "multilingual"],
    },
]

CURATED_BY_ID = {entry["id"]: entry for entry in CURATED_DATASETS}

# Probed and rejected — kept as documentation, never offered in the UI.
DEAD_DATASETS: dict[str, str] = {
    "Amod/mental_health_counseling_conversations": "404 from datasets-server (private/gated)",
    "OpenMed/Medical-Dialogue": "404 from datasets-server (private/gated)",
}

# Fallback role-column pairs, tried in order when a row has no explicit
# mapping and no message-list column. Longest pairs win so that datasets
# with several answer columns prefer the richest mapping.
ROLE_COLUMN_PAIRS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("Patient", "Doctor"), ("user", "assistant")),
    (("patient", "doctor"), ("user", "assistant")),
    # `instruction` beats `input`: in medical_meadow the instruction is the
    # actual question while `input` only carries supporting context.
    (("instruction", "output"), ("user", "assistant")),
    (("question", "answer"), ("user", "assistant")),
    (("Question", "Answer"), ("user", "assistant")),
    (("problem", "answer"), ("user", "assistant")),
    (("input", "output"), ("user", "assistant")),
    (("prompt", "response"), ("user", "assistant")),
    (("query", "response"), ("user", "assistant")),
]

# Role markers found inside single-text-column datasets.
ROLE_MARKERS: dict[str, str] = {
    "human": "user",
    "user": "user",
    "patient": "user",
    "assistant": "assistant",
    "doctor": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "system": "system",
    "reasoning": "reasoning",
    "thought": "reasoning",
}

# Column names that hold a whole conversation as a list of dicts.
MESSAGE_LIST_FIELDS: tuple[str, ...] = (
    "messages",
    "conversations",
    "conversation",
    "dialog",
    "dialogue",
    "turns",
    "chat",
)

# Columns that are metadata rather than conversation content, used to build
# the `extra` bag shown as chips in the UI. Kept deliberately tight: any name
# listed here can never become an assistant turn.
METADATA_FIELDS: tuple[str, ...] = (
    "description",
    "topic",
    "category",
    "department",
    "specialty",
    "title",
    "source",
    "language",
    "lang",
)

# Metadata is also read case-insensitively, so `Description` and
# `description` both land in `extra`.
METADATA_LOOKUP: dict[str, str] = {field.lower(): field for field in METADATA_FIELDS}

# Columns that identify a row rather than describe the conversation.
IGNORED_COLUMN_SUFFIXES = ("_id", "_type", "_name", "qtype")
IGNORED_COLUMNS = {"id", "index", "url", "unique_id", "qid"}


def is_ignored_column(key: str) -> bool:
    lowered = key.lower()
    if lowered in IGNORED_COLUMNS:
        return True
    return lowered.endswith(IGNORED_COLUMN_SUFFIXES)


def curated_entry(dataset_id: str) -> dict[str, Any] | None:
    return CURATED_BY_ID.get(dataset_id)


def default_config_split(dataset_id: str) -> tuple[str | None, str | None]:
    entry = CURATED_BY_ID.get(dataset_id)
    if not entry:
        return None, None
    return entry.get("default_config"), entry.get("default_split")