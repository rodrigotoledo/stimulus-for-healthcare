"""Tests for analysis output language (pt-BR by default).

The language is a property of the whole prompt — guardrail, instructions,
transcript labels and the final language rule — so these assert on the prompt
that actually reaches Ollama, not just on a config value.
"""

from __future__ import annotations

import pytest

from app.lenses import (
    DEFAULT_LANGUAGE,
    get_lens,
    language_catalog,
    lens_catalog,
    normalize_language,
)
from app.schemas import AnalyzeRequest, Message
from app.services.analysis import build_chat_messages, render_transcript

TRANSCRIPT = [
    Message(role="user", content="I have a headache."),
    Message(role="assistant", content="Take ibuprofen."),
]


def _prompt(**kwargs) -> str:
    request = AnalyzeRequest(messages=TRANSCRIPT, **kwargs)
    parts = build_chat_messages(request)
    return "\n".join(part["content"] for part in parts)


# --------------------------------------------------------------------------
# language resolution
# --------------------------------------------------------------------------
def test_default_language_is_portuguese():
    assert DEFAULT_LANGUAGE == "pt-BR"


@pytest.mark.parametrize(
    "value", ["pt", "pt-BR", "pt_br", "PTBR", "português", "portugues", "portuguese"]
)
def test_portuguese_spellings_normalize(value):
    assert normalize_language(value) == "pt-BR"


@pytest.mark.parametrize("value", ["en", "en-US", "english", "inglês"])
def test_english_spellings_normalize(value):
    assert normalize_language(value) == "en"


@pytest.mark.parametrize("value", ["", None, "klingon", "  "])
def test_unknown_language_falls_back_to_the_default(value):
    assert normalize_language(value) == "pt-BR"


# --------------------------------------------------------------------------
# the prompt actually written for Ollama
# --------------------------------------------------------------------------
def test_default_request_produces_a_portuguese_prompt():
    prompt = _prompt()

    # The instruction that governs the answer language, in Portuguese.
    assert "Escreva toda a sua resposta em português do Brasil." in prompt
    assert "Write your entire answer in English." not in prompt


def test_portuguese_prompt_uses_portuguese_instructions():
    prompt = _prompt(lens="clinical_review")

    assert "Revise a conversa e apresente" in prompt
    assert "Review the conversation and report" not in prompt


def test_portuguese_prompt_uses_portuguese_transcript_labels():
    prompt = _prompt()

    assert "TRANSCRIÇÃO" in prompt
    assert "--- Turno 1 · PACIENTE / USUÁRIO ---" in prompt
    assert "TRANSCRIPT" not in prompt
    assert "PATIENT / USER" not in prompt


def test_transcript_content_is_never_translated():
    """Only the reply language changes; the source dialogue stays verbatim."""
    prompt = _prompt()

    assert "I have a headache." in prompt
    assert "Take ibuprofen." in prompt


def test_language_rule_is_the_last_instruction():
    """It must come after the English transcript to outweigh it."""
    prompt = _prompt()
    rule = "Escreva toda a sua resposta em português do Brasil."

    assert prompt.rstrip().endswith(rule)


def test_english_still_available_and_produces_english_prompt():
    prompt = _prompt(language="en")

    assert "Write your entire answer in English." in prompt
    assert "Review the conversation and report" in prompt
    assert "TRANSCRIPT" in prompt
    assert "Escreva toda a sua resposta" not in prompt


def test_guardrail_follows_the_language():
    pt = build_chat_messages(AnalyzeRequest(messages=TRANSCRIPT))
    en = build_chat_messages(AnalyzeRequest(messages=TRANSCRIPT, language="en"))

    assert "revisor clínico criterioso" in pt[0]["content"]
    assert "careful clinical reviewer" in en[0]["content"]


def test_explicit_system_prompt_still_wins():
    parts = build_chat_messages(
        AnalyzeRequest(messages=TRANSCRIPT, system="Custom system text.")
    )

    assert parts[0]["content"] == "Custom system text."


def test_every_lens_has_a_portuguese_variant():
    for entry in lens_catalog("pt-BR"):
        assert entry["title"], entry
        # A Portuguese title should not be the English one, except for the
        # deliberately language-neutral SOAP note.
        if entry["id"] != "soap_note":
            assert entry["title"] != get_lens(entry["id"], "en")["title"], entry["id"]


def test_lens_titles_are_localized():
    pt = {entry["id"]: entry["title"] for entry in lens_catalog("pt-BR")}
    en = {entry["id"]: entry["title"] for entry in lens_catalog("en")}

    assert pt["clinical_review"] == "Revisão clínica"
    assert pt["risk_audit"] == "Auditoria de risco"
    assert pt["plain_summary"] == "Resumo em linguagem simples"
    assert en["clinical_review"] == "Clinical review"


def test_language_catalog_exposes_both_languages():
    catalog = {entry["id"]: entry["label"] for entry in language_catalog()}

    assert catalog["pt-BR"] == "Português (Brasil)"
    assert catalog["en"] == "English"


def test_every_lens_renders_in_every_language_without_key_errors():
    for language in ("pt-BR", "en"):
        for entry in lens_catalog(language):
            lens = get_lens(entry["id"], language)
            assert lens["instructions"].strip()
            assert lens["system"].strip()
            assert lens["language_rule"].strip()


def test_render_transcript_defaults_to_portuguese_labels():
    text = render_transcript(TRANSCRIPT)

    assert "Turno 1" in text and "Turno 2" in text
    assert "PACIENTE / USUÁRIO" in text
    assert "PROFISSIONAL / ASSISTENTE" in text


def test_soap_note_uses_the_localized_not_stated_phrase():
    prompt = _prompt(lens="soap_note")

    assert "não informado" in prompt
    assert "not stated" not in prompt
