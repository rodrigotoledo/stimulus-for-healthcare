"""Tests for the row -> transcript normalizer, using real dataset rows."""

from __future__ import annotations

import pytest

from app.services.dialog import (
    NormalizationError,
    normalize_row,
    normalize_rows,
    parse_marked_text,
)


def test_lavita_keeps_every_parallel_answer(lavita_row):
    result = normalize_row(lavita_row, "lavita/medical-qa-datasets")

    assert result["strategy"] == "explicit_mapping"
    assert [m["role"] for m in result["messages"]] == [
        "user",
        "assistant",
        "assistant",
        "assistant",
    ]
    labels = [m["label"] for m in result["messages"]]
    assert labels == [None, "icliniq", "chatgpt", "chatdoctor"]
    assert "mumps" in result["messages"][0]["content"]


def test_medical_o1_surfaces_the_reasoning_trace(medical_o1_row):
    result = normalize_row(medical_o1_row)

    assert result["strategy"] == "explicit_mapping"
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "reasoning", "assistant"]
    assert result["messages"][1]["label"] == "complex_cot"
    assert "foramen ovale" in result["messages"][2]["content"]


def test_helios_parses_role_markers(helios_row):
    result = normalize_row(helios_row)

    assert result["strategy"] == "role_markers"
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]
    assert result["messages"][0]["content"] == "What is a panic attack?"
    assert "Panic attacks come on suddenly" in result["messages"][1]["content"]


def test_parse_marked_text_handles_multiple_turns_and_preface():
    text = (
        "Some preamble.\n"
        "<HUMAN>: first question\n"
        "<ASSISTANT>: first answer\n"
        "<HUMAN>: follow up\n"
        "<ASSISTANT>: second answer\n"
    )
    messages = parse_marked_text(text)

    assert [m["role"] for m in messages] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == "Some preamble."
    assert messages[1]["content"] == "first question"
    assert messages[4]["content"] == "second answer"


def test_parse_marked_text_rejects_a_single_block():
    # Only one turn -> not a dialogue, so another strategy should win.
    assert parse_marked_text("<HUMAN>: just one line") == []
    assert parse_marked_text("no markers at all") == []


def test_ruslanmv_maps_patient_and_doctor(ruslanmv_row):
    """`Description` is a reworded title, not a quote, so it survives as extra."""
    result = normalize_row(ruslanmv_row)

    assert result["strategy"] == "role_columns"
    user_turns = [m for m in result["messages"] if m["role"] == "user"]
    assert len(user_turns) == 1, "the secondary 'Description' column must not add a turn"
    assert user_turns[0]["label"] == "Patient"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["label"] == "Doctor"
    assert result["extra"]["Description"] == "Q. What does abutment of the nerve root mean?"


def test_metadata_duplicated_in_the_transcript_is_dropped():
    """A metadata column that repeats a turn verbatim becomes a noisy chip."""
    row = {
        "Patient": "I have had a headache for three days.",
        "Doctor": "Since when exactly?",
        "topic": "Since when exactly?",
        "department": "neurology",
    }
    result = normalize_row(row)

    assert "topic" not in result["extra"]
    assert result["extra"]["department"] == "neurology"


def test_metadata_is_read_case_insensitively():
    row = {"Patient": "hi", "Doctor": "hello", "SOURCE": "icliniq"}
    result = normalize_row(row)

    assert result["extra"]["SOURCE"] == "icliniq"


def test_metadata_columns_never_become_assistant_turns():
    """`Description` is a title, not an answer, so it must not be a reply."""
    row = {
        "Description": "Q. What does abutment of the nerve root mean?",
        "Patient": "What does abutting of the nerve root mean in a back issue?",
        "Doctor": "I have gone through your query with diligence.",
    }
    result = normalize_row(row)

    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]
    assert result["messages"][1]["label"] == "Doctor"
    assert result["extra"]["Description"].startswith("Q. What does abutment")


def test_identifier_columns_never_become_assistant_turns():
    """Live datasets expose `qtype` / `unique_id` columns — not answers."""
    row = {
        "question": "What is the treatment for hypertension?",
        "answer": "Lifestyle changes and antihypertensives.",
        "qtype": "treatment",
        "unique_id": 4242,
    }
    result = normalize_row(row)

    assert [m["label"] for m in result["messages"]] == ["question", "answer"]


def test_num_turns_counts_parallel_answers_as_one_turn(lavita_row):
    """Three answers to one question is a single turn, not three."""
    result = normalize_row(lavita_row)

    assert len(result["messages"]) == 4
    assert result["num_turns"] == 1


def test_medalpaca_prefers_instruction_over_input(medalpaca_row):
    result = normalize_row(medalpaca_row)

    user_turns = [m for m in result["messages"] if m["role"] == "user"]
    assert len(user_turns) == 1
    assert user_turns[0]["label"] == "instruction"
    assert "abetalipoproteimemia" in user_turns[0]["content"]
    assert result["strategy"] == "role_columns"


def test_sharegpt_message_list_is_understood(sharegpt_row):
    result = normalize_row(sharegpt_row)

    assert result["strategy"] == "message_list"
    assert [m["role"] for m in result["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "three days" in result["messages"][0]["content"]


def test_message_list_as_pairs():
    row = {"messages": [["user", "hello"], ["assistant", "hi there"]]}
    result = normalize_row(row)

    assert result["strategy"] == "message_list"
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]


def test_single_text_column_still_produces_something():
    row = {"content": "A long clinical note about an unspecified presentation."}
    result = normalize_row(row)

    assert result["strategy"] == "single_text"
    assert result["messages"][0]["role"] == "user"


def test_empty_row_raises():
    with pytest.raises(NormalizationError):
        normalize_row({})


def test_normalize_rows_indexes_and_reports_warnings():
    rows = [{"Patient": "hi", "Doctor": "hello"}, {}]
    dialogs, warnings = normalize_rows(rows, offset=20)

    assert [d["index"] for d in dialogs] == [20]
    assert warnings and warnings[0].startswith("row 21:")


def test_unparseable_rows_fall_back_to_raw_inspection():
    rows = [{"a": 1, "b": 2}]
    dialogs, warnings = normalize_rows(rows)

    assert len(dialogs) == 1
    assert dialogs[0]["messages"][0]["label"] == "raw"
    assert any("raw rows" in warning for warning in warnings)


def test_long_messages_are_truncated():
    row = {"Patient": "x" * 10_000, "Doctor": "ok"}
    result = normalize_row(row)

    assert len(result["messages"][0]["content"]) < 10_000
    assert result["messages"][0]["content"].endswith("[truncated]")
