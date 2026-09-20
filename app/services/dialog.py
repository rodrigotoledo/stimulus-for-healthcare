"""Normalize a HuggingFace dataset row into a chat transcript.

HuggingFace dialog datasets have no common schema, so detection runs in
order of confidence:

1. an explicit per-dataset mapping for one specific dataset id;
2. a *message-list* column (``messages``/``conversations``/...) holding
   ``[{role, content}, ...]`` or ``[[user, text], ...]``;
3. *role markers* inside a single text column (``<HUMAN>: ...``);
4. *role column pairs* inside the row (``Patient``/``Doctor``);
5. a last-resort single user message so the row is still inspectable.

Every function here is pure: it takes plain row data and returns plain
dicts, which is what the API serializes and what the tests exercise.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from ..catalog import (
    MESSAGE_LIST_FIELDS,
    METADATA_LOOKUP,
    ROLE_COLUMN_PAIRS,
    ROLE_MARKERS,
    is_ignored_column,
)

MAX_CHARS_PER_MESSAGE = 6000

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "patient": "user",
    "assistant": "assistant",
    "doctor": "assistant",
    "gpt": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "system": "system",
    "reasoning": "reasoning",
    "thought": "reasoning",
    "complex_cot": "reasoning",
}

# `<HUMAN>:`, `HUMAN:`, `### Human:` at the start of a line.
_MARKER_RE = re.compile(
    r"^\s*(?:#{1,4}\s*)?[<\[\(]{0,2}([A-Za-z_ ]{2,20})[>\]\)]{0,2}\s*:\s*",
    re.MULTILINE,
)


class NormalizationError(ValueError):
    """Raised when a row cannot be turned into a transcript at all."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _trim(text: str) -> str:
    if len(text) <= MAX_CHARS_PER_MESSAGE:
        return text
    return text[:MAX_CHARS_PER_MESSAGE] + " …[truncated]"


def _message(role: str, content: Any, label: str | None = None) -> dict[str, Any] | None:
    text = _trim(_clean(content))
    if not text:
        return None
    return {"role": role, "content": text, "label": label}


def _normalize_role(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower().replace(" ", "_")
    return ROLE_ALIASES.get(key) or ROLE_MARKERS.get(key)


def _sequences(value: Any) -> list[Any] | None:
    """Return ``value`` as a list if it looks like a transcript sequence."""
    if isinstance(value, list) and value:
        return value
    return None


# --------------------------------------------------------------------------
# strategy 1 — explicit per-dataset mapping
# --------------------------------------------------------------------------
def _explicit_mapping(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Dataset-specific handling for the shapes seen in the curated catalog.

    ``input`` + three parallel answers (lavita/medical-qa-datasets) becomes
    one user turn and three labelled assistant answers.
    """
    keys = set(row)
    answers = [
        key
        for key in ("answer_icliniq", "answer_chatgpt", "answer_chatdoctor")
        if key in keys
    ]
    if "input" in keys and answers:
        messages: list[dict[str, Any]] = []
        ask = _message("user", row.get("input"))
        if ask:
            messages.append(ask)
        for key in answers:
            label = key.replace("answer_", "")
            reply = _message("assistant", row.get(key), label=label)
            if reply:
                messages.append(reply)
        return messages or None

    # A reasoning trace is surfaced as its own role so the UI can style it.
    if "Complex_CoT" in keys and ("Question" in keys or "Response" in keys):
        messages = []
        question = _message("user", row.get("Question"))
        if question:
            messages.append(question)
        cot = _message("reasoning", row.get("Complex_CoT"), label="complex_cot")
        if cot:
            messages.append(cot)
        answer = _message("assistant", row.get("Response"))
        if answer:
            messages.append(answer)
        return messages or None

    return None


# --------------------------------------------------------------------------
# strategy 2 — a column holding a list of turns
# --------------------------------------------------------------------------
def _from_message_list(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str] | None:
    for field in MESSAGE_LIST_FIELDS:
        if field not in row:
            continue
        items = _sequences(row[field])
        if not items:
            continue

        messages: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                role = (
                    _normalize_role(item.get("role"))
                    or _normalize_role(item.get("from"))
                    or _normalize_role(item.get("speaker"))
                    or _normalize_role(item.get("author"))
                )
                content = item.get("content")
                if content is None:
                    content = item.get("value")
                if content is None:
                    content = item.get("text")
                if role is None or content is None:
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                role = _normalize_role(item[0])
                content = item[1]
            else:
                continue

            message = _message(role or "user", content)
            if message:
                messages.append(message)

        if messages:
            return messages, field
    return None


# --------------------------------------------------------------------------
# strategy 3 — role markers inside a text blob
# --------------------------------------------------------------------------
def parse_marked_text(text: str) -> list[dict[str, Any]]:
    """Split ``<HUMAN>: ... <ASSISTANT>: ...`` into turns."""
    text = _clean(text)
    if not text:
        return []

    matches = [m for m in _MARKER_RE.finditer(text) if _normalize_role(m.group(1))]
    if not matches:
        return []

    messages: list[dict[str, Any]] = []

    # Anything before the first marker is unattributed preface.
    preface = text[: matches[0].start()].strip()
    if preface:
        messages.append({"role": "user", "content": _trim(preface), "label": None})

    for position, match in enumerate(matches):
        role = _normalize_role(match.group(1)) or "user"
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        message = _message(role, text[start:end])
        if message:
            messages.append(message)

    # A single labelled block is probably not a dialogue — let other
    # strategies have a look instead.
    if len(messages) < 2:
        return []
    return messages


def _from_markers(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str] | None:
    for key, value in row.items():
        if not isinstance(value, str):
            continue
        messages = parse_marked_text(value)
        if messages:
            return messages, key
    return None


# --------------------------------------------------------------------------
# strategy 4 — a pair of role columns
# --------------------------------------------------------------------------
def _from_role_columns(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str] | None:
    lowered = {str(key).lower(): key for key in row}

    for user_keys, roles in ROLE_COLUMN_PAIRS:
        for user_key in user_keys:
            actual_user = lowered.get(user_key.lower())
            if actual_user is None:
                continue
            head = _message(roles[0], row.get(actual_user), label=actual_user)
            if not head:
                continue

            # Every other column of the row that looks like a reply becomes
            # one assistant turn, so parallel answers are all preserved — but
            # never a metadata/identifier column such as `Description`,
            # `qtype` or `unique_id`.
            canonical = {key.lower() for key in user_keys}
            replies: list[tuple[str, Any]] = []
            for key, value in row.items():
                if key == actual_user or value is None:
                    continue
                key_lower = str(key).lower()
                if is_ignored_column(str(key)):
                    continue
                if key_lower in METADATA_LOOKUP and key_lower not in canonical:
                    continue
                if isinstance(value, str) and len(_clean(value)) > 1:
                    replies.append((str(key), value))

            # Prefer the canonical pair columns and the longest answers.
            replies.sort(
                key=lambda item: (
                    item[0].lower() not in canonical,
                    -len(_clean(item[1])),
                )
            )
            if not replies:
                continue

            messages = [head]
            for key, value in replies[:4]:
                reply = _message("assistant", value, label=key)
                if reply:
                    messages.append(reply)
            if len(messages) > 1:
                return messages, actual_user
    return None


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def normalize_row(row: dict[str, Any], dataset_id: str | None = None) -> dict[str, Any]:
    """Turn one dataset row into ``{messages, num_turns, extra}``.

    Raises :class:`NormalizationError` when nothing usable is found.
    """
    if not isinstance(row, dict) or not row:
        raise NormalizationError("row is empty")

    strategy = "unknown"
    messages = _explicit_mapping(row)
    if messages:
        strategy = "explicit_mapping"

    if not messages:
        found = _from_message_list(row)
        if found:
            messages, _field = found
            strategy = "message_list"

    if not messages:
        found = _from_markers(row)
        if found:
            messages, _field = found
            strategy = "role_markers"

    if not messages:
        found = _from_role_columns(row)
        if found:
            messages, _field = found
            strategy = "role_columns"

    if not messages:
        # Last resort: keep the row inspectable rather than dropping it.
        for key, value in row.items():
            text = _clean(value)
            if len(text) > 20:
                candidate = _message("user", text, label=str(key))
                if candidate:
                    messages = [candidate]
                    strategy = "single_text"
                    break

    if not messages:
        raise NormalizationError("no usable conversation content")

    extra: dict[str, Any] = {}
    for key, value in row.items():
        if key == "input" or key == "output":
            continue
        if str(key).lower() not in METADATA_LOOKUP:
            continue
        text = _clean(value)
        if text and len(text) <= 300:
            extra[str(key)] = text

    # Metadata that duplicates content already in the transcript adds noise.
    transcript = " ".join(m["content"] for m in messages)
    extra = {
        key: value
        for key, value in extra.items()
        if value not in transcript
    }

    return {
        "messages": messages,
        # Number of conversational *turns*, not messages: parallel answers to
        # the same question are one turn (lavita returns three of them).
        "num_turns": sum(1 for m in messages if m["role"] == "user"),
        "extra": extra,
        "strategy": strategy,
    }


def normalize_rows(
    rows: Iterable[dict[str, Any]],
    dataset_id: str | None = None,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize a page of rows; returns ``(dialogs, warnings)``."""
    dialogs: list[dict[str, Any]] = []
    warnings: list[str] = []

    for position, row in enumerate(rows):
        index = offset + position
        try:
            normalized = normalize_row(row, dataset_id)
        except NormalizationError as error:
            warnings.append(f"row {index}: {error}")
            continue
        dialogs.append(
            {
                "index": index,
                "messages": normalized["messages"],
                "num_turns": normalized["num_turns"],
                "extra": normalized["extra"],
                "raw": row,
            }
        )

    if not dialogs and rows:
        # Nothing parsed — surface the raw rows so the dataset is still usable.
        warnings.append(
            "no dialog structure detected; returning raw rows for manual inspection"
        )
        for position, row in enumerate(rows):
            dialogs.append(
                {
                    "index": offset + position,
                    "messages": [
                        {
                            "role": "user",
                            "content": _trim(
                                "\n".join(f"{k}: {_clean(v)}" for k, v in row.items())
                            ),
                            "label": "raw",
                        }
                    ],
                    "num_turns": 1,
                    "extra": {},
                    "raw": row,
                }
            )

    return dialogs, warnings
