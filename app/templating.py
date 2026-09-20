"""Jinja2 environment, filters and shared template context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ROLE_LABELS = {
    "user": "Paciente / Usuário",
    "assistant": "Profissional / Assistente",
    "reasoning": "Trace de raciocínio",
    "system": "Sistema",
}


def _message_field(message: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a Pydantic model or a plain dict."""
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def first_user_content(messages: list[Any], limit: int = 180) -> str:
    """Preview text for a dialog card: the first user turn, else anything.

    Accepts both `Message` models (from the API schema) and plain dicts (from
    the normalizer), because templates render either.
    """
    if not messages:
        return ""
    source = next(
        (m for m in messages if _message_field(m, "role") == "user"), messages[0]
    )
    text = " ".join((_message_field(source, "content", "") or "").split())
    return text if len(text) <= limit else f"{text[:limit]}…"


def importmap_json() -> str:
    """The importmap served to the browser (Stimulus + app controllers)."""
    raw = (STATIC_DIR / "importmap.json").read_text(encoding="utf-8")
    return json.dumps(json.loads(raw), indent=2)


templates.env.filters["first_user_content"] = first_user_content

BASE_CONTEXT: dict[str, Any] = {
    "importmap": importmap_json(),
    "role_labels": ROLE_LABELS,
}
