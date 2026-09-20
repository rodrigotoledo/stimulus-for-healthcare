"""Analysis orchestration: build the prompt from a transcript, call Ollama."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from . import markdown
from .ollama import OllamaClient
from ..lenses import get_lens, normalize_language
from ..schemas import AnalyzeRequest, Message


# The transcript is presented verbatim; only these labels and the prompt around
# it follow the requested output language.
TRANSCRIPT_LABELS = {
    "pt-BR": {
        "turn": "Turno",
        "transcript": "TRANSCRIÇÃO",
        "lens": "Lente",
        "dataset": "Conjunto de dados",
        "row": "Linha",
        "system": "SISTEMA",
        "user": "PACIENTE / USUÁRIO",
        "assistant": "PROFISSIONAL / ASSISTENTE",
        "reasoning": "TRACE DE RACIOCÍNIO",
    },
    "en": {
        "turn": "Turn",
        "transcript": "TRANSCRIPT",
        "lens": "Lens",
        "dataset": "Dataset",
        "row": "Row",
        "system": "SYSTEM",
        "user": "PATIENT / USER",
        "assistant": "CLINICIAN / ASSISTANT",
        "reasoning": "REASONING TRACE",
    },
}


def labels_for(language: str | None) -> dict[str, str]:
    return TRANSCRIPT_LABELS[normalize_language(language)]


def render_transcript(messages: list[Message], language: str | None = None) -> str:
    labels = labels_for(language)
    lines: list[str] = []
    for position, message in enumerate(messages, start=1):
        heading = labels.get(message.role, message.role.upper())
        if message.label:
            heading = f"{heading} ({message.label})"
        lines.append(f"--- {labels['turn']} {position} · {heading} ---")
        lines.append(message.content)
        lines.append("")
    return "\n".join(lines).strip()


def build_chat_messages(request: AnalyzeRequest) -> list[dict[str, str]]:
    """Assemble the ``/api/chat`` payload for one analysis request."""
    language = normalize_language(request.language)
    lens = get_lens(request.lens, language)
    labels = labels_for(language)

    messages = request.messages or []
    if not messages and request.prompt:
        messages = [Message(role="user", content=request.prompt)]

    if not messages:
        raise ValueError("no transcript supplied: send `messages` or `prompt`")

    transcript = render_transcript(messages, language)
    system = request.system or lens["system"]

    context: list[str] = []
    if request.dataset:
        context.append(f"{labels['dataset']}: {request.dataset}")
    if request.dialog_index is not None:
        context.append(f"{labels['row']}: {request.dialog_index}")

    user_content = "\n".join(
        [
            f"{labels['lens']}: {lens['title']}",
            *context,
            "",
            labels["transcript"],
            "==========",
            transcript,
            "==========",
            "",
            lens["instructions"],
            "",
            # Last thing the model reads, so it outweighs the English
            # transcript it just saw.
            lens["language_rule"],
        ]
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def sse(event: str, data: dict[str, Any]) -> str:
    """Format one JSON server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# How much model text accumulates before an HTML fragment is emitted. Each
# flush carries the whole reply, so a small value would mean hundreds of
# ever-larger frames.
HTML_FLUSH_CHARS = 240


def sse_fragment(html: str) -> str:
    """Format one server-rendered HTML fragment as an SSE event.

    The payload is escaped so that ``\\n\\n`` inside the fragment cannot be
    mistaken for an event boundary (which would corrupt the stream), and the
    client unescapes it before inserting the markup.
    """
    payload = html.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
    return f"event: fragment\ndata: {payload}\n\n"


def sse_raw(event: str, text: str) -> str:
    """Format a plain-text SSE event (errors and the final summary)."""
    single_line = " ".join(text.split())
    return f"event: {event}\ndata: {single_line}\n\n"


async def analyze_stream(
    request: AnalyzeRequest,
    ollama: OllamaClient,
    default_model: str,
    *,
    as_html: bool = False,
) -> AsyncIterator[str]:
    """Stream an analysis.

    With ``as_html=False`` emits JSON ``start``/``chunk``/``done`` events (the
    JSON API contract, covered by tests). With ``as_html=True`` it emits
    server-rendered HTML fragments for the Stimulus UI instead.
    """
    try:
        chat_messages = build_chat_messages(request)
    except ValueError as error:
        yield sse("error", {"message": str(error)}) if not as_html else sse_raw(
            "error", str(error)
        )
        return

    model = request.model or default_model
    options: dict[str, Any] = {"temperature": request.temperature}
    if request.max_tokens:
        options["num_predict"] = request.max_tokens

    if as_html:
        yield sse_raw(
            "meta",
            f"{model} · {request.lens} · {len(chat_messages)} messages",
        )
    else:
        yield sse(
            "start",
            {
                "model": model,
                "lens": request.lens,
                "dataset": request.dataset,
                "dialog_index": request.dialog_index,
                "num_messages": len(chat_messages),
            },
        )

    collected: list[str] = []
    last_flush = 0
    try:
        async for chunk in ollama.chat_stream(model, chat_messages, options):
            if chunk.get("error"):
                if as_html:
                    yield sse_raw("error", str(chunk["error"]))
                else:
                    yield sse("error", {"message": str(chunk["error"])})
                return

            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                collected.append(piece)
                if as_html:
                    # Each flush re-renders and sends the WHOLE reply (the client
                    # replaces the panel), so the payload grows to the full
                    # length. Flushing every ~24 chars of a long answer means
                    # hundreds of ever-larger frames, so coalesce more coarsely —
                    # still well under a second of perceived latency, since
                    # Ollama emits several tokens per network round-trip.
                    total = len("".join(collected))
                    if total - last_flush >= HTML_FLUSH_CHARS:
                        last_flush = total
                        yield sse_fragment(markdown.render("".join(collected)))
                else:
                    yield sse("chunk", {"content": piece})

            if chunk.get("done"):
                if as_html:
                    # Final payload: the complete reply, so the panel is always
                    # consistent with what the model actually said.
                    yield sse_fragment(markdown.render("".join(collected)))
                    seconds = (chunk.get("total_duration") or 0) / 1e9
                    tokens = chunk.get("eval_count") or 0
                    yield sse_raw(
                        "done",
                        f"{model} · {tokens} tokens · {seconds:.1f}s",
                    )
                else:
                    yield sse(
                        "done",
                        {
                            "model": model,
                            "eval_count": chunk.get("eval_count"),
                            "total_duration": chunk.get("total_duration"),
                            "chars": sum(len(part) for part in collected),
                        },
                    )
                return
    except Exception as error:  # OllamaError and anything unexpected mid-stream.
        if as_html:
            yield sse_raw("error", str(error))
        else:
            yield sse("error", {"message": str(error)})
        return

    # Stream ended without an explicit done flag.
    if as_html:
        yield sse_fragment(markdown.render("".join(collected)))
        yield sse_raw("done", f"{model} · stream ended")
    else:
        yield sse("done", {"model": model, "chars": sum(len(p) for p in collected)})
