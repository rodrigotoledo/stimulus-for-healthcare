"""Render the model's markdown reply as sanitized HTML.

The analysis output goes straight into the DOM, so this module is a security
boundary as much as a formatter. It is deliberately a small, dependency-free
renderer that **escapes first and then builds markup**, so a crafted reply
cannot inject script tags, event handlers or ``javascript:`` URLs.

Supported (a deliberate subset — enough for the lenses in ``lenses.py``, which
ask for numbered steps, headings and emphasis):

* fenced ``` code blocks
* ATX headings (``#`` .. ``######``)
* unordered (``-``, ``*``, ``+``) and ordered (``1.``) lists
* blockquotes (``>``)
* horizontal rules
* tables (pipe syntax)
* ``**bold**``, ``*italic*``, ``` `code` `` inline
"""

from __future__ import annotations

import re
from html import escape

_CODE_FENCE = re.compile(r"^\s*```(\w*)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_UNORDERED = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_BLOCKQUOTE = re.compile(r"^\s*>\s?(.*)$")
_HRULE = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:-]*-[\s|:-]*\|?\s*$")

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<![\*\w])\*([^\*\n]+)\*(?!\*)")
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
# Matches ANY destination, not just http(s). An unsafe scheme is consumed and
# rendered as plain text, so the raw `[text](javascript:…)` syntax never shows.
_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)")


def _safe_url(url: str) -> str | None:
    """Only http(s) links survive; everything else is dropped."""
    lowered = url.strip().lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return escape(url.strip(), quote=True)
    return None


def inline(text: str) -> str:
    """Apply inline formatting to already-escaped text."""
    # Inline code is protected first so its contents are not re-formatted.
    placeholders: list[str] = []

    def stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(1))
        return f"\x00{len(placeholders) - 1}\x00"

    text = _INLINE_CODE.sub(stash, text)

    def link(match: re.Match[str]) -> str:
        label = match.group(1)
        url = _safe_url(match.group(2))
        if url is None:
            # Unsafe scheme (javascript:, data:, …): drop the destination but
            # keep the label so the reply stays readable and no syntax leaks.
            return label
        return (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'class="text-primary underline decoration-primary/40 hover:decoration-primary">{label}</a>'
        )

    text = _LINK.sub(link, text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    text = _STRIKE.sub(r"<del>\1</del>", text)

    for index, code in enumerate(placeholders):
        text = text.replace(
            f"\x00{index}\x00",
            f'<code class="rounded bg-base-300 px-1 py-0.5 font-mono text-[0.85em] text-primary">{code}</code>',
        )
    return text


def _table(rows: list[str]) -> str:
    def cells(row: str) -> list[str]:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    header = cells(rows[0])
    body = [cells(row) for row in rows[2:] if row.strip()]

    head_html = "".join(
        f'<th class="border-b border-base-300 px-2 py-1 text-left text-xs font-semibold uppercase tracking-wide text-base-content/60">{inline(escape(cell))}</th>'
        for cell in header
    )
    body_html = "".join(
        "<tr>"
        + "".join(
            f'<td class="border-b border-base-300/50 px-2 py-1 align-top">{inline(escape(cell))}</td>'
            for cell in row
        )
        + "</tr>"
        for row in body
    )
    return (
        '<div class="overflow-x-auto my-3"><table class="w-full text-sm">'
        f"<thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table></div>"
    )


def render(source: str) -> str:
    """Markdown -> sanitized HTML.

    Safe to call on a partial document: the caller re-renders the whole buffer
    on every flush, so an unterminated ``**`` or an open code fence simply
    renders literally until it completes.
    """
    if not source:
        return ""

    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []

    paragraph: list[str] = []
    list_items: list[tuple[str, str]] = []  # (kind, html)
    list_open: str | None = None
    quote: list[str] = []
    table_rows: list[str] = []
    code_lines: list[str] = []
    code_lang = ""
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(paragraph).strip()
            if joined:
                out.append(
                    f'<p class="my-2 leading-relaxed">{inline(escape(joined))}</p>'
                )
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_open
        if list_items and list_open:
            tag = list_open
            items = "".join(f"<li>{html}</li>" for _, html in list_items)
            marker = (
                "list-decimal" if tag == "ol" else "list-disc"
            )
            out.append(
                f'<{tag} class="{marker} ml-5 my-2 space-y-1 leading-relaxed">{items}</{tag}>'
            )
        list_items.clear()
        list_open = None

    def flush_quote() -> None:
        if quote:
            joined = " ".join(quote).strip()
            if joined:
                out.append(
                    '<blockquote class="border-l-2 border-primary/50 pl-3 my-3 '
                    f'text-base-content/80 italic">{inline(escape(joined))}</blockquote>'
                )
            quote.clear()

    def flush_table() -> None:
        if len(table_rows) >= 2:
            out.append(_table(table_rows))
        elif table_rows:
            # A lone "| header |" line is not a table; treat it as text so the
            # content is never silently dropped.
            for row in table_rows:
                out.append(
                    f'<p class="my-2 leading-relaxed">{inline(escape(row.strip()))}</p>'
                )
        table_rows.clear()

    def flush_all() -> None:
        flush_paragraph()
        flush_list()
        flush_quote()
        flush_table()

    for line in lines:
        fence = _CODE_FENCE.match(line)
        if fence:
            if in_code:
                body = escape("\n".join(code_lines))
                label = (
                    f'<span class="text-[10px] uppercase tracking-wider text-base-content/40">{escape(code_lang)}</span>'
                    if code_lang
                    else ""
                )
                out.append(
                    '<div class="my-3 rounded-lg border border-base-300 bg-base-100 overflow-hidden">'
                    f'<div class="px-3 py-1.5 border-b border-base-300/60">{label}</div>'
                    f'<pre class="p-3 m-0 overflow-x-auto text-[12.5px] leading-relaxed font-mono whitespace-pre">{body}</pre>'
                    "</div>"
                )
                code_lines.clear()
                code_lang = ""
                in_code = False
            else:
                flush_all()
                in_code = True
                code_lang = fence.group(1)
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_all()
            continue

        if _HRULE.match(line):
            flush_all()
            out.append('<hr class="my-4 border-base-300" />')
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_all()
            level = len(heading.group(1))
            sizes = {
                1: "text-lg font-semibold mt-4 mb-2",
                2: "text-base font-semibold mt-4 mb-2",
                3: "text-sm font-semibold mt-3 mb-1.5",
                4: "text-sm font-medium mt-3 mb-1",
                5: "text-xs font-semibold uppercase tracking-wide mt-3 mb-1",
                6: "text-xs font-semibold uppercase tracking-wide mt-3 mb-1",
            }
            css = sizes.get(level, sizes[6])
            out.append(
                f'<h{level} class="{css} text-base-content">{inline(escape(heading.group(2).strip()))}</h{level}>'
            )
            continue

        if _TABLE_SEP.match(line) and table_rows:
            table_rows.append(line)
            continue

        if line.strip().startswith("|"):
            flush_paragraph()
            flush_list()
            flush_quote()
            table_rows.append(line)
            continue

        if table_rows:
            flush_table()

        unordered = _UNORDERED.match(line)
        if unordered:
            flush_paragraph()
            flush_quote()
            if list_open not in (None, "ul"):
                flush_list()
            list_open = "ul"
            list_items.append(("ul", inline(escape(unordered.group(1).strip()))))
            continue

        ordered = _ORDERED.match(line)
        if ordered:
            flush_paragraph()
            flush_quote()
            if list_open not in (None, "ol"):
                flush_list()
            list_open = "ol"
            list_items.append(("ol", inline(escape(ordered.group(2).strip()))))
            continue

        if list_open:
            flush_list()

        quoted = _BLOCKQUOTE.match(line)
        if quoted:
            flush_paragraph()
            quote.append(quoted.group(1).strip())
            continue

        if quote:
            flush_quote()

        paragraph.append(line.strip())

    # An unclosed fence still renders, so a partial stream is always visible.
    if in_code:
        body = escape("\n".join(code_lines))
        out.append(
            '<div class="my-3 rounded-lg border border-base-300 bg-base-100 overflow-hidden">'
            f'<pre class="p-3 m-0 overflow-x-auto text-[12.5px] leading-relaxed font-mono whitespace-pre">{body}</pre>'
            "</div>"
        )
    else:
        flush_all()

    return "".join(out)
