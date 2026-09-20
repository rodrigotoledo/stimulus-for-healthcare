# HuggingFace Dialog Selector

Browse medical dialogue datasets on the HuggingFace Hub, pick individual
transcripts, and analyze them with a local [Ollama](https://ollama.com) model.

Server-rendered Jinja templates driven by [Stimulus](https://stimulus.hotwired.dev)
controllers. No React, no bundler, no `node_modules` — the whole app is one
Python process.

```
┌──────────────┬─────────────────────┬──────────────────────────┐
│ 1 · Dataset  │ 2 · Dialogs         │ 3 · Analysis             │
│ curated list │ config/split picker │ model + lens picker      │
│ HF search    │ dialog list         │ streamed model output    │
│              │ chat transcript     │                          │
└──────────────┴─────────────────────┴──────────────────────────┘
```

## Quick start (Docker)

```bash
docker compose up --build
# → http://localhost:8020
```

Ollama stays on the host so the models you already pulled are reused; the
container reaches it through `host.docker.internal` (configured in
`docker-compose.yml`). The compose file also has a commented-out `ollama`
service if you would rather run it in Docker too.

To point at a different Ollama host or a gated dataset:

```bash
HFD_OLLAMA_URL=http://192.168.1.10:11434 HFD_HF_TOKEN=hf_xxx docker compose up
```

## Quick start (local, no Docker)

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --reload --port 8010
# → http://localhost:8010
```

Ollama must be running (`ollama serve`) with at least one model pulled. The
default is `MedGemma:4b` (`ollama pull MedGemma:4b`); any installed model can be
picked in the UI, and the header shows a live Ollama status dot.

## How the frontend works

There is no build step. `app/static/importmap.json` maps the bare specifiers to
CDN/module URLs, `app/static/js/app.js` registers the Stimulus controllers, and
every controller lives in `app/static/js/controllers/`.

| Controller | Responsibility |
| --- | --- |
| `dataset-browser` | Switches curated/Hub tabs, loads list fragments, announces `dataset:selected` |
| `dialog-explorer` | Loads splits, pages rows, swaps in the transcript fragment |
| `analysis-stream` | POSTs the transcript and appends HTML fragments from the SSE stream |
| `status` | Polls `/fragments/health` so the header dot stays current |

Controllers never build markup. They fetch a URL, and the server answers with
finished HTML. Cross-column communication is a `window` event
(`dataset:selected`, `dialog:selected`) rather than one controller reaching into
another.

### HTML fragments over SSE

`POST /fragments/analyze` streams `event: fragment` frames whose payload is a
server-rendered HTML fragment, which the client appends. This is the Python
approximation of Hotwire's Turbo Streams: the server owns the markup, the client
only places it.

Model output is untrusted, so each fragment is HTML-escaped server-side before
it enters the stream. Newlines are escaped as `\n` too, because a literal blank
line inside a `data:` payload would be read as an event boundary and corrupt the
frame. There is a test asserting both properties.

Tokens are coalesced (~24 chars) before flushing: Ollama emits one token at a
time, and forwarding every token would mean hundreds of DOM insertions.

### Styling

Tailwind + daisyUI are **compiled ahead of time** into
`app/static/css/app.css` — the served page has no CSS runtime and makes no CDN
request. One daisyUI theme (`hf-dark`) lives in `:root`: indigo-forward and
deliberately dark, defined in `tailwind.config.js`.

```bash
npm install          # once: tailwindcss + daisyui
npm run css          # compile (minified)
npm run css:watch    # rebuild on change while editing templates
```

**Re-run `npm run css` after adding classes to a template.** Tailwind scans
`app/templates/**/*.html` and `app/static/js/**/*.js`, so a class that only ever
appears in a Stimulus controller's `classList` toggle will not be generated —
those live in the `safelist` in `tailwind.config.js`.

The Docker build runs this compilation in a `node:22-alpine` stage and copies
only the resulting stylesheet into the runtime image, so no Node toolchain ships.

> daisyUI theme values must be real CSS colors (hex is fine). A bare
> `"58.54% 0.204 277"` triplet is rejected — the values are parsed as CSS.

## API

The JSON API is unchanged and independent of the UI — it is what the test suite
and any external consumer uses. The HTML routes are the ones the browser hits.

| Route | Returns |
| --- | --- |
| `GET /` | The three-column app |
| `GET /fragments/health` | Status badge |
| `GET /fragments/datasets/curated` | Dataset list |
| `GET /fragments/datasets/search?q=` | Dataset list (Hub search) |
| `GET /fragments/datasets/{id}/splits` | Config/split pickers |
| `GET /fragments/datasets/{id}/rows` | Dialog cards + transcript |
| `POST /fragments/analyze` | **HTML fragments** over SSE |
| `GET /api/health`, `/api/models`, `/api/lenses` | JSON |
| `GET /api/datasets/…` | JSON rows/splits/search |
| `POST /api/analyze` | **JSON** events (`start`/`chunk`/`done`) over SSE |

Rows are proxied through the
[HuggingFace datasets-server](https://huggingface.co/docs/datasets-server), so
browsing a 256k-row dataset costs one HTTP request — nothing is downloaded or
cached on disk.

### Dialog normalization

HuggingFace dialog datasets share no schema, so `app/services/dialog.py` detects
a transcript in order of confidence:

1. **Explicit mapping** — for shapes in the curated catalog, e.g. lavita's one
   question + three parallel answers, or medical-o1's reasoning trace.
2. **Message-list column** — `messages` / `conversations` / ... holding
   `[{role, content}]` or `[[user, text]]` (ShareGPT-style).
3. **Role markers** — a single text column containing `<HUMAN>: ...`
   `<ASSISTANT>: ...`.
4. **Role column pairs** — `Patient`/`Doctor`, `instruction`/`output`, ...
   Extra answer columns are preserved as additional labelled turns.
5. **Single-text fallback** so an unrecognized row is still inspectable.

Metadata (`Description`, `topic`, ...) and identifier columns (`qtype`,
`unique_id`) are never mistaken for answers; they surface as chips above the
transcript instead.

### Curated datasets

Every entry in `app/catalog.py` was verified live against the datasets-server.
Two commonly-cited datasets — `Amod/mental_health_counseling_conversations` and
`OpenMed/Medical-Dialogue` — return 404 (private/gated) and are listed in
`DEAD_DATASETS` so they are not re-added.

`lavita/medical-qa-datasets` · `ruslanmv/ai-medical-chatbot` ·
`medalpaca/medical_meadow_mediqa` · `keivalya/MedQuad-MedicalQnADataset` ·
`BI55/MedText` · `FreedomIntelligence/medical-o1-reasoning-SFT` ·
`heliosbrahma/mental_health_chatbot_dataset` · `HPAI-BSC/CareQA`

### Analysis lenses

Each lens is a system guardrail plus instructions in `app/lenses.py`: clinical
review, risk audit, SOAP note, compare-answers, reasoning check, and
plain-language summary. The prompt is assembled from the *current* transcript, so
lens and conversation stay in sync.

### Output language

The analysis is written in **pt-BR by default**; English is available from the
picker in the analysis column (`language` in the request body). Only the model's
reply is localized — the transcript is always passed through verbatim, so the
analysis stays faithful to the source dialogue.

A lens therefore carries a guardrail, instructions and UI labels **per
language**, because instructions written in English make a model answer in
English no matter what the system prompt says. The language rule is appended as
the *last* line of the prompt, after the transcript, so it outweighs the English
text the model just read.

To add a language: add an entry to `LANGUAGES` and a variant to each lens in
`LENSES`. `get_lens()` falls back to English for any lens that lacks a variant,
and `normalize_language()` maps unknown values onto the default.

## Configuration

Settings use the `HFD_` prefix (environment or `.env`, see `.env.example`):

| Variable | Default | Notes |
| --- | --- | --- |
| `HFD_OLLAMA_URL` | `http://localhost:11434` | In Docker: `http://host.docker.internal:11434` |
| `HFD_OLLAMA_DEFAULT_MODEL` | `MedGemma:4b` | Falls back to any installed model |
| `HFD_HF_TOKEN` | — | Needed for gated/private datasets |
| `HFD_HF_TIMEOUT` | `30` | Seconds per HuggingFace request |
| `HFD_MAX_PAGE_SIZE` | `50` | Upper bound on rows per request |

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q
```

`tests/test_dialog.py` and `tests/test_api.py` cover the normalizer and the JSON
API against rows captured verbatim from the live datasets-server. `tests/test_ui.py`
covers the Stimulus surface: the rendered page, every fragment, HTML escaping of
model output, and SSE frame integrity.

## Disclaimer

**Research tool.** Analyses are generated by a local language model and are not
medical advice. Never use them to make clinical decisions about a real person.
