"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import pages, router
from .config import get_settings
from .templating import STATIC_DIR

settings = get_settings()

app = FastAPI(
    title="HuggingFace Dialog Selector",
    description=(
        "Browse HuggingFace dialogue datasets and analyze individual "
        "transcripts with a local Ollama model. The UI is server-rendered "
        "Jinja templates driven by Stimulus controllers; /api exposes JSON."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(pages)
app.include_router(router)
