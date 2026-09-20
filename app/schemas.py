"""Pydantic schemas shared by the API routes and the React client."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "reasoning"]


class Message(BaseModel):
    role: Role
    content: str
    label: str | None = None


class Dialog(BaseModel):
    """One dataset row, normalized into a chat transcript."""

    index: int
    messages: list[Message]
    num_turns: int
    extra: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class DialogPage(BaseModel):
    dataset: str
    config: str
    split: str
    offset: int
    length: int
    num_rows_total: int | None = None
    num_rows_page: int
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    dialogs: list[Dialog]


class SplitInfo(BaseModel):
    config: str
    split: str
    num_rows: int | None = None


class SplitsResponse(BaseModel):
    dataset: str
    splits: list[SplitInfo]
    pending: list[dict[str, Any]] = Field(default_factory=list)
    failed: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    id: str
    description: str | None = None
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = Field(default_factory=list)
    gated: bool = False
    curated: bool = False
    verified: bool | None = None
    default_config: str | None = None
    default_split: str | None = None


class CuratedDataset(DatasetInfo):
    curated: bool = True
    note: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[DatasetInfo]


class ModelInfo(BaseModel):
    name: str
    size: int | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    modified_at: str | None = None


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]


class AnalyzeRequest(BaseModel):
    model: str | None = None
    dataset: str | None = None
    # int | None rather than `int | None = None` alone: Pydantic must reject a
    # null here, because the UI always knows which row it is analyzing.
    dialog_index: int | None = None
    messages: list[Message] | None = None
    prompt: str | None = None
    system: str | None = None
    lens: str = "clinical_review"
    # Language the *reply* is written in. The transcript is never translated.
    language: str = "pt-BR"
    temperature: float = 0.2
    max_tokens: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: dict[str, Any]
    defaults: dict[str, Any]
    version: str
