"""HTTP routes: server-rendered pages and fragments, plus the JSON API.

Two surfaces share one service layer:

* ``/`` and ``/fragments/...`` return HTML for the Stimulus UI, and
  ``POST /api/analyze/events`` streams HTML fragments over SSE.
* ``/api/...`` returns JSON, which is what the tests and any external
  consumer use.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from ..catalog import CURATED_DATASETS, curated_entry, default_config_split
from ..config import Settings, get_settings
from ..lenses import (
    DEFAULT_LANGUAGE,
    DEFAULT_LENS,
    language_catalog,
    lens_catalog,
)
from ..schemas import (
    AnalyzeRequest,
    CuratedDataset,
    DatasetInfo,
    DialogPage,
    HealthResponse,
    ModelsResponse,
    SearchResponse,
    SplitInfo,
    SplitsResponse,
)
from ..services.analysis import analyze_stream
from ..services.dialog import normalize_rows
from ..services.huggingface import HFDatasetsError, HuggingFaceClient
from ..services.ollama import OllamaClient
from ..templating import BASE_CONTEXT, templates

router = APIRouter(prefix="/api")
pages = APIRouter(include_in_schema=False)

SettingsDep = Annotated[Settings, Depends(get_settings)]


def hf_client(settings: SettingsDep) -> HuggingFaceClient:
    return HuggingFaceClient(settings)


def ollama_client(settings: SettingsDep) -> OllamaClient:
    return OllamaClient(settings)


HFDep = Annotated[HuggingFaceClient, Depends(hf_client)]
OllamaDep = Annotated[OllamaClient, Depends(ollama_client)]


# ==========================================================================
# shared helpers
# ==========================================================================
async def _load_health(settings: Settings, ollama: OllamaClient) -> HealthResponse:
    models: list[dict[str, Any]] = []
    reachable = True
    try:
        models = await ollama.list_models()
    except Exception:
        reachable = False

    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama={
            "url": settings.ollama_url,
            "reachable": reachable,
            "model_count": len(models),
            "models": [model["name"] for model in models][:30],
        },
        defaults={
            "model": settings.ollama_default_model,
            "lens": DEFAULT_LENS,
            "page_size": settings.default_page_size,
        },
        version="0.2.0",
    )


async def _load_models(settings: Settings, ollama: OllamaClient) -> ModelsResponse:
    """Installed models, with a default that actually exists."""
    try:
        models = await ollama.list_models()
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    names = {model["name"] for model in models}
    default = settings.ollama_default_model
    if default not in names and models:
        default = next(
            (name for name in sorted(names) if "medgemma" in name.lower()),
            models[0]["name"],
        )
    return ModelsResponse(default_model=default, models=models)


async def _load_dialogs(
    hf: HuggingFaceClient,
    settings: Settings,
    dataset_id: str,
    config: str | None,
    split: str | None,
    offset: int,
    length: int,
) -> DialogPage:
    """Resolve config/split, fetch rows, and normalize them into dialogs."""
    if not config or not split:
        curated_config, curated_split = default_config_split(dataset_id)
        config = config or curated_config
        split = split or curated_split

    if not config or not split:
        payload = await hf.list_splits(dataset_id)
        available = payload.get("splits") or []
        if not available:
            detail = payload.get("failed") or payload.get("pending") or "no splits"
            raise HTTPException(
                status_code=409, detail=f"dataset has no usable splits: {detail}"
            )
        config = config or available[0].get("config")
        split = split or available[0].get("split")

    payload = await hf.fetch_rows(dataset_id, config, split, offset, length)
    raw_rows = [entry.get("row", {}) for entry in payload.get("rows", [])]
    dialogs, warnings = normalize_rows(raw_rows, dataset_id, offset)

    return DialogPage(
        dataset=dataset_id,
        config=config,
        split=split,
        offset=offset,
        length=length,
        num_rows_total=payload.get("num_rows_total"),
        num_rows_page=len(dialogs),
        truncated=payload.get("truncated", False),
        warnings=warnings,
        dialogs=dialogs,
    )


# ==========================================================================
# HTML pages and fragments (Stimulus UI)
# ==========================================================================
@pages.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: SettingsDep, ollama: OllamaDep):
    """The whole app: three columns, hydrated by Stimulus controllers."""
    try:
        health: HealthResponse | None = await _load_health(settings, ollama)
    except Exception:
        health = None

    try:
        models = (await _load_models(settings, ollama)).models
    except HTTPException:
        models = []

    return templates.TemplateResponse(
        request,
        "layout.html",
        {
            **BASE_CONTEXT,
            "health": health,
            "datasets": [
                CuratedDataset(
                    id=entry["id"],
                    note=entry.get("note"),
                    tags=entry.get("tags", []),
                    default_config=entry.get("default_config"),
                    default_split=entry.get("default_split"),
                    verified=True,
                )
                for entry in CURATED_DATASETS
            ],
            "selected_id": None,
            "models": models,
            "default_model": settings.ollama_default_model,
            "lenses": lens_catalog(DEFAULT_LANGUAGE),
            "default_lens": DEFAULT_LENS,
            "languages": language_catalog(),
            "default_language": DEFAULT_LANGUAGE,
        },
    )


@pages.get("/fragments/health", response_class=HTMLResponse)
async def health_fragment(request: Request, settings: SettingsDep, ollama: OllamaDep):
    try:
        health: HealthResponse | None = await _load_health(settings, ollama)
    except Exception:
        health = None
    return templates.TemplateResponse(
        request,
        "partials/status_badge.html",
        {**BASE_CONTEXT, "health": health},
    )


@pages.get("/fragments/datasets/curated", response_class=HTMLResponse)
async def datasets_curated_fragment(request: Request):
    datasets = [
        CuratedDataset(
            id=entry["id"],
            note=entry.get("note"),
            tags=entry.get("tags", []),
            default_config=entry.get("default_config"),
            default_split=entry.get("default_split"),
            verified=True,
        )
        for entry in CURATED_DATASETS
    ]
    return templates.TemplateResponse(
        request,
        "partials/dataset_list.html",
        {**BASE_CONTEXT, "datasets": datasets, "selected_id": None},
    )


@pages.get("/fragments/datasets/search", response_class=HTMLResponse)
async def datasets_search_fragment(
    request: Request,
    hf: HFDep,
    q: str = Query(""),
):
    try:
        raw = await hf.search_datasets(q, 25)
    except HFDatasetsError as error:
        return HTMLResponse(
            f'<div class="alert alert-error text-xs py-2">{error}</div>',
            status_code=error.status_code,
        )

    datasets: list[DatasetInfo] = []
    for item in raw:
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        entry = curated_entry(dataset_id)
        config, split = default_config_split(dataset_id) if entry else (None, None)
        datasets.append(
            DatasetInfo(
                id=dataset_id,
                description=(item.get("description") or "").strip()[:400] or None,
                downloads=item.get("downloads"),
                likes=item.get("likes"),
                tags=[t for t in (item.get("tags") or []) if isinstance(t, str)][:12],
                gated=bool(item.get("gated")),
                curated=entry is not None,
                default_config=config,
                default_split=split,
            )
        )

    return templates.TemplateResponse(
        request,
        "partials/dataset_list.html",
        {**BASE_CONTEXT, "datasets": datasets, "selected_id": None},
    )


@pages.get("/fragments/datasets/{dataset_id:path}/splits", response_class=HTMLResponse)
async def dataset_splits_fragment(
    request: Request,
    dataset_id: str,
    hf: HFDep,
):
    try:
        payload = await hf.list_splits(dataset_id)
    except HFDatasetsError as error:
        return HTMLResponse(
            f'<div class="alert alert-error text-xs py-2">{error}</div>',
            status_code=error.status_code,
        )

    splits = [
        SplitInfo(
            config=entry.get("config", ""),
            split=entry.get("split", ""),
            num_rows=entry.get("num_rows"),
        )
        for entry in payload.get("splits", [])
    ]
    configs = list(dict.fromkeys(entry.config for entry in splits))
    config, split = default_config_split(dataset_id)
    if config not in configs:
        config = configs[0] if configs else None

    visible = [entry for entry in splits if entry.config == config]
    if split not in {entry.split for entry in visible}:
        split = visible[0].split if visible else None

    return templates.TemplateResponse(
        request,
        "partials/dialog_pickers.html",
        {
            **BASE_CONTEXT,
            "dataset_id": dataset_id,
            "splits": visible,
            "configs": configs,
            "config": config,
            "split": split,
            "warnings": [],
            "error": None,
        },
    )


@pages.get("/fragments/datasets/{dataset_id:path}/rows", response_class=HTMLResponse)
async def dataset_rows_fragment(
    request: Request,
    dataset_id: str,
    hf: HFDep,
    settings: SettingsDep,
    config: str | None = Query(None),
    split: str | None = Query(None),
    offset: int = Query(0, ge=0),
    length: int = Query(10, ge=1, le=50),
    selected: int | None = Query(None),
):
    length = min(length, settings.max_page_size)

    try:
        # Splits are re-listed so the picker survives a fragment swap.
        splits_payload = await hf.list_splits(dataset_id)
    except HFDatasetsError as error:
        return HTMLResponse(
            f'<div class="alert alert-error text-xs py-2">{error}</div>',
            status_code=error.status_code,
        )

    all_splits = [
        SplitInfo(
            config=entry.get("config", ""),
            split=entry.get("split", ""),
            num_rows=entry.get("num_rows"),
        )
        for entry in splits_payload.get("splits", [])
    ]
    configs = list(dict.fromkeys(entry.config for entry in all_splits))

    curated_config, curated_split = default_config_split(dataset_id)
    config = config or curated_config or (configs[0] if configs else None)
    if config not in configs:
        config = configs[0] if configs else None
    visible = [entry for entry in all_splits if entry.config == config]
    split = split or curated_split
    if split not in {entry.split for entry in visible}:
        split = visible[0].split if visible else None

    if not config or not split:
        return HTMLResponse(
            '<div class="alert alert-error text-xs py-2">dataset has no usable splits</div>',
            status_code=409,
        )

    try:
        page = await _load_dialogs(hf, settings, dataset_id, config, split, offset, length)
    except HTTPException as error:
        return HTMLResponse(
            f'<div class="alert alert-error text-xs py-2">{error.detail}</div>',
            status_code=error.status_code,
        )
    except HFDatasetsError as error:
        return HTMLResponse(
            f'<div class="alert alert-error text-xs py-2">{error}</div>',
            status_code=error.status_code,
        )

    chosen = next(
        (dialog for dialog in page.dialogs if dialog.index == selected), None
    )

    total = page.num_rows_total
    has_next = (
        page.num_rows_page == length
        if total is None
        else offset + length < total
    )

    # The dialogs the analysis column needs, serialized once. `</` is escaped
    # so model-or-dataset text can never close the <script> element early.
    dialogs_payload = json.dumps(
        [dialog.model_dump() for dialog in page.dialogs], ensure_ascii=False
    ).replace("</", "<\\/")

    return templates.TemplateResponse(
        request,
        "partials/dialog_body.html",
        {
            **BASE_CONTEXT,
            "dataset_id": dataset_id,
            "page": page,
            "selected": chosen,
            "offset": page.offset,
            "has_next": has_next,
            "warnings": page.warnings,
            "dialogs_payload": dialogs_payload,
        },
    )


@pages.post("/fragments/analyze", response_class=HTMLResponse)
async def analyze_events(
    request: AnalyzeRequest,
    settings: SettingsDep,
    ollama: OllamaDep,
) -> StreamingResponse:
    """The Stimulus UI's analysis endpoint: HTML fragments over SSE."""
    if not request.messages and not request.prompt:
        raise HTTPException(
            status_code=422, detail="send `messages` (a transcript) or `prompt`"
        )

    return StreamingResponse(
        analyze_stream(request, ollama, settings.ollama_default_model, as_html=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================================================================
# JSON API
# ==========================================================================
@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep, ollama: OllamaDep) -> HealthResponse:
    return await _load_health(settings, ollama)


@router.get("/models", response_model=ModelsResponse)
async def list_models(settings: SettingsDep, ollama: OllamaDep) -> ModelsResponse:
    return await _load_models(settings, ollama)


@router.get("/lenses")
async def list_lenses() -> dict[str, Any]:
    return {
        "default": DEFAULT_LENS,
        "default_language": DEFAULT_LANGUAGE,
        "languages": language_catalog(),
        "lenses": lens_catalog(DEFAULT_LANGUAGE),
    }


@router.get("/datasets/curated", response_model=list[CuratedDataset])
async def curated_datasets() -> list[CuratedDataset]:
    return [
        CuratedDataset(
            id=entry["id"],
            note=entry.get("note"),
            tags=entry.get("tags", []),
            default_config=entry.get("default_config"),
            default_split=entry.get("default_split"),
            verified=True,
        )
        for entry in CURATED_DATASETS
    ]


@router.get("/datasets/search", response_model=SearchResponse)
async def search_datasets(
    hf: HFDep,
    settings: SettingsDep,
    q: str = Query("", description="Hub search query"),
    limit: Annotated[int, Query(ge=1, le=50)] = 25,
) -> SearchResponse:
    try:
        raw = await hf.search_datasets(q, limit)
    except HFDatasetsError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    results: list[DatasetInfo] = []
    for item in raw:
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        entry = curated_entry(dataset_id)
        default_config, default_split = (
            default_config_split(dataset_id) if entry else (None, None)
        )
        results.append(
            DatasetInfo(
                id=dataset_id,
                description=(item.get("description") or "").strip()[:400] or None,
                downloads=item.get("downloads"),
                likes=item.get("likes"),
                tags=[tag for tag in (item.get("tags") or []) if isinstance(tag, str)][:12],
                gated=bool(item.get("gated")),
                curated=entry is not None,
                default_config=default_config,
                default_split=default_split,
            )
        )

    _ = settings  # kept in the signature so overrides apply consistently
    return SearchResponse(query=q, results=results)


@router.get("/datasets/{dataset_id:path}/splits", response_model=SplitsResponse)
async def dataset_splits(dataset_id: str, hf: HFDep) -> SplitsResponse:
    try:
        payload = await hf.list_splits(dataset_id)
    except HFDatasetsError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    splits = [
        SplitInfo(
            config=entry.get("config", ""),
            split=entry.get("split", ""),
            num_rows=entry.get("num_rows"),
        )
        for entry in payload.get("splits", [])
    ]

    return SplitsResponse(
        dataset=dataset_id,
        splits=splits,
        pending=payload.get("pending") or [],
        failed=payload.get("failed") or [],
    )


@router.get("/datasets/{dataset_id:path}/rows", response_model=DialogPage)
async def dataset_rows(
    dataset_id: str,
    hf: HFDep,
    settings: SettingsDep,
    config: str | None = Query(None),
    split: str | None = Query(None),
    offset: Annotated[int, Query(ge=0)] = 0,
    length: Annotated[int, Query(ge=1, le=50)] = 10,
) -> DialogPage:
    length = min(length, settings.max_page_size)
    try:
        return await _load_dialogs(
            hf, settings, dataset_id, config, split, offset, length
        )
    except HFDatasetsError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    settings: SettingsDep,
    ollama: OllamaDep,
) -> StreamingResponse:
    """JSON SSE variant of the analysis stream (``start``/``chunk``/``done``)."""
    if not request.messages and not request.prompt:
        raise HTTPException(
            status_code=422, detail="send `messages` (a transcript) or `prompt`"
        )

    return StreamingResponse(
        analyze_stream(request, ollama, settings.ollama_default_model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
