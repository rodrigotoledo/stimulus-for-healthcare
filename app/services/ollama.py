"""Minimal async client for a local Ollama server."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..config import Settings, get_settings


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class OllamaClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            self.settings.ollama_read_timeout,
            connect=self.settings.ollama_connect_timeout,
        )

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                response = await client.get(f"{self.settings.ollama_url}/api/tags")
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise OllamaError(
                f"could not reach Ollama at {self.settings.ollama_url}: {error}"
            ) from error

        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []

        result: list[dict[str, Any]] = []
        for model in models:
            if not isinstance(model, dict) or not model.get("name"):
                continue
            details = model.get("details") or {}
            result.append(
                {
                    "name": model["name"],
                    "size": model.get("size"),
                    "modified_at": model.get("modified_at"),
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization_level": details.get("quantization_level"),
                    "capabilities": model.get("capabilities") or [],
                }
            )
        result.sort(key=lambda item: item["name"])
        return result

    async def is_up(self) -> bool:
        try:
            await self.list_models()
        except OllamaError:
            return False
        return True

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw chunks from ``/api/chat`` with ``stream: true``."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive

        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                async with client.stream(
                    "POST", f"{self.settings.ollama_url}/api/chat", json=body
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode(
                            "utf-8", errors="replace"
                        )
                        raise OllamaError(
                            f"Ollama returned {response.status_code}: {detail[:300]}"
                        )
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as error:
            raise OllamaError(
                f"could not reach Ollama at {self.settings.ollama_url}: {error}"
            ) from error
