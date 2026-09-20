"""HTTP client for the HuggingFace Hub and the datasets-server.

The datasets-server ``/rows`` endpoint streams individual rows, so we never
download a whole dataset: browsing is cheap and works on huge corpora such
as ``ruslanmv/ai-medical-chatbot`` (256k rows).
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings, get_settings


class HFDatasetsError(RuntimeError):
    """Raised when HuggingFace cannot serve the request."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def _message_from_response(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            if key in payload:
                return str(payload[key])
    return f"HTTP {response.status_code}"


class HuggingFaceClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=self.settings.hf_headers,
            timeout=timeout or self.settings.hf_timeout,
            follow_redirects=True,
        )

    async def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        try:
            async with self._client(timeout) as client:
                response = await client.get(url, params=params or {})
        except httpx.TimeoutException as error:
            raise HFDatasetsError(
                f"HuggingFace timed out after {self.settings.hf_timeout:.0f}s",
                status_code=504,
            ) from error
        except httpx.HTTPError as error:
            raise HFDatasetsError(f"could not reach HuggingFace: {error}") from error

        if response.status_code == 404:
            raise HFDatasetsError(_message_from_response(response), status_code=404)
        if response.status_code == 401:
            raise HFDatasetsError(
                "dataset is private or gated; set HFD_HF_TOKEN to access it",
                status_code=401,
            )
        if response.status_code >= 400:
            raise HFDatasetsError(_message_from_response(response))

        try:
            return response.json()
        except ValueError as error:
            raise HFDatasetsError("HuggingFace returned a non-JSON response") from error

    # ------------------------------------------------------------------
    # Hub search
    # ------------------------------------------------------------------
    async def search_datasets(
        self, query: str = "", limit: int = 25
    ) -> list[dict[str, Any]]:
        """Search the Hub for dataset candidates.

        With a query we use the Hub full-text search; without one we list
        the most downloaded datasets carrying the ``conversational`` tag.
        """
        params: dict[str, Any] = {"limit": limit, "full": "false", "sort": "downloads"}
        if query.strip():
            params["search"] = query.strip()
        else:
            params["filter"] = "conversational"

        payload = await self._get(f"{self.settings.hf_hub_url}/api/datasets", params)
        if not isinstance(payload, list):
            raise HFDatasetsError("unexpected search response from the Hub")
        return [item for item in payload if isinstance(item, dict)]

    # ------------------------------------------------------------------
    # datasets-server
    # ------------------------------------------------------------------
    async def list_splits(self, dataset: str) -> dict[str, Any]:
        payload = await self._get(
            f"{self.settings.hf_datasets_server_url}/splits",
            {"dataset": dataset},
        )
        if not isinstance(payload, dict) or "splits" not in payload:
            raise HFDatasetsError("unexpected splits response from datasets-server")
        return payload

    async def fetch_rows(
        self,
        dataset: str,
        config: str,
        split: str,
        offset: int = 0,
        length: int = 10,
    ) -> dict[str, Any]:
        payload = await self._get(
            f"{self.settings.hf_datasets_server_url}/rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=max(self.settings.hf_timeout, 60.0),
        )
        if not isinstance(payload, dict) or "rows" not in payload:
            raise HFDatasetsError("unexpected rows response from datasets-server")
        return payload

    async def dataset_info(self, dataset: str) -> dict[str, Any]:
        payload = await self._get(f"{self.settings.hf_hub_url}/api/datasets/{dataset}")
        if not isinstance(payload, dict):
            raise HFDatasetsError("unexpected dataset info response from the Hub")
        return payload

    async def check_dataset(self, dataset: str) -> bool:
        """True when the datasets-server can serve this dataset."""
        try:
            await self.list_splits(dataset)
        except HFDatasetsError:
            return False
        return True
