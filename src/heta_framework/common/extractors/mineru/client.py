"""MinerU document extraction client."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import time
from typing import Any

import httpx

from heta_framework.common.extractors.types import (
    DocumentInput,
    ExtractedDocument,
    ExtractionOptions,
)
from heta_framework.common.extractors.mineru.artifacts import (
    mineru_artifact_to_extracted_document,
    parse_mineru_zip,
)
from heta_framework.common.extractors.mineru.types import (
    MinerUClientConfig,
    MinerUParseOptions,
)


class MinerUClient:
    """Extract document content through MinerU Cloud or a local MinerU service."""

    def __init__(
        self,
        config: MinerUClientConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = client

    async def extract(
        self,
        document: DocumentInput,
        options: ExtractionOptions | MinerUParseOptions | None = None,
    ) -> ExtractedDocument:
        """Extract one document and return provider-neutral blocks."""
        mineru_options = _mineru_options(options)
        if self.config.provider == "local":
            zip_content = await self._parse_local(document, mineru_options)
        else:
            zip_content = await self._parse_cloud(document, mineru_options)
        artifact = parse_mineru_zip(zip_content)
        return mineru_artifact_to_extracted_document(artifact)

    async def _parse_local(self, document: DocumentInput, options: MinerUParseOptions) -> bytes:
        if self.config.local_api_mode == "tasks":
            return await self._parse_local_task(document, options)
        return await self._parse_local_file_parse(document, options)

    async def _parse_local_task(
        self,
        document: DocumentInput,
        options: MinerUParseOptions,
    ) -> bytes:
        response = await self._request(
            "POST",
            self._local_url("/tasks"),
            files=_local_files(document),
            data=_local_parse_form_data(options),
            timeout=self.config.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"MinerU local task creation failed: HTTP {response.status_code}")
        task_id = _local_task_id(response.json())
        if task_id is None:
            raise RuntimeError("MinerU local task creation did not return task_id")

        await self._wait_for_local_task(task_id)
        result_response = await self._request(
            "GET",
            self._local_url(f"/tasks/{task_id}/result"),
            timeout=self.config.request_timeout,
        )
        if result_response.status_code != 200:
            raise RuntimeError(f"MinerU local result download failed: HTTP {result_response.status_code}")
        if _looks_like_zip_response(result_response):
            return result_response.content
        raise RuntimeError("MinerU local task result is not a zip response")

    async def _parse_local_file_parse(
        self,
        document: DocumentInput,
        options: MinerUParseOptions,
    ) -> bytes:
        response = await self._request(
            "POST",
            self._local_url("/file_parse"),
            files=_local_files(document),
            data=_local_parse_form_data(options),
            timeout=self.config.parse_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"MinerU local parse failed: HTTP {response.status_code}")
        if _looks_like_zip_response(response):
            return response.content
        raise RuntimeError("MinerU local response is not a zip response")

    async def _wait_for_local_task(self, task_id: str) -> None:
        deadline = time.monotonic() + self.config.parse_timeout
        while time.monotonic() < deadline:
            response = await self._request(
                "GET",
                self._local_url(f"/tasks/{task_id}"),
                timeout=self.config.request_timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(f"MinerU local task polling failed: HTTP {response.status_code}")
            payload = response.json()
            status = _local_task_status(payload)
            if status in {"completed", "done", "success", "finished"}:
                return
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"MinerU local task failed: {_local_task_error(payload)}")
            await asyncio.sleep(self.config.poll_interval)
        raise TimeoutError(f"MinerU local task timed out after {self.config.parse_timeout}s")

    async def _parse_cloud(self, document: DocumentInput, options: MinerUParseOptions) -> bytes:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        create_response = await self._request(
            "POST",
            self.config.cloud_base_url.rstrip("/") + "/file-urls/batch",
            headers=headers,
            json={
                "files": [
                    {
                        "name": document.filename,
                        "data_id": _safe_mineru_data_id(document.filename.rsplit(".", 1)[0]),
                    }
                ],
                "language": options.language,
                "enable_table": options.enable_table,
                "enable_formula": options.enable_formula,
                "model_version": options.model_version,
            },
            timeout=self.config.request_timeout,
        )
        if create_response.status_code != 200:
            raise RuntimeError(f"MinerU cloud task creation failed: HTTP {create_response.status_code}")
        payload = create_response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"MinerU cloud task creation failed: {payload.get('msg')}")
        batch_id = payload.get("data", {}).get("batch_id")
        file_urls = payload.get("data", {}).get("file_urls")
        if not batch_id or not isinstance(file_urls, list) or not file_urls:
            raise RuntimeError("MinerU cloud did not return batch_id and file_urls")

        upload_response = await self._request(
            "PUT",
            file_urls[0],
            content=document.data,
            timeout=self.config.request_timeout,
        )
        if upload_response.status_code not in {200, 201, 204}:
            raise RuntimeError(f"MinerU cloud upload failed: HTTP {upload_response.status_code}")

        zip_url = await self._poll_cloud_zip_url(str(batch_id), str(document.filename), headers)
        zip_response = await self._request("GET", zip_url, timeout=self.config.request_timeout)
        if zip_response.status_code != 200:
            raise RuntimeError(f"MinerU zip download failed: HTTP {zip_response.status_code}")
        return zip_response.content

    async def _poll_cloud_zip_url(
        self,
        batch_id: str,
        filename: str,
        headers: dict[str, str],
    ) -> str:
        deadline = time.monotonic() + self.config.parse_timeout
        url = self.config.cloud_base_url.rstrip("/") + f"/extract-results/batch/{batch_id}"
        while time.monotonic() < deadline:
            response = await self._request(
                "GET",
                url,
                headers=headers,
                timeout=self.config.request_timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(f"MinerU cloud polling failed: HTTP {response.status_code}")
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(f"MinerU cloud polling failed: {payload.get('msg')}")
            result = _mineru_batch_result(payload, filename=filename)
            state = result.get("state")
            if state == "done":
                zip_url = result.get("full_zip_url")
                if not zip_url:
                    raise RuntimeError("MinerU cloud result did not include full_zip_url")
                return str(zip_url)
            if state == "failed":
                raise RuntimeError(
                    f"MinerU cloud parsing failed: {result.get('err_msg') or result.get('err_code')}"
                )
            await asyncio.sleep(self.config.poll_interval)
        raise TimeoutError(f"MinerU cloud parsing timed out after {self.config.parse_timeout}s")

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._client is not None:
            return await self._client.request(method, url, **kwargs)
        async with httpx.AsyncClient() as client:
            return await client.request(method, url, **kwargs)

    def _local_url(self, path: str) -> str:
        return (self.config.endpoint_url or "").rstrip("/") + path


def _mineru_options(options: ExtractionOptions | MinerUParseOptions | None) -> MinerUParseOptions:
    if options is None:
        return MinerUParseOptions()
    if isinstance(options, MinerUParseOptions):
        return options
    return MinerUParseOptions(
        language=options.language,
        enable_table=options.enable_table,
        enable_formula=options.enable_formula,
        include_images=options.include_images,
    )


def _local_files(document: DocumentInput) -> dict[str, tuple[str, bytes, str]]:
    media_type = document.media_type or mimetypes.guess_type(document.filename)[0]
    return {"files": (document.filename, document.data, media_type or "application/octet-stream")}


def _local_parse_form_data(options: MinerUParseOptions) -> dict[str, str]:
    data = {
        "lang_list": options.language,
        "backend": options.backend,
        "effort": options.effort,
        "parse_method": options.parse_method,
        "formula_enable": _bool_string(options.enable_formula),
        "table_enable": _bool_string(options.enable_table),
        "image_analysis": _bool_string(options.image_analysis),
        "start_page_id": str(options.start_page_id),
        "return_md": "true",
        "return_middle_json": "true",
        "return_model_output": "true",
        "return_content_list": "true",
        "return_images": _bool_string(options.include_images),
        "return_original_file": "true",
        "response_format_zip": "true",
    }
    if options.end_page_id is not None:
        data["end_page_id"] = str(options.end_page_id)
    return data


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def _looks_like_zip_response(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    return "zip" in content_type or response.content.startswith(b"PK\x03\x04")


def _local_task_id(payload: dict[str, Any]) -> str | None:
    task_id = payload.get("task_id")
    if isinstance(task_id, str) and task_id.strip():
        return task_id
    data = payload.get("data")
    if isinstance(data, dict):
        task_id = data.get("task_id") or data.get("id")
        if isinstance(task_id, str) and task_id.strip():
            return task_id
    return None


def _local_task_status(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, str):
        return status.lower()
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("status"), str):
        return str(data["status"]).lower()
    return ""


def _local_task_error(payload: dict[str, Any]) -> str:
    for key in ("error", "err_msg", "message", "msg"):
        value = payload.get(key)
        if value:
            return str(value)
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("error", "err_msg", "message", "msg"):
            value = data.get(key)
            if value:
                return str(value)
    return "unknown error"


def _mineru_batch_result(payload: dict[str, Any], *, filename: str) -> dict[str, Any]:
    results = payload.get("data", {}).get("extract_result")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict) and result.get("file_name") == filename:
                return result
        if results and isinstance(results[0], dict):
            return results[0]
    return {}


def _safe_mineru_data_id(stem: str, *, max_bytes: int = 120) -> str:
    encoded = stem.encode("utf-8")
    if len(encoded) <= max_bytes:
        return stem
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    head = encoded[: max_bytes - len(digest) - 1].decode("utf-8", errors="ignore")
    return f"{head}_{digest}"
