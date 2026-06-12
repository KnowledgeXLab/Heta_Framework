"""LiteLLM-backed language model client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict
from typing import Any

from heta_framework.common.models.language.config import ModelConfig
from heta_framework.common.models.language.errors import ModelRequestError, ModelResponseError
from heta_framework.common.models.language.types import (
    ContentPart,
    ImagePart,
    ModelChunk,
    ModelOptions,
    ModelRequest,
    ModelResult,
    TextPart,
    TokenUsage,
)

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional runtime dependency.
    repair_json = None


class LanguageModel:
    """Async-first language model client backed by LiteLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        request_timeout: float = 120,
        max_retries: int = 3,
        max_concurrent_requests: int = 10,
        default_temperature: float = 0.1,
        drop_unsupported_params: bool = True,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        self.config = ModelConfig(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            request_timeout=request_timeout,
            max_retries=max_retries,
            max_concurrent_requests=max_concurrent_requests,
            default_temperature=default_temperature,
            drop_unsupported_params=drop_unsupported_params,
            provider_options=provider_options,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "LanguageModel":
        """Create a model from ``ModelConfig``."""
        return cls(**asdict(config))

    @property
    def model_name(self) -> str:
        """Model name passed to LiteLLM."""
        return self.config.model_name

    async def invoke(self, request: ModelRequest) -> ModelResult:
        """Run one model request and return its final result."""
        litellm = _load_litellm()
        try:
            async with self._semaphore:
                response = await litellm.acompletion(**self._build_payload(request, stream=False))
        except Exception as exc:
            raise ModelRequestError(
                "model request failed",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        return self._parse_result(response, request)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        """Run multiple model requests concurrently while preserving order."""
        return list(await asyncio.gather(*(self.invoke(request) for request in requests)))

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        """Stream one model request as text deltas."""
        if request.response_schema is not None:
            raise ModelRequestError(
                "stream does not support response_schema; use invoke for structured output",
                trace_context=request.trace_context,
            )
        litellm = _load_litellm()
        try:
            async with self._semaphore:
                stream = await litellm.acompletion(**self._build_payload(request, stream=True))
                async for raw_chunk in stream:
                    chunk = self._parse_chunk(raw_chunk, request)
                    if chunk is not None:
                        yield chunk
        except ModelResponseError:
            raise
        except Exception as exc:
            raise ModelRequestError(
                "model stream failed",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc

    async def aclose(self) -> None:
        """Close resources held by the model.

        LiteLLM manages its own HTTP clients, so there is currently nothing to close.
        """

    async def __aenter__(self) -> "LanguageModel":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def _build_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        options = request.options or ModelOptions()
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": _build_messages(request),
            "stream": stream,
            "timeout": self.config.request_timeout,
            "num_retries": self.config.max_retries,
            "drop_params": self.config.drop_unsupported_params,
            "temperature": (
                options.temperature
                if options.temperature is not None
                else self.config.default_temperature
            ),
        }
        if self.config.api_key:
            payload["api_key"] = self.config.api_key
        if self.config.api_base:
            payload["api_base"] = self.config.api_base
        if options.max_output_tokens is not None:
            payload["max_tokens"] = options.max_output_tokens
        if options.top_p is not None:
            payload["top_p"] = options.top_p
        if options.stop_sequences:
            payload["stop"] = options.stop_sequences

        response_format = options.response_format
        if response_format is None and request.response_schema is not None:
            response_format = {"type": "json_object"}
        if isinstance(response_format, str):
            response_format = {"type": response_format}
        if response_format is not None:
            payload["response_format"] = response_format

        if self.config.provider_options:
            payload.update(self.config.provider_options)
        if options.provider_options:
            payload.update(options.provider_options)
        return payload

    def _parse_result(self, response: Any, request: ModelRequest) -> ModelResult:
        data = _to_dict(response)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
        except Exception as exc:
            raise ModelResponseError(
                "model response is missing expected completion fields",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        parsed = _parse_structured_output(text, request) if request.response_schema else None
        return ModelResult(
            text=text,
            parsed=parsed,
            model_name=self.config.model_name,
            token_usage=_parse_token_usage(data.get("usage")),
            finish_reason=choice.get("finish_reason"),
            trace_context=request.trace_context,
            raw_response=data,
        )

    def _parse_chunk(self, raw_chunk: Any, request: ModelRequest) -> ModelChunk | None:
        data = _to_dict(raw_chunk)
        choices = data.get("choices")
        if choices == []:
            return None
        try:
            choice = choices[0]
            delta = choice.get("delta") or {}
            text_delta = delta.get("content") or ""
            finish_reason = choice.get("finish_reason")
        except Exception as exc:
            raise ModelResponseError(
                "model stream chunk is missing expected completion fields",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        if not text_delta and not finish_reason:
            return None
        return ModelChunk(
            text_delta=text_delta,
            model_name=self.config.model_name,
            finish_reason=finish_reason,
            token_usage=_parse_token_usage(data.get("usage")),
            trace_context=request.trace_context,
            raw_chunk=data,
        )


def _load_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise ModelRequestError(
            "LiteLLM is not installed; install the `heta` package with runtime dependencies",
            cause=exc,
        ) from exc
    return litellm


def _build_messages(request: ModelRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": _build_user_content(request)})
    return messages


def _build_user_content(request: ModelRequest) -> str | list[dict[str, Any]]:
    if request.prompt is not None:
        return request.prompt
    if request.content is None:
        raise ValueError("ModelRequest requires prompt or content")
    return [_build_content_part(part) for part in request.content]


def _build_content_part(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        image_url: dict[str, Any] = {"url": part.url}
        if part.detail is not None:
            image_url["detail"] = part.detail
        image_format = part.format or part.mime_type
        if image_format is not None:
            image_url["format"] = image_format
        return {"type": "image_url", "image_url": image_url}
    raise TypeError(f"unsupported model content part: {type(part)!r}")


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "json"):
        return json.loads(value.json())
    raise TypeError(f"cannot convert {type(value)!r} to dict")


def _parse_token_usage(raw_usage: Any) -> TokenUsage | None:
    if raw_usage is None:
        return None
    usage = raw_usage if isinstance(raw_usage, dict) else _to_dict(raw_usage)
    return TokenUsage(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _parse_structured_output(text: str, request: ModelRequest) -> Any:
    try:
        data = json.loads(_extract_json_text(text))
    except json.JSONDecodeError:
        if repair_json is None:
            raise ModelResponseError(
                "model response could not be parsed as JSON",
                trace_context=request.trace_context,
            )
        try:
            data = json.loads(repair_json(_extract_json_text(text)))
        except Exception as exc:
            raise ModelResponseError(
                "model response could not be parsed as JSON",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
    schema = request.response_schema
    if schema is None or isinstance(schema, dict):
        return data
    if hasattr(schema, "model_validate"):
        return schema.model_validate(data)
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(data)
    return data


def _extract_json_text(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value
