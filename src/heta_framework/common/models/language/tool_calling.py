"""LiteLLM-backed language model client with explicit tool-calling support."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from heta_framework.common.models.language.config import ModelConfig
from heta_framework.common.models.language.errors import ModelRequestError, ModelResponseError
from heta_framework.common.models.language.model import (
    LanguageModel,
    _load_litellm,
    _parse_token_usage,
    _to_dict,
    repair_json,
)
from heta_framework.common.models.language.types import (
    ModelOptions,
    ToolCall,
    ToolCallingModelRequest,
    ToolCallingModelResult,
    ToolDefinition,
    ToolMessage,
)


class ToolCallingLanguageModel(LanguageModel):
    """LiteLLM language model client for native tool-calling exchanges.

    This class is an explicit opt-in variant of :class:`LanguageModel`. It keeps
    normal text generation available through the inherited ``invoke`` methods
    and adds ``invoke_with_tools`` for models/providers that support OpenAI-style
    ``tools`` and ``tool_choice`` parameters.
    """

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
        validate_function_calling_support: bool = True,
    ) -> None:
        super().__init__(
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
        self.validate_function_calling_support = validate_function_calling_support

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ToolCallingLanguageModel":
        """Create a tool-calling model from ``ModelConfig``."""
        return cls(**asdict(config))

    async def invoke_with_tools(
        self,
        request: ToolCallingModelRequest,
    ) -> ToolCallingModelResult:
        """Run one native tool-calling model request."""
        litellm = _load_litellm()
        self._ensure_function_calling_supported(litellm, request)
        try:
            async with self._semaphore:
                response = await litellm.acompletion(**self._build_tool_payload(request))
        except Exception as exc:
            raise ModelRequestError(
                "tool-calling model request failed",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        return self._parse_tool_result(response, request)

    def supports_function_calling(self) -> bool:
        """Return whether LiteLLM reports function-calling support for this model."""
        litellm = _load_litellm()
        supports = getattr(litellm, "supports_function_calling", None)
        if supports is None:
            return False
        return bool(supports(model=self.config.model_name))

    def _ensure_function_calling_supported(
        self,
        litellm: Any,
        request: ToolCallingModelRequest,
    ) -> None:
        if not self.validate_function_calling_support or not request.tools:
            return
        supports = getattr(litellm, "supports_function_calling", None)
        if supports is None:
            raise ModelRequestError(
                "LiteLLM does not expose supports_function_calling; "
                "disable validate_function_calling_support to bypass this check",
                trace_context=request.trace_context,
            )
        try:
            supported = bool(supports(model=self.config.model_name))
        except Exception as exc:
            raise ModelRequestError(
                "failed to check LiteLLM function-calling support",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        if not supported:
            raise ModelRequestError(
                f"model does not support function calling according to LiteLLM: "
                f"{self.config.model_name}",
                trace_context=request.trace_context,
            )

    def _build_tool_payload(self, request: ToolCallingModelRequest) -> dict[str, Any]:
        options = request.options or ModelOptions()
        payload = _base_payload(
            model_name=self.config.model_name,
            request_timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            drop_unsupported_params=self.config.drop_unsupported_params,
            default_temperature=self.config.default_temperature,
            options=options,
        )
        payload["messages"] = [_build_tool_message(message) for message in request.messages]
        if request.tools:
            payload["tools"] = [_build_tool_definition(tool) for tool in request.tools]
        if request.tools or request.tool_choice != "auto":
            payload["tool_choice"] = _build_tool_choice(request.tool_choice)
        _apply_response_format(payload, options, request.response_schema)
        if self.config.api_key:
            payload["api_key"] = self.config.api_key
        if self.config.api_base:
            payload["api_base"] = self.config.api_base
        if self.config.provider_options:
            payload.update(self.config.provider_options)
        if options.provider_options:
            payload.update(options.provider_options)
        return payload

    def _parse_tool_result(
        self,
        response: Any,
        request: ToolCallingModelRequest,
    ) -> ToolCallingModelResult:
        data = _to_dict(response)
        try:
            choice = data["choices"][0]
            raw_message = choice["message"]
            if not isinstance(raw_message, dict):
                raw_message = _to_dict(raw_message)
            content = raw_message.get("content")
            raw_tool_calls = raw_message.get("tool_calls") or ()
        except Exception as exc:
            raise ModelResponseError(
                "model response is missing expected tool-calling completion fields",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc

        try:
            message = ToolMessage(
                role="assistant",
                content=content or None,
                tool_calls=tuple(
                    _parse_tool_call(tool_call, trace_context=request.trace_context)
                    for tool_call in raw_tool_calls
                ),
            )
        except ModelResponseError:
            raise
        except (TypeError, ValueError) as exc:
            raise ModelResponseError(
                "model response contains an invalid tool-calling assistant message",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc

        return ToolCallingModelResult(
            message=message,
            model_name=self.config.model_name,
            token_usage=_parse_token_usage(data.get("usage")),
            finish_reason=choice.get("finish_reason"),
            trace_context=request.trace_context,
            raw_response=data,
        )


def _base_payload(
    *,
    model_name: str,
    request_timeout: float,
    max_retries: int,
    drop_unsupported_params: bool,
    default_temperature: float,
    options: ModelOptions,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "stream": False,
        "timeout": request_timeout,
        "num_retries": max_retries,
        "drop_params": drop_unsupported_params,
        "temperature": (
            options.temperature if options.temperature is not None else default_temperature
        ),
    }
    if options.max_output_tokens is not None:
        payload["max_tokens"] = options.max_output_tokens
    if options.top_p is not None:
        payload["top_p"] = options.top_p
    if options.stop_sequences:
        payload["stop"] = options.stop_sequences
    return payload


def _build_tool_definition(tool: ToolDefinition) -> dict[str, Any]:
    parameters = dict(tool.parameters_schema) or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _build_tool_choice(tool_choice: str) -> str | dict[str, Any]:
    if tool_choice in {"auto", "none", "required"}:
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice}}


def _build_tool_message(message: ToolMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    elif message.role == "assistant" and message.tool_calls:
        payload["content"] = None
    if message.tool_calls:
        payload["tool_calls"] = [_build_tool_call(tool_call) for tool_call in message.tool_calls]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _build_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(
                dict(tool_call.arguments),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _apply_response_format(
    payload: dict[str, Any],
    options: ModelOptions,
    response_schema: type | dict[str, Any] | None,
) -> None:
    response_format = options.response_format
    if response_format is None and response_schema is not None:
        response_format = {"type": "json_object"}
    if isinstance(response_format, str):
        response_format = {"type": response_format}
    if response_format is not None:
        payload["response_format"] = response_format


def _parse_tool_call(raw_tool_call: Any, *, trace_context: dict[str, Any] | None) -> ToolCall:
    data = raw_tool_call if isinstance(raw_tool_call, dict) else _to_dict(raw_tool_call)
    try:
        tool_call_id = data["id"]
        function = data["function"]
        if not isinstance(function, dict):
            function = _to_dict(function)
        name = function["name"]
        arguments = _parse_tool_arguments(function.get("arguments"), trace_context=trace_context)
    except ModelResponseError:
        raise
    except Exception as exc:
        raise ModelResponseError(
            "model response contains an invalid tool call",
            trace_context=trace_context,
            cause=exc,
        ) from exc
    try:
        return ToolCall(id=tool_call_id, name=name, arguments=arguments)
    except ValueError as exc:
        raise ModelResponseError(
            "model response contains an invalid tool call",
            trace_context=trace_context,
            cause=exc,
        ) from exc


def _parse_tool_arguments(
    raw_arguments: Any,
    *,
    trace_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if raw_arguments is None or raw_arguments == "":
        return {}
    if isinstance(raw_arguments, dict):
        return dict(raw_arguments)
    if not isinstance(raw_arguments, str):
        raise ModelResponseError(
            "tool call arguments must be a JSON object string",
            trace_context=trace_context,
        )
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        if repair_json is None:
            raise ModelResponseError(
                "tool call arguments could not be parsed as JSON",
                trace_context=trace_context,
            )
        try:
            parsed = json.loads(repair_json(raw_arguments))
        except Exception as exc:
            raise ModelResponseError(
                "tool call arguments could not be parsed as JSON",
                trace_context=trace_context,
                cause=exc,
            ) from exc
    if not isinstance(parsed, dict):
        raise ModelResponseError(
            "tool call arguments must decode to a JSON object",
            trace_context=trace_context,
        )
    return parsed
