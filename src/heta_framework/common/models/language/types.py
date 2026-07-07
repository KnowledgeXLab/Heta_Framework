"""Data types shared by language model clients."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by a model service."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelOptions:
    """Per-request model options."""

    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    response_format: str | dict[str, Any] | None = None
    provider_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class TextPart:
    """Text part for a language model request."""

    text: str

    def __post_init__(self) -> None:
        if self.text.strip() == "":
            raise ValueError("text must not be empty")


@dataclass(frozen=True)
class ImagePart:
    """Image part for a language model request."""

    url: str
    mime_type: str | None = None
    detail: str | None = None
    format: str | None = None

    def __post_init__(self) -> None:
        if self.url.strip() == "":
            raise ValueError("url must not be empty")
        if self.mime_type is not None and self.mime_type.strip() == "":
            raise ValueError("mime_type must not be empty")
        if self.format is not None and self.format.strip() == "":
            raise ValueError("format must not be empty")

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        mime_type: str | None = None,
        detail: str | None = None,
        format: str | None = None,
    ) -> "ImagePart":
        """Create an image part from a remote URI or already encoded data URI."""
        return cls(url=uri, mime_type=mime_type, detail=detail, format=format)

    @classmethod
    def from_file(
        cls,
        file: str | PathLike[str],
        *,
        mime_type: str | None = None,
        detail: str | None = None,
        format: str | None = None,
    ) -> "ImagePart":
        """Create an image part from a local image file."""
        if str(file).strip() == "":
            raise ValueError("file must not be empty")
        url, resolved_mime_type = _data_url_from_file(file, mime_type=mime_type)
        return cls(
            url=url,
            mime_type=resolved_mime_type,
            detail=detail,
            format=format,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mime_type: str,
        detail: str | None = None,
        format: str | None = None,
    ) -> "ImagePart":
        """Create an image part from in-memory image bytes."""
        if len(data) == 0:
            raise ValueError("data must not be empty")
        if mime_type.strip() == "":
            raise ValueError("mime_type is required when data is provided")
        return cls(
            url=_to_data_url(data, mime_type),
            mime_type=mime_type,
            detail=detail,
            format=format,
        )


ContentPart = TextPart | ImagePart


def _data_url_from_file(file: str | PathLike[str], *, mime_type: str | None) -> tuple[str, str]:
    from pathlib import Path
    import mimetypes

    file_path = Path(file)
    inferred_mime_type = mime_type or mimetypes.guess_type(file_path.name)[0]
    if inferred_mime_type is None:
        raise ValueError(f"could not infer media type for image path: {file_path}")
    return _to_data_url(file_path.read_bytes(), inferred_mime_type), inferred_mime_type


def _to_data_url(data: bytes, mime_type: str) -> str:
    import base64

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@dataclass(frozen=True)
class ModelRequest:
    """One language model request."""

    prompt: str | None = None
    content: list[ContentPart] | None = None
    system_prompt: str | None = None
    options: ModelOptions | None = None
    response_schema: type | dict[str, Any] | None = None
    trace_context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.prompt is None and not self.content:
            raise ValueError("ModelRequest requires prompt or content")
        if self.prompt is not None and self.content is not None:
            raise ValueError("ModelRequest cannot include both prompt and content")
        if self.prompt is not None and self.prompt.strip() == "":
            raise ValueError("prompt must not be empty")
        if self.content is not None and not self.content:
            raise ValueError("content must not be empty")


@dataclass(frozen=True)
class ModelResult:
    """Final result for one language model request."""

    text: str
    parsed: Any | None = None
    model_name: str = ""
    token_usage: TokenUsage | None = None
    finish_reason: str | None = None
    trace_context: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelChunk:
    """Streaming delta for one language model request."""

    text_delta: str
    model_name: str
    finish_reason: str | None = None
    token_usage: TokenUsage | None = None
    trace_context: dict[str, Any] | None = None
    raw_chunk: dict[str, Any] | None = None


ToolMessageRole: TypeAlias = Literal["system", "user", "assistant", "tool"]
ToolChoice: TypeAlias = Literal["auto", "none", "required"] | str


@dataclass(frozen=True)
class ToolDefinition:
    """Tool schema exposed to a tool-calling language model."""

    name: str
    description: str
    parameters_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.description.strip() == "":
            raise ValueError("description must not be empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "parameters_schema", dict(self.parameters_schema))


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by a language model."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ToolMessage:
    """One chat message in a tool-calling model exchange."""

    role: ToolMessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("role must be one of: system, user, assistant, tool")
        if self.content is not None and self.content.strip() == "":
            raise ValueError("content must not be empty when provided")

        if self.role == "tool":
            if self.content is None:
                raise ValueError("tool messages require content")
            if self.tool_call_id is None or self.tool_call_id.strip() == "":
                raise ValueError("tool messages require tool_call_id")
            if self.tool_calls:
                raise ValueError("tool messages must not include tool_calls")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")

        if self.role in {"system", "user"}:
            if self.content is None:
                raise ValueError(f"{self.role} messages require content")
            if self.tool_calls:
                raise ValueError(f"{self.role} messages must not include tool_calls")

        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")

        object.__setattr__(self, "content", self.content.strip() if self.content else None)
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(
            self,
            "tool_call_id",
            self.tool_call_id.strip() if self.tool_call_id else None,
        )


@dataclass(frozen=True)
class ToolCallingModelRequest:
    """One tool-calling language model request."""

    messages: tuple[ToolMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice = "auto"
    options: ModelOptions | None = None
    response_schema: type | dict[str, Any] | None = None
    trace_context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages must not be empty")
        if isinstance(self.tool_choice, str) and self.tool_choice.strip() == "":
            raise ValueError("tool_choice must not be empty")
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("tool_choice='required' requires at least one tool")

        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tools must not contain duplicate names")

        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "tool_choice", self.tool_choice.strip())
        object.__setattr__(
            self,
            "trace_context",
            dict(self.trace_context) if self.trace_context is not None else None,
        )


@dataclass(frozen=True)
class ToolCallingModelResult:
    """Final result from one tool-calling language model request."""

    message: ToolMessage
    model_name: str = ""
    token_usage: TokenUsage | None = None
    finish_reason: str | None = None
    trace_context: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.message.role != "assistant":
            raise ValueError("tool-calling model results must contain an assistant message")
        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(
            self,
            "trace_context",
            dict(self.trace_context) if self.trace_context is not None else None,
        )
