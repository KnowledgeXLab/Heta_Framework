"""Data types shared by language model clients."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any


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
