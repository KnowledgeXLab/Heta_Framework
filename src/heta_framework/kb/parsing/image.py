"""Image parser."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass

from heta_framework.common.models import ImagePart, ModelRequest, TextPart
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.kb.parsing.prompts import DEFAULT_IMAGE_DESCRIPTION_PROMPT
from heta_framework.kb.parsing.types import ParsedDocument, ParsedPage, ParsedSource, make_document_id


@dataclass(frozen=True)
class ImageParserConfig:
    """Configuration for standalone image parsing."""

    prompt: str = DEFAULT_IMAGE_DESCRIPTION_PROMPT
    detail: str | None = None
    supported_file_types: tuple[str, ...] = (
        "jpg",
        "jpeg",
        "png",
        "gif",
        "webp",
        "tiff",
        "bmp",
        "ico",
    )

    def __post_init__(self) -> None:
        if self.prompt.strip() == "":
            raise ValueError("prompt must not be empty")
        if not self.supported_file_types:
            raise ValueError("supported_file_types must not be empty")
        if any(file_type.strip() == "" for file_type in self.supported_file_types):
            raise ValueError("supported_file_types must not contain empty values")
        if self.detail is not None and self.detail.strip() == "":
            raise ValueError("detail must not be empty")


class ImageParser:
    """Parse standalone image files into VLM-generated text."""

    def __init__(
        self,
        vision_model: LanguageModelProtocol,
        config: ImageParserConfig | None = None,
    ) -> None:
        self.config = config or ImageParserConfig()
        self._vision_model = vision_model
        self.supported_file_types = {
            file_type.lower().lstrip(".") for file_type in self.config.supported_file_types
        }

    async def parse(self, source: ParsedSource, data: bytes) -> ParsedDocument:
        """Parse raw standalone image bytes into a ParsedDocument."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not data:
            raise ValueError("data must not be empty")
        file_type = source.file_type.lower().lstrip(".")
        if file_type not in self.supported_file_types:
            raise ValueError(f"unsupported file type for ImageParser: {source.file_type}")

        mime_type = _image_mime_type(source.name, file_type)
        result = await self._vision_model.invoke(
            ModelRequest(
                content=[
                    TextPart(_build_prompt(self.config.prompt, source=source, mime_type=mime_type)),
                    ImagePart.from_bytes(
                        data,
                        mime_type=mime_type,
                        detail=self.config.detail,
                    ),
                ]
            )
        )
        description = result.text.strip()
        if description == "":
            raise ValueError("image description is empty")

        return ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=[
                ParsedPage(
                    page_index=0,
                    text=f"Image: {source.name}\n\nImage description: {description}",
                )
            ],
        )


def _build_prompt(prompt: str, *, source: ParsedSource, mime_type: str) -> str:
    return (
        f"{prompt.strip()}\n\n"
        f"Image metadata:\n"
        f"- file_name: {source.name}\n"
        f"- file_type: {source.file_type}\n"
        f"- media_type: {mime_type}"
    )


def _image_mime_type(filename: str, file_type: str) -> str:
    guessed = mimetypes.guess_type(filename)[0]
    if guessed is not None and guessed.startswith("image/"):
        return guessed
    if file_type == "jpg":
        return "image/jpeg"
    if file_type == "ico":
        return "image/x-icon"
    return f"image/{file_type}"
