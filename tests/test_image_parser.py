import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    ImagePart,
    ModelChunk,
    ModelRequest,
    ModelResult,
    TextPart,
)
from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserProtocol,
    ImageParser,
    ImageParserConfig,
    make_document_id,
    make_parsed_source,
)


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)


class FakeVisionModel:
    def __init__(self, text: str = "a red square image") -> None:
        self.text = text
        self.last_request: ModelRequest | None = None

    @property
    def model_name(self) -> str:
        return "fake-vision-model"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.last_request = request
        return ModelResult(text=self.text)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        yield ModelChunk(text_delta=(await self.invoke(request)).text)


def test_image_parser_satisfies_protocol():
    assert isinstance(ImageParser(FakeVisionModel()), DocumentParserProtocol)


def test_image_parser_describes_standalone_image():
    source = make_parsed_source(
        key="raw/red.png",
        name="red.png",
        file_type="png",
        data=PNG_BYTES,
    )
    model = FakeVisionModel()

    async def run():
        return await ImageParser(model).parse(source, PNG_BYTES)

    document = asyncio.run(run())

    assert document.document_id == make_document_id(source.content_sha256)
    assert document.source == source
    assert document.pages[0].page_index == 0
    assert document.pages[0].text == "Image: red.png\n\nImage description: a red square image"
    assert model.last_request is not None
    assert model.last_request.content is not None
    assert isinstance(model.last_request.content[0], TextPart)
    assert isinstance(model.last_request.content[1], ImagePart)
    assert "detailed, retrieval-friendly description" in model.last_request.content[0].text
    assert "file_name: red.png" in model.last_request.content[0].text
    assert "media_type: image/png" in model.last_request.content[0].text
    assert model.last_request.content[1].mime_type == "image/png"
    assert model.last_request.content[1].url.startswith("data:image/png;base64,")


def test_image_parser_infers_jpeg_mime_type_from_file_type():
    source = make_parsed_source(
        key="raw/photo.unknown",
        name="photo.unknown",
        file_type="jpg",
        data=b"jpeg-bytes",
    )
    model = FakeVisionModel()

    async def run():
        return await ImageParser(model).parse(source, b"jpeg-bytes")

    asyncio.run(run())

    assert model.last_request is not None
    assert model.last_request.content is not None
    image = model.last_request.content[1]
    assert isinstance(image, ImagePart)
    assert image.mime_type == "image/jpeg"


def test_image_parser_rejects_unsupported_file_type():
    source = make_parsed_source(
        key="raw/archive.zip",
        name="archive.zip",
        file_type="zip",
        data=b"zip",
    )

    async def run():
        return await ImageParser(FakeVisionModel()).parse(source, b"zip")

    with pytest.raises(ValueError, match="unsupported file type"):
        asyncio.run(run())


def test_image_parser_rejects_empty_descriptions():
    source = make_parsed_source(
        key="raw/red.png",
        name="red.png",
        file_type="png",
        data=PNG_BYTES,
    )

    async def run():
        return await ImageParser(FakeVisionModel(text=" ")).parse(source, PNG_BYTES)

    with pytest.raises(ValueError, match="description"):
        asyncio.run(run())


def test_image_parser_config_validates_supported_file_types():
    with pytest.raises(ValueError, match="supported_file_types"):
        ImageParserConfig(supported_file_types=())
