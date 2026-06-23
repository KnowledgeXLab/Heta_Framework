import asyncio
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.extractors import DocumentExtractorProtocol, DocumentInput  # noqa: E402
from heta_framework.common.extractors.mineru import (  # noqa: E402
    MinerUClient,
    MinerUClientConfig,
    mineru_artifact_to_extracted_document,
    parse_mineru_zip,
)
from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserProtocol,
    OfficeParser,
    PdfParser,
    make_parsed_source,
)


def test_mineru_zip_maps_markdown_order_to_extracted_blocks():
    artifact = parse_mineru_zip(
        _mineru_zip(
            "# Title\n\nIntro text\n\n"
            "<table><tr><td>A</td><td>B</td></tr></table>\n\n"
            "Before figure\n\n"
            "![](images/figure.jpg)\n\n"
            "Figure caption",
            content_list=[
                {
                    "type": "text",
                    "text": "Title",
                    "page_idx": 0,
                    "bbox": [1, 2, 3, 4],
                },
                {
                    "type": "table",
                    "table_caption": ["Scores"],
                    "table_body": "<table><tr><td>A</td><td>B</td></tr></table>",
                    "page_idx": 0,
                    "bbox": [10, 20, 30, 40],
                },
                {
                    "type": "image",
                    "img_path": "images/figure.jpg",
                    "page_idx": 1,
                    "bbox": [100, 200, 300, 400],
                },
            ],
            images={"images/figure.jpg": b"jpg-bytes"},
        )
    )

    document = mineru_artifact_to_extracted_document(artifact)

    assert [block.kind for block in document.blocks] == ["text", "table", "text", "image", "text"]
    assert document.blocks[1].text.startswith("Scores\n<table>")
    assert document.blocks[1].page_index == 0
    assert document.blocks[3].page_index == 1
    assert document.blocks[3].asset is not None
    assert document.blocks[3].asset.key == "artifacts/images/figure.jpg"
    assert document.blocks[3].asset.data == b"jpg-bytes"
    assert document.blocks[3].asset.content_sha256 is not None
    rendered = document.to_text()
    assert "Before figure" in rendered
    assert "Image: artifacts/images/figure.jpg" in rendered
    assert rendered.index("Before figure") < rendered.index("Image: artifacts/images/figure.jpg")
    assert document.metadata["unmatched_table_count"] == 0


def test_mineru_zip_uses_content_list_v2_for_page_indexes():
    artifact = parse_mineru_zip(
        _mineru_zip(
            "# Page 1\n\nPage 2",
            content_list_v2=[
                [
                    {
                        "type": "title",
                        "content": {
                            "title_content": [{"type": "text", "content": "Page 1"}],
                            "level": 1,
                        },
                        "bbox": [1, 2, 3, 4],
                    }
                ],
                [
                    {
                        "type": "paragraph",
                        "content": {
                            "paragraph_content": [{"type": "text", "content": "Page 2"}],
                        },
                        "bbox": [5, 6, 7, 8],
                    }
                ],
            ],
        )
    )

    document = mineru_artifact_to_extracted_document(artifact)

    assert document.metadata["used_content_list"] == "v2"
    assert [block.page_index for block in document.blocks] == [0, 1]
    assert [block.text for block in document.blocks] == ["Page 1", "Page 2"]


def test_pdf_parser_satisfies_protocol_and_groups_pages():
    parser = PdfParser(FakeExtractor())
    data = b"%PDF"
    source = make_parsed_source(key="raw/doc.pdf", name="doc.pdf", file_type="pdf", data=data)

    async def run():
        return await parser.parse(source, data)

    parsed = asyncio.run(run())

    assert isinstance(parser, DocumentParserProtocol)
    assert parsed.document_id.startswith("doc_")
    assert [page.page_index for page in parsed.pages] == [0, 1]
    assert "Intro" in parsed.pages[0].text
    assert "Table:" in parsed.pages[0].text
    assert "Image: artifacts/images/figure.jpg" in parsed.pages[1].text
    assert "Image description: Figure description" in parsed.pages[1].text


def test_office_parser_uses_document_extractor():
    parser = OfficeParser(FakeExtractor())
    data = b"pptx"
    source = make_parsed_source(key="raw/slides.pptx", name="slides.pptx", file_type="pptx", data=data)

    async def run():
        return await parser.parse(source, data)

    parsed = asyncio.run(run())

    assert isinstance(parser, DocumentParserProtocol)
    assert "Intro" in parsed.pages[0].text


def test_pdf_parser_rejects_office_file_type():
    parser = PdfParser(FakeExtractor())
    data = b"pptx"
    source = make_parsed_source(key="raw/slides.pptx", name="slides.pptx", file_type="pptx", data=data)

    async def run():
        return await parser.parse(source, data)

    with pytest.raises(ValueError, match="PdfParser"):
        asyncio.run(run())


def test_mineru_client_uses_local_tasks_endpoint_by_default():
    zip_content = _mineru_zip("# Local\n\nParsed")
    client = FakeAsyncClient(
        [
            httpx.Response(200, json={"task_id": "task-1"}),
            httpx.Response(200, json={"status": "completed"}),
            httpx.Response(
                200,
                content=zip_content,
                headers={"content-type": "application/zip"},
            ),
        ]
    )
    mineru = MinerUClient(
        MinerUClientConfig(
            provider="local",
            endpoint_url="http://127.0.0.1:8000",
            parse_timeout=1,
            poll_interval=0.01,
        ),
        client=client,
    )

    async def run():
        return await mineru.extract(DocumentInput(data=b"%PDF", filename="paper.pdf"))

    document = asyncio.run(run())

    assert document.to_text() == "# Local\n\nParsed"
    assert [request["method"] for request in client.requests] == ["POST", "GET", "GET"]
    assert client.requests[0]["url"] == "http://127.0.0.1:8000/tasks"
    assert client.requests[0]["data"]["response_format_zip"] == "true"
    assert client.requests[0]["data"]["effort"] == "medium"
    assert client.requests[1]["url"] == "http://127.0.0.1:8000/tasks/task-1"
    assert client.requests[2]["url"] == "http://127.0.0.1:8000/tasks/task-1/result"


def test_mineru_client_can_use_legacy_local_file_parse_endpoint():
    client = FakeAsyncClient(
        [
            httpx.Response(
                200,
                content=_mineru_zip("# Local\n\nParsed"),
                headers={"content-type": "application/zip"},
            )
        ]
    )
    mineru = MinerUClient(
        MinerUClientConfig(
            provider="local",
            endpoint_url="http://127.0.0.1:8000",
            local_api_mode="file_parse",
        ),
        client=client,
    )

    async def run():
        return await mineru.extract(DocumentInput(data=b"%PDF", filename="paper.pdf"))

    document = asyncio.run(run())

    assert document.to_text() == "# Local\n\nParsed"
    assert client.requests[0]["method"] == "POST"
    assert client.requests[0]["url"] == "http://127.0.0.1:8000/file_parse"
    assert client.requests[0]["data"]["response_format_zip"] == "true"
    assert "files" in client.requests[0]


def test_mineru_client_uses_cloud_batch_flow():
    zip_content = _mineru_zip("# Cloud\n\nParsed")
    client = FakeAsyncClient(
        [
            httpx.Response(
                200,
                json={"code": 0, "data": {"batch_id": "batch-1", "file_urls": ["https://upload"]}},
            ),
            httpx.Response(200),
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "paper.pdf",
                                "state": "done",
                                "full_zip_url": "https://result.zip",
                            }
                        ]
                    },
                },
            ),
            httpx.Response(200, content=zip_content),
        ]
    )
    mineru = MinerUClient(
        MinerUClientConfig(provider="cloud", api_key="token", parse_timeout=1, poll_interval=0.01),
        client=client,
    )

    async def run():
        return await mineru.extract(DocumentInput(data=b"%PDF", filename="paper.pdf"))

    document = asyncio.run(run())

    assert document.to_text() == "# Cloud\n\nParsed"
    assert [request["method"] for request in client.requests] == ["POST", "PUT", "GET", "GET"]
    assert client.requests[0]["json"]["files"][0]["name"] == "paper.pdf"
    assert client.requests[0]["headers"]["Authorization"] == "Bearer token"
    assert client.requests[1]["url"] == "https://upload"


def test_mineru_config_validation():
    with pytest.raises(ValueError, match="api_key"):
        MinerUClientConfig(provider="cloud")
    with pytest.raises(ValueError, match="endpoint_url"):
        MinerUClientConfig(provider="local")


def _mineru_zip(
    markdown: str,
    *,
    content_list: list[dict] | None = None,
    content_list_v2: list[list[dict]] | None = None,
    images: dict[str, bytes] | None = None,
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("full.md", markdown)
        if content_list is not None:
            archive.writestr("demo_content_list.json", json.dumps(content_list))
        if content_list_v2 is not None:
            archive.writestr("demo_content_list_v2.json", json.dumps(content_list_v2))
        for name, data in (images or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


class FakeExtractor:
    async def extract(self, document, options=None):
        del document, options
        artifact = parse_mineru_zip(
            _mineru_zip(
                "Intro\n\n<table><tr><td>A</td></tr></table>\n\n![](images/figure.jpg)",
                content_list=[
                    {
                        "type": "table",
                        "table_body": "<table><tr><td>A</td></tr></table>",
                        "page_idx": 0,
                    },
                    {
                        "type": "image",
                        "img_path": "images/figure.jpg",
                        "page_idx": 1,
                        "image_caption": ["Figure description"],
                    },
                ],
                images={"images/figure.jpg": b"jpg"},
            )
        )
        return mineru_artifact_to_extracted_document(artifact)


class FakeAsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        response = self._responses.pop(0)
        response.request = httpx.Request(method, url)
        return response


def test_fake_extractor_satisfies_protocol():
    assert isinstance(FakeExtractor(), DocumentExtractorProtocol)
