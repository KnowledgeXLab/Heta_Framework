import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.extractors import (  # noqa: E402
    DocumentInput,
    ExtractedBlock,
    ExtractedDocument,
)
from heta_framework.kb.parsing import (  # noqa: E402
    BasicHtmlExtractor,
    DocumentParserProtocol,
    HtmlParser,
    HtmlParserConfig,
    make_document_id,
    make_parsed_source,
)


def test_html_parser_satisfies_protocol():
    assert isinstance(HtmlParser(), DocumentParserProtocol)


def test_basic_html_extractor_extracts_title_description_and_text():
    html = """
    <html>
      <head>
        <title>Article</title>
        <meta name="description" content="Useful summary">
      </head>
      <body>
        <aside>Related links</aside>
        <main><h1>Main title</h1><p>Main body</p></main>
      </body>
    </html>
    """

    async def run():
        return await BasicHtmlExtractor().extract(
            DocumentInput(data=html.encode("utf-8"), filename="article.html", media_type="text/html")
        )

    extracted = asyncio.run(run())

    assert extracted.metadata["title"] == "Article"
    assert extracted.metadata["description"] == "Useful summary"
    assert "Main title" in extracted.to_text()
    assert "Main body" in extracted.to_text()
    assert "Related links" not in extracted.to_text()


def test_html_parser_extracts_text_images_and_tables():
    html = """
    <html>
      <head>
        <title>RAG Paper</title>
        <meta name="description" content="A retrieval paper">
        <style>.hidden { display: none; }</style>
        <script>ignored()</script>
      </head>
      <body>
        <nav>Navigation should disappear</nav>
        <h1>Introduction</h1>
        <p>Heta extracts knowledge.</p>
        <img src="/images/chart.png" alt="Accuracy chart">
        <table>
          <caption>Scores</caption>
          <tr><th>Model</th><th>Score</th></tr>
          <tr><td>Heta</td><td>99</td></tr>
        </table>
        <div style="background-image: url('/images/bg.jpg')">Hero</div>
        <picture>
          <source srcset="/images/small.webp 1x, /images/large.webp 2x">
        </picture>
      </body>
    </html>
    """.encode("utf-8")
    source = make_parsed_source(
        key="raw/page.html",
        name="page.html",
        file_type="html",
        data=html,
    )

    async def run():
        parser = HtmlParser(HtmlParserConfig(source_url="https://example.com/docs/"))
        return await parser.parse(source, html)

    document = asyncio.run(run())
    text = document.pages[0].text

    assert document.document_id == make_document_id(source.content_sha256)
    assert document.source == source
    assert document.pages[0].page_index == 0
    assert "Title: RAG Paper" in text
    assert "Description: A retrieval paper" in text
    assert "Introduction" in text
    assert "Heta extracts knowledge." in text
    assert "Navigation should disappear" not in text
    assert "ignored" not in text
    assert "Image: Accuracy chart (https://example.com/images/chart.png)" in text
    assert "Table: Scores" in text
    assert "Model | Score" in text
    assert "Heta | 99" in text
    assert "Background image: https://example.com/images/bg.jpg" in text
    assert "Image source: https://example.com/images/small.webp" in text
    assert "https://example.com/images/large.webp" in text


def test_html_parser_can_describe_cleaned_images_with_vlm():
    html = """
    <html>
      <body>
        <img src="/images/chart.png" alt="Accuracy chart">
        <img src="/images/logo.png" width="32" height="32">
        <img src="/images/tracker-pixel.png">
        <img src="/images/fullaccess_300.webp">
        <nav><img src="/images/nav-chart.png" alt="Navigation image"></nav>
        <p>Body text</p>
      </body>
    </html>
    """.encode("utf-8")
    source = make_parsed_source(key="raw/page.html", name="page.html", file_type="html", data=html)
    model = FakeVisionModel()

    async def run():
        parser = HtmlParser(
            HtmlParserConfig(source_url="https://example.com", describe_images=True),
            vision_model=model,
        )
        return await parser.parse(source, html)

    document = asyncio.run(run())
    text = document.pages[0].text

    assert model.urls == ["https://example.com/images/chart.png"]
    assert "detailed, retrieval-friendly description" in model.prompts[0]
    assert "image_url: https://example.com/images/chart.png" in model.prompts[0]
    assert "existing_caption_or_alt: Accuracy chart" in model.prompts[0]
    assert "Image: Accuracy chart (https://example.com/images/chart.png)" in text
    assert "Image description: described https://example.com/images/chart.png" in text
    assert "logo.png" not in text
    assert "tracker-pixel.png" not in text
    assert "fullaccess_300.webp" not in text
    assert "Navigation image" not in text


def test_html_parser_requires_vision_model_when_describing_images():
    with pytest.raises(ValueError, match="vision_model"):
        HtmlParser(HtmlParserConfig(describe_images=True))


def test_html_parser_uses_og_description_when_needed():
    html = """
    <html>
      <head><meta property="og:description" content="OG description"></head>
      <body><p>Body text</p></body>
    </html>
    """.encode("utf-8")
    source = make_parsed_source(key="raw/page.htm", name="page.htm", file_type="htm", data=html)

    async def run():
        return await HtmlParser().parse(source, html)

    document = asyncio.run(run())

    assert "Description: OG description" in document.pages[0].text
    assert "Body text" in document.pages[0].text


def test_html_parser_rejects_unsupported_file_type():
    data = b"<p>hello</p>"
    source = make_parsed_source(
        key="raw/doc.txt",
        name="doc.txt",
        file_type="txt",
        data=data,
    )

    async def run():
        return await HtmlParser().parse(source, data)

    with pytest.raises(ValueError, match="unsupported file type"):
        asyncio.run(run())


def test_html_parser_accepts_document_extractor_protocol():
    html = b"<html><body>Raw</body></html>"
    source = make_parsed_source(key="raw/page.html", name="page.html", file_type="html", data=html)
    extractor = FakeDocumentExtractor()

    async def run():
        return await HtmlParser(extractor=extractor).parse(source, html)

    document = asyncio.run(run())

    assert extractor.inputs == [("page.html", "text/html", html)]
    assert document.pages[0].text == (
        "MinerU parsed HTML\n\n"
        "Image: artifacts/images/chart.jpg\n\n"
        "Image description: Chart description"
    )


class FakeDocumentExtractor:
    def __init__(self):
        self.inputs = []

    async def extract(self, document, options=None):
        del options
        self.inputs.append((document.filename, document.media_type, document.data))
        return ExtractedDocument(
            blocks=(
                ExtractedBlock(kind="text", text="MinerU parsed HTML"),
                ExtractedBlock(
                    kind="image",
                    text="Chart description",
                    asset=SimpleNamespace(key="artifacts/images/chart.jpg", name="chart.jpg"),
                ),
            )
        )


class FakeVisionModel:
    @property
    def model_name(self):
        return "fake-vision"

    def __init__(self):
        self.urls = []
        self.prompts = []

    async def invoke(self, request):
        self.prompts.append(request.content[0].text)
        image_part = request.content[1]
        self.urls.append(image_part.url)
        return SimpleNamespace(text=f"described {image_part.url}")

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    def stream(self, request):
        del request
        raise NotImplementedError
