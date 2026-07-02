# Document Parsers

Document Parsers convert raw file bytes into a normalized `ParsedDocument`. Downstream split, index, graph extraction, and query code do not need to know whether the original file was PDF, HTML, image, Markdown, or spreadsheet.

Parsers only parse. They do not write to storage. Object paths, versioning, and build orchestration belong to `ObjectStore`, Recipe, `KnowledgeBase`, or build steps.

## Quick Start

```python
from heta_framework.kb.parsing import TextParser, make_parsed_source

data = b"# Heta\n\nA framework-oriented knowledge base toolkit."
source = make_parsed_source(
    key="raw/readme.md",
    name="readme.md",
    file_type="md",
    data=data,
)

document = await TextParser().parse(source, data)
```

Use the result:

```python
document.document_id
document.source
document.pages[0].text
```

Write it to an ObjectStore when needed:

```python
await object_store.put(
    f"parsed/{document.document_id}.json",
    document.to_json_bytes(),
)
```

## ParsedDocument

All parsers return the same structure:

```python
ParsedDocument(
    document_id="doc_...",
    source=ParsedSource(
        key="raw/rag_paper.pdf",
        name="rag_paper.pdf",
        file_type="pdf",
        content_sha256="...",
    ),
    pages=[
        ParsedPage(
            page_index=0,
            text="full text of this page",
        )
    ],
)
```

| Object | Meaning |
| --- | --- |
| `ParsedSource` | Raw object metadata: object key, file name, file type, and content SHA-256. |
| `ParsedPage` | Page-like text unit. PDF pages, HTML pages, image descriptions, and table chunks all map to pages. |
| `ParsedDocument` | Unified parser output that can be serialized to JSON. |

`document_id` is derived from content SHA-256. The same content gets a stable ID even if the file name changes.

## Built-In Parsers

| Parser | File types | Meaning |
| --- | --- | --- |
| `TextParser` | `txt`, `text`, `md`, `markdown` | Reads plain text and Markdown. |
| `HtmlParser` | `html`, `htm` | Extracts HTML body text, tables, and optional image descriptions. |
| `PdfParser` | `pdf` | Parses PDF through a document extractor. |
| `OfficeParser` | `doc`, `docx`, `ppt`, `pptx` | Parses Office files through a document extractor. |
| `SheetParser` | `csv`, `xls`, `xlsx`, `xlsm`, `xlsb`, `ods`, `odf`, `odt` | Converts spreadsheet files into Markdown table text. |
| `ImageParser` | `jpg`, `jpeg`, `png`, `gif`, `webp`, `tiff`, `bmp`, `ico` | Uses a vision model to describe standalone image files. |

PDF and Office parsers depend on `DocumentExtractorProtocol`. The default path can use a MinerU extractor, or you can provide your own extractor.

## Parser Registry

`DocumentParserRegistry` registers the parsers you want and routes by `file_type`:

```python
from heta_framework.kb.parsing import DocumentParserRegistry, SheetParser, TextParser

registry = DocumentParserRegistry([
    TextParser(),
    SheetParser(),
])

document = await registry.parse(source, data)
```

The registry does not enable every parser by default. It supports only what you register:

```python
registry.supported_file_types
registry.find_parser("csv")
registry.get_parser("md")
```

One file type cannot be registered by multiple parsers unless replacement is explicit:

```python
registry.register(custom_text_parser, replace=True)
```

This prevents parser routing from being overwritten silently.

## ImageParser

`ImageParser` handles standalone image files uploaded directly by users.

```python
from heta_framework.common.models import LanguageModel
from heta_framework.kb.parsing import ImageParser

vision_model = LanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key="...",
)

document = await ImageParser(vision_model).parse(source, image_bytes)
```

The image is sent as multimodal input. The description is written into text:

```text
Image: chart.png

Image description: ...
```

Images produced while parsing HTML, PDF, or Office documents are handled by that parser or extractor so their position in the document context is not lost.

## SheetParser

`SheetParser` converts spreadsheet files into Markdown table text:

```python
from heta_framework.kb.parsing import SheetParser

document = await SheetParser().parse(source, data)
```

CSV uses the Python standard library. Excel and ODF files use `python-calamine`.

The sheet parser normalizes tabular text:

- Multiple sheets become multiple page-like text blocks.
- Empty headers become `column_1`.
- Dates and times become stable text.
- Integer-like floats such as `2018.0` become `2018`.
- Markdown table pipes and newlines are escaped or flattened.

It does not infer units, merge multi-row headers, classify column types, or build text-to-SQL schemas. Those belong to later table-understanding or database-building steps.

## HtmlParser

`HtmlParser` extracts page body text, title, description, and tables. Image description is optional:

```python
parser = HtmlParser(
    HtmlParserConfig(
        source_url="https://example.com/page.html",
        describe_images=True,
    ),
    vision_model=vision_model,
)
```

HTML image prompts use the same image-description rules as standalone images and include `image_url`, `alt`, or `title` when available.

## Custom Parser

Custom parsers only need to satisfy `DocumentParserProtocol`:

```python
from heta_framework.kb.parsing import ParsedDocument, ParsedPage, make_document_id

class JsonParser:
    supported_file_types = {"json"}

    async def parse(self, source, data):
        return ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=[
                ParsedPage(
                    page_index=0,
                    text=data.decode("utf-8"),
                )
            ],
        )
```

Register it:

```python
registry.register(JsonParser())
document = await registry.parse(source, data)
```

## Scope

Document Parsers convert bytes into `ParsedDocument`, preserve source metadata, normalize different formats into page-like text, and provide default image/table description capabilities.

They do not upload raw files, manage ObjectStore paths, split chunks, build vector indexes, extract graphs, write SQL tables, track lineage, or manage the `KnowledgeBase` lifecycle.
