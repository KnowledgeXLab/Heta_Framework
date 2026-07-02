# Document Parsers

Document Parsers 把原始文件 bytes 转换为统一的 `ParsedDocument`。下游的 split、index、graph extraction 和 query 不需要关心文件来自 PDF、HTML、图片、Markdown 还是表格。

Parser 只负责解析，不负责写入存储。写入路径、版本管理和构建编排由 `ObjectStore`、Recipe、KnowledgeBase 或 build steps 负责。

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

返回结果：

```python
document.document_id
document.source
document.pages[0].text
```

如果需要写入 ObjectStore，可以把结果序列化：

```python
await object_store.put(
    f"parsed/{document.document_id}.json",
    document.to_json_bytes(),
)
```

## ParsedDocument

所有 parser 输出同一个结构：

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
            text="这一页完整文本",
        )
    ],
)
```

| 对象 | 说明 |
| --- | --- |
| `ParsedSource` | 原始对象元信息，包括 object key、文件名、文件类型和内容 SHA-256。 |
| `ParsedPage` | page-like 文本单元。真实 PDF 页面、HTML 页面、图片描述和表格 chunk 都会映射为 page。 |
| `ParsedDocument` | parser 的统一输出，可序列化为 JSON。 |

`document_id` 由内容 SHA-256 生成。相同内容会得到稳定 ID；文件名变化不会改变内容 ID。

## Built-In Parsers

| Parser | 文件类型 | 说明 |
| --- | --- | --- |
| `TextParser` | `txt`, `text`, `md`, `markdown` | 读取纯文本和 Markdown。 |
| `HtmlParser` | `html`, `htm` | 解析 HTML 主体文本、表格和可选图片描述。 |
| `PdfParser` | `pdf` | 基于 document extractor 解析 PDF。 |
| `OfficeParser` | `doc`, `docx`, `ppt`, `pptx` | 基于 document extractor 解析 Office 文件。 |
| `SheetParser` | `csv`, `xls`, `xlsx`, `xlsm`, `xlsb`, `ods`, `odf`, `odt` | 将表格文件转换为 Markdown 表格文本。 |
| `ImageParser` | `jpg`, `jpeg`, `png`, `gif`, `webp`, `tiff`, `bmp`, `ico` | 使用视觉模型描述独立图片文件。 |

PDF 和 Office parser 依赖 `DocumentExtractorProtocol`。默认场景中可以接入 MinerU extractor，也可以传入用户自己的 extractor。

## Parser Registry

`DocumentParserRegistry` 用于注册用户需要的 parser，并按 `file_type` 自动路由：

```python
from heta_framework.kb.parsing import DocumentParserRegistry, SheetParser, TextParser

registry = DocumentParserRegistry([
    TextParser(),
    SheetParser(),
])

document = await registry.parse(source, data)
```

Registry 不会默认启用所有 parser。用户注册什么，registry 就支持什么：

```python
registry.supported_file_types
registry.find_parser("csv")
registry.get_parser("md")
```

同一个 file type 默认不允许被多个 parser 同时注册。需要替换已有 parser 时显式声明：

```python
registry.register(custom_text_parser, replace=True)
```

这个设计避免 parser 路由被悄悄覆盖。

## ImageParser

`ImageParser` 只处理独立图片文件，例如用户直接上传的 `png`、`jpg` 或 `webp`。

```python
from heta_framework.common.models import LanguageModel
from heta_framework.kb.parsing import ImageParser

vision_model = LanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key="...",
)

document = await ImageParser(vision_model).parse(source, image_bytes)
```

图片会作为多模态输入发送给视觉模型，返回文本写入：

```text
Image: chart.png

Image description: ...
```

HTML、PDF 或 Office 解析过程中产生的内部图片不由 `ImageParser` 接管。它们应由对应 parser 或 extractor 在原始上下文中处理，避免破坏图片在文档中的位置关系。

默认图片 prompt 由 `kb.parsing.prompts` 管理。用户可以覆盖：

```python
from heta_framework.kb.parsing import ImageParserConfig

parser = ImageParser(
    vision_model,
    config=ImageParserConfig(prompt="请详细描述图片中的文字、图表和空间关系。"),
)
```

## SheetParser

`SheetParser` 将表格文件转换为 Markdown 表格文本：

```python
from heta_framework.kb.parsing import SheetParser

document = await SheetParser().parse(source, data)
```

CSV 使用 Python 标准库读取。Excel 和 ODF 文件使用 `python-calamine` 读取，覆盖 `xls`、`xlsx`、`xlsm`、`xlsb`、`ods`、`odf` 和 `odt`。

表格 parser 只做格式读取和文本表示规范化：

- 多 sheet 输出为多个 page-like 文本块。
- 空表头补为 `column_1`。
- 日期和时间转换为稳定文本。
- `2018.0` 这类整数型 float 输出为 `2018`。
- Markdown 表格中的 `|` 和换行会被转义或压平。

表格 parser 不做表格语义推断，例如单位识别、多行表头合并、字段类型判断或 text-to-sql schema 构建。这些能力应放在后续的表格理解或数据库构建步骤中。

可以让语言模型为表格生成描述：

```python
parser = SheetParser(
    SheetParserConfig(describe_tables=True),
    language_model=llm,
)
```

描述会出现在表格第一页的 `Description:` 段落中。

## HtmlParser

`HtmlParser` 解析网页主体文本、标题、描述和表格。图片描述是可选能力：

```python
parser = HtmlParser(
    HtmlParserConfig(
        source_url="https://example.com/page.html",
        describe_images=True,
    ),
    vision_model=vision_model,
)
```

HTML 图片 prompt 与独立图片共用同一套图片描述准则，并额外携带 `image_url`、`alt` 或 `title` 作为提示。网页上下文只作为辅助信息；如果上下文与图片可见内容冲突，以图片内容为准。

## Custom Parser

自定义 parser 不需要继承父类，只要满足 `DocumentParserProtocol`：

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

注册后即可参与路由：

```python
registry.register(JsonParser())
document = await registry.parse(source, data)
```

## Scope

Document Parsers 负责：

- 将原始文件 bytes 转换为 `ParsedDocument`。
- 保留原始来源元信息。
- 将不同格式统一为 page-like 文本。
- 为图片和表格提供必要的默认描述能力。

Document Parsers 不负责：

- 原始文件上传和落盘。
- ObjectStore 写入路径管理。
- Chunk 切分。
- 向量索引、图谱抽取或 SQL 建库。
- 版本管理、血缘追踪或 `KnowledgeBase` 生命周期。

这些能力应由 ObjectStore、Recipe、KnowledgeBase 或后续构建步骤承担。
