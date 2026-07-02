# Parse Documents

`ParseDocuments` 是知识库构建链路的入口 step。它从 ObjectStore 读取原始文件，并通过 `DocumentParserRegistry` 解析为统一的 `ParsedDocument` JSON。

```text
raw objects -> ParsedDocument JSON
```

后续 step 不再关心原文件来自 PDF、HTML、图片还是纯文本，只消费统一的 parsed document。

## Contract

`ParseDocuments` 使用两个 recipe components：

```text
stores.objects
parsers.documents
```

默认读取和写入路径：

```text
raw/
parsed/
```

执行流程：

```text
list raw objects
  -> infer file_type from object key
  -> route through DocumentParserRegistry
  -> produce ParsedDocument
  -> write parsed/{document_id}.json
  -> expose parsed document keys as artifacts
```

`ParseDocuments` 不创建 parser registry，也不创建 object store。它只声明依赖，由 recipe 在运行时提供对应组件。

## Configuration

```python
ParseDocumentsConfig(
    raw_prefix="raw",
    parsed_prefix="parsed",
    skip_unsupported=True,
    object_store=None,
    parser_registry=None,
)
```

| 参数 | 说明 |
| --- | --- |
| `raw_prefix` | 原始文件所在 ObjectStore prefix。 |
| `parsed_prefix` | `ParsedDocument` JSON 写入 prefix。 |
| `skip_unsupported` | 没有匹配 parser 时是否跳过。设为 `False` 会报错。 |
| `object_store` | 命名 ObjectStore。默认引用 `stores.objects`。 |
| `parser_registry` | 命名 parser registry。默认引用 `parsers.documents`。 |

命名组件会改变 component reference：

```python
ParseDocumentsConfig(
    object_store="local",
    parser_registry="strict",
)
```

对应引用：

```text
stores.objects.local
parsers.documents.strict
```

## Requirements

默认 requirements：

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        parser_ref(),
    })
)
```

含义：

| Component ref | 需要的组件 |
| --- | --- |
| `stores.objects` | 满足 `ObjectStoreProtocol` 的对象存储。 |
| `parsers.documents` | `DocumentParserRegistry`。 |

如果配置了命名组件，requirements 会引用命名 key，例如：

```text
stores.objects.local
parsers.documents.strict
```

## Capabilities

完成后提供两个 artifacts：

```python
StepCapabilities(
    artifacts=frozenset({
        "parse_documents_result",
        "parsed_document_keys",
    })
)
```

它不直接提供 query mode。查询能力由后续索引或图谱 step 解锁。

## Artifacts

`parse_documents_result` 是 `ParseDocumentsResult`：

```python
ParseDocumentsResult(
    document_keys=(
        "parsed/doc_abc123.json",
    ),
    skipped_keys=(
        "raw/archive.zip",
    ),
)
```

| 字段 | 说明 |
| --- | --- |
| `document_keys` | 已写入 ObjectStore 的 `ParsedDocument` JSON keys。 |
| `skipped_keys` | 因没有匹配 parser 而跳过的 raw object keys。 |

`parsed_document_keys` 是 `document_keys` 的快捷 tuple，方便下游 step 直接读取 parsed JSON。

## Parsed Output

每个写入的 JSON 都是 `ParsedDocument`：

```python
ParsedDocument(
    document_id="doc_...",
    source=ParsedSource(
        key="raw/paper.pdf",
        name="paper.pdf",
        file_type="pdf",
        content_sha256="...",
    ),
    pages=[
        ParsedPage(
            page_index=0,
            text="...",
        )
    ],
)
```

ObjectStore 中的默认位置：

```text
parsed/{document_id}.json
```

## Unsupported Files

文件类型是否支持由 `DocumentParserRegistry` 决定：

```text
file suffix -> file_type -> registry.find_parser(file_type)
```

没有匹配 parser 时：

| 配置 | 行为 |
| --- | --- |
| `skip_unsupported=True` | 跳过文件并记录到 `skipped_keys`。 |
| `skip_unsupported=False` | 抛出错误。 |

这个设计允许 recipe 只启用需要的 parser，而不是默认加载所有解析能力。
