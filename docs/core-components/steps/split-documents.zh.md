# Split Documents

`SplitDocuments` 将 `ParsedDocument` JSON 切分为检索、索引和图谱抽取使用的 `ParsedChunk` JSON。

```text
ParsedDocument JSON -> ParsedChunk JSON
```

它只负责 chunk 生成。Embedding、向量索引、图谱抽取、chunk merge 和 rechunk 都属于后续 step。

## Contract

`SplitDocuments` 使用一个 recipe component：

```text
stores.objects
```

默认读取上游 artifact：

```text
parsed_document_keys
```

默认写入路径：

```text
chunks/
```

执行流程：

```text
read parsed_document_keys
  -> load ParsedDocument JSON from ObjectStore
  -> split each page text into token windows
  -> write chunks/{chunk_id}.json
  -> expose chunk keys as artifacts
```

`SplitDocuments` 不读取 raw 文件，也不调用 parser。它只消费 `ParseDocuments` 产生的 parsed document keys。

## Configuration

```python
SplitDocumentsConfig(
    chunks_prefix="chunks",
    chunk_size=1024,
    overlap=50,
    encoding_name="cl100k_base",
    split_punctuation=("。", ".", ",", "，", "!", "?", "！", "？", "\n"),
    object_store=None,
    parsed_document_keys_artifact="parsed_document_keys",
)
```

| 参数 | 说明 |
| --- | --- |
| `chunks_prefix` | `ParsedChunk` JSON 写入 prefix。 |
| `chunk_size` | 每个 chunk 的目标 token 数。 |
| `overlap` | 相邻 chunk 的 token overlap。必须小于 `chunk_size`。 |
| `encoding_name` | tokenizer 名称。默认 `cl100k_base`。 |
| `split_punctuation` | 优先截断的标点集合。 |
| `object_store` | 命名 ObjectStore。默认引用 `stores.objects`。 |
| `parsed_document_keys_artifact` | 上游 parsed document key artifact 名称。 |

`encoding_name="unicode"` 是内置离线 tokenizer，适合测试或无网络环境。生产默认使用 `cl100k_base`，与 HetaDB 原始 chunker 对齐。

## Requirements

默认 requirements：

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
    }),
    artifacts=frozenset({
        "parsed_document_keys",
    }),
)
```

含义：

| Requirement | 说明 |
| --- | --- |
| `stores.objects` | 满足 `ObjectStoreProtocol` 的对象存储。 |
| `parsed_document_keys` | `ParseDocuments` 产生的 parsed JSON key 列表。 |

如果配置了命名 ObjectStore，例如 `object_store="local"`，component reference 会变成：

```text
stores.objects.local
```

## Capabilities

`SplitDocuments` 提供两个 artifacts：

```python
StepCapabilities(
    artifacts=frozenset({
        "split_documents_result",
        "chunk_keys",
    })
)
```

它不直接提供 query mode。查询能力由后续 `IndexVectors`、`IndexFullText`、`BuildGraph` 或其它索引 step 提供。

## Artifacts

`split_documents_result` 是 `SplitDocumentsResult`：

```python
SplitDocumentsResult(
    chunk_keys=(
        "chunks/chunk_abc123.json",
    ),
    document_count=1,
    chunk_count=12,
)
```

| 字段 | 说明 |
| --- | --- |
| `chunk_keys` | 已写入 ObjectStore 的 `ParsedChunk` JSON keys。 |
| `document_count` | 本次处理的 parsed document 数量。 |
| `chunk_count` | 本次生成的 chunk 数量。 |

`chunk_keys` 是 `SplitDocumentsResult.chunk_keys` 的快捷 tuple，方便后续 embedding、index 或 graph step 直接读取 chunk JSON。

## Chunk Output

每个写入的 JSON 都是 `ParsedChunk`：

```python
ParsedChunk(
    chunk_id="chunk_...",
    document_id="doc_...",
    source=ParsedSource(
        key="raw/paper.pdf",
        name="paper.pdf",
        file_type="pdf",
        content_sha256="...",
    ),
    page_index=0,
    chunk_index=0,
    text="...",
    token_start=0,
    token_end=512,
)
```

ObjectStore 中的默认位置：

```text
chunks/{chunk_id}.json
```

`chunk_id` 由 document id、page index、chunk index 和文本内容生成。这样不同文档中相同文本不会意外得到同一个 chunk id。

字段说明：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | chunk 的稳定 ID。后续 embedding、vector store、graph relation、citation 都通过它引用 chunk。 |
| `document_id` | chunk 所属的 `ParsedDocument` ID。用于按文档聚合、删除、重建和回溯。 |
| `source` | 原始文件信息。用于展示来源、过滤文件类型和生成 citation。 |
| `page_index` | chunk 来自哪个 page-like 单元。PDF 页、HTML 页面、图片描述和表格 chunk 都会映射为 page-like 单元。 |
| `chunk_index` | chunk 在 document 内的顺序。用于相邻 chunk 扩展、上下文拼接和稳定排序。 |
| `text` | chunk 文本。后续 embedding、BM25、LLM context 和图谱抽取都会读取它。 |
| `token_start` | chunk 在当前 page token 序列中的起始位置。用于调试边界、评估 overlap、定位原文和后续 rechunk。 |
| `token_end` | chunk 在当前 page token 序列中的结束位置。与 `token_start` 一起描述 chunk 的精确 token 范围。 |

`token_start` 和 `token_end` 是位置元信息，不是业务字段。它们由 splitter 直接产生，计算成本接近 0，存储开销很小，但能让 chunk 保持可追踪、可调试和可重组。

## Splitting Behavior

默认切分策略与 HetaDB 原始 chunker 对齐：

```text
chunk_size = 1024
overlap = 50
encoding_name = cl100k_base
split_punctuation = 。 . , ， ! ? ！ ？ \n
```

切分过程：

```text
encode page text
  -> take token window
  -> prefer the last punctuation boundary inside the window
  -> trim empty text
  -> move next start by token_end - overlap
```

每个 `ParsedPage` 独立切分。`chunk_index` 在同一个 document 内递增。

## HetaDB Alignment

`SplitDocuments` 对齐 HetaDB 的基础 chunk 逻辑：

- token-based splitting
- `cl100k_base`
- `chunk_size=1024`
- `overlap=50`
- punctuation-aware boundary
- stable chunk id

Framework 版本没有复刻 HetaDB 的整段 chunk/rechunk stage。HetaDB 原流程还包含：

- embedding API 调用
- Milvus 写入
- PostgreSQL chunk table 写入
- LLM-based chunk merge
- rechunk by source

这些能力在 Heta Framework 中应拆成独立 step，例如：

```text
EmbedChunks
IndexVectors
MergeChunks
RechunkDocuments
PersistChunks
```

`SplitDocuments` 只负责 `ParsedDocument -> ParsedChunk`。
