# Extract Entities

`ExtractEntities` 从 `ParsedChunk` 中抽取图谱实体，并把结果写回 ObjectStore。

它只负责实体抽取，不抽取关系、不做实体去重，也不写入图谱库。

```text
ParsedChunk JSON -> ExtractedEntity JSON
```

## Contract

`ExtractEntities` 使用两个 recipe components：

```text
stores.objects
models.language
```

默认读取：

```text
chunk_keys
```

默认输出：

```text
extract_entities_result
entity_keys
```

写入位置：

```text
entities/{chunk_id}/{entity_id}.json
```

## Entity Shape

LLM 只输出实体语义字段：

```json
{
  "entities": [
    {
      "name": "上海市",
      "type": "客观实体",
      "subtype": "行政区划",
      "description": "上海市是中华人民共和国直辖市。",
      "attributes": {
        "所属国家": "中华人民共和国"
      }
    }
  ]
}
```

框架保存的 `ExtractedEntity` 会补齐运行时元数据：

```json
{
  "entity_id": "entity_...",
  "chunk_id": "chunk_...",
  "document_id": "doc_...",
  "name": "上海市",
  "type": "客观实体",
  "subtype": "行政区划",
  "description": "上海市是中华人民共和国直辖市。",
  "attributes": {
    "所属国家": "中华人民共和国"
  },
  "source_chunk_ids": ["chunk_..."]
}
```

这些字段由 step 生成，不由模型生成。

字段含义：

| 字段 | 说明 |
| --- | --- |
| `entity_id` | 框架生成的稳定实体记录 ID。 |
| `chunk_id` | 当前用于实体抽取的 chunk ID。 |
| `document_id` | 实体来源文档 ID。 |
| `name` | 实体名称，对应旧版 HetaDB 的 `NodeName`。 |
| `type` | 实体一级类型。 |
| `subtype` | 实体细分类型；无法可靠判断时为 `null`。 |
| `description` | 实体描述，用于展示、embedding 和后续去重判断。 |
| `attributes` | 实体属性字典，对应旧版 HetaDB 中平铺的业务属性或 `Attr`。 |
| `source_chunk_ids` | 原始证据 chunk IDs。输入为 rechunked chunk 时继承 `parent_chunk_ids`。 |

`ExtractEntitiesResult` 写入运行上下文，不落 ObjectStore：

```python
ExtractEntitiesResult(
    entity_keys=("entities/chunk_.../entity_....json",),
    chunk_count=1,
    entity_count=1,
    failed_chunk_ids=(),
)
```

## Configuration

```python
ExtractEntitiesConfig(
    entities_prefix="entities",
    chunk_keys_artifact="chunk_keys",
    max_attempts=3,
    temperature=0.0,
)
```

| 参数 | 说明 |
| --- | --- |
| `entities_prefix` | 实体 JSON 写入 ObjectStore 的前缀。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称。 |
| `max_attempts` | 单个 chunk 的最大抽取尝试次数。 |
| `temperature` | 实体抽取请求的模型温度。 |

如果模型输出无法解析或不满足协议，step 会带着错误原因重试。达到 `max_attempts` 后，该 chunk 会进入 `failed_chunk_ids`，不会中断整个构建流程。

## Input Choice

直接基于原始 chunk 建图：

```python
ExtractEntities(
    ExtractEntitiesConfig(chunk_keys_artifact="chunk_keys")
)
```

基于 rechunk 后文本建图：

```python
ExtractEntities(
    ExtractEntitiesConfig(chunk_keys_artifact="rechunked_chunk_keys")
)
```

当输入 chunk 带有 `parent_chunk_ids` 时，`source_chunk_ids` 会继承这些原始 chunk id；否则使用当前 `chunk_id`。这保证后续关系抽取、去重和图谱落库仍然可以回溯到原始证据。
