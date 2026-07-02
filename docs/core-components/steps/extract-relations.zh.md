# Extract Relations

`ExtractRelations` 从 `ParsedChunk` 和同一 extraction unit 内的 `ExtractedEntity` 中抽取图谱关系。

它只负责关系抽取，不做关系去重、不写图谱库，也不生成 embedding。

```text
ParsedChunk JSON + ExtractedEntity JSON -> ExtractedRelation JSON
```

这里的 `chunk_id` 表示当前关系抽取使用的 extraction unit。它可以是原始 chunk，也可以是 `RechunkDocuments` 产出的 rechunked chunk。原始证据来源由 `source_chunk_ids` 表示。

## Contract

`ExtractRelations` 使用两个 recipe components：

```text
stores.objects
models.language
```

默认读取：

```text
chunk_keys
entity_keys
```

默认输出：

```text
extract_relations_result
relation_keys
```

写入位置：

```text
relations/{chunk_id}/{relation_id}.json
```

## Relation Shape

LLM 只输出关系语义字段：

```json
{
  "relations": [
    {
      "source": "上海市",
      "target": "徐汇区",
      "type": "空间关系",
      "name": "包含行政区",
      "description": "徐汇区是上海市下辖的市辖区。",
      "attributes": {}
    }
  ]
}
```

`source` 和 `target` 必须匹配当前 chunk 已抽取出的实体名称。模型不能在关系抽取阶段新增实体。

框架保存的 `ExtractedRelation` 会补齐运行时元数据：

```json
{
  "relation_id": "relation_...",
  "chunk_id": "chunk_...",
  "document_id": "doc_...",
  "source_entity_id": "entity_...",
  "target_entity_id": "entity_...",
  "source_entity_name": "上海市",
  "target_entity_name": "徐汇区",
  "type": "空间关系",
  "name": "包含行政区",
  "description": "徐汇区是上海市下辖的市辖区。",
  "attributes": {},
  "source_chunk_ids": ["chunk_..."]
}
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `relation_id` | 框架生成的稳定关系记录 ID。 |
| `chunk_id` | 当前用于关系抽取的 extraction unit ID。 |
| `document_id` | 关系来源文档 ID。 |
| `source_entity_id` | 关系起点实体 ID。 |
| `target_entity_id` | 关系终点实体 ID。 |
| `source_entity_name` | 关系起点实体名称。 |
| `target_entity_name` | 关系终点实体名称。 |
| `type` | 关系一级类型。 |
| `name` | 具体关系名称，对应旧版 HetaDB 的 `Relation`。 |
| `description` | 关系描述，用于展示、embedding 和后续去重判断。 |
| `attributes` | 关系属性字典。 |
| `source_chunk_ids` | 原始证据 chunk IDs。输入为 rechunked chunk 时继承 `parent_chunk_ids`。 |

`ExtractRelationsResult` 写入运行上下文，不落 ObjectStore：

```python
ExtractRelationsResult(
    relation_keys=("relations/chunk_.../relation_....json",),
    chunk_count=1,
    relation_count=1,
    skipped_chunk_ids=(),
    failed_chunk_ids=(),
)
```

## Configuration

```python
ExtractRelationsConfig(
    relations_prefix="relations",
    chunk_keys_artifact="chunk_keys",
    entity_keys_artifact="entity_keys",
    max_attempts=3,
    temperature=0.0,
)
```

| 参数 | 说明 |
| --- | --- |
| `relations_prefix` | 关系 JSON 写入 ObjectStore 的前缀。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称。 |
| `entity_keys_artifact` | 输入 entity key artifact 名称。 |
| `max_attempts` | 单个 chunk 的最大抽取尝试次数。 |
| `temperature` | 关系抽取请求的模型温度。 |

如果模型输出无法解析、不满足协议、引用未知实体或生成自环关系，step 会带着错误原因重试。达到 `max_attempts` 后，该 chunk 会进入 `failed_chunk_ids`，不会中断整个构建流程。

如果某个 chunk 的实体数量少于 2，step 会跳过该 chunk，并记录到 `skipped_chunk_ids`。

## Input Choice

直接基于原始 chunk 建图：

```python
ExtractRelations(
    ExtractRelationsConfig(chunk_keys_artifact="chunk_keys")
)
```

基于 rechunk 后文本建图：

```python
ExtractRelations(
    ExtractRelationsConfig(chunk_keys_artifact="rechunked_chunk_keys")
)
```

Heta-style graph building 的关系抽取边界是当前 extraction unit：

```text
chunk text + entities from this chunk -> relations from this chunk
```

跨文档或全局关系推理不是 `ExtractRelations` 的职责。未来如果需要，可以作为单独 step 扩展。
