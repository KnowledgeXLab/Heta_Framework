# Deduplicate Entities

`DeduplicateEntities` 对当前构建批次中的 `ExtractedEntity` 做实体去重，并继续输出 `ExtractedEntity` JSON。它不写 GraphStore，也不和历史图谱库做全局合并。

```text
ExtractedEntity JSON -> ExtractedEntity JSON
```

## Contract

`DeduplicateEntities` 使用 recipe components：

```text
stores.objects
models.language
models.embedding  # semantic_merge=True 时需要
```

默认读取：

```text
entity_keys
```

默认输出：

```text
deduplicate_entities_result
deduplicated_entity_keys
entity_id_mapping
```

写入位置：

```text
deduplicated_entities/{entity_id}.json
```

## Behavior

去重分两层，对齐 HetaDB 当前批次内实体去重流程：

| 阶段 | 说明 |
| --- | --- |
| 精确合并 | 按标准化后的实体名称分组，组内交给 LLM 合并；支持多轮迭代、批量合并和 split 输出。 |
| 语义合并 | 对实体名称、类型、描述和属性做 embedding，相似实体交给 LLM 返回 `entity_list` 和 `mapping_table`，再按映射表合并。 |

`semantic_merge` 默认开启。关闭后只做精确合并，不需要 `models.embedding`。

精确合并阶段支持 LLM 返回一个实体或多个实体。多个实体表示 HetaDB 式 split：第一个匹配原实体名的结果作为主合并实体，其余作为拆分实体保留。

语义合并阶段使用 HetaDB 式映射表：

```json
{
  "entity_list": [
    {
      "NodeName": "上海市",
      "Type": "城市",
      "Subtype": "直辖市",
      "Description": "上海市是中华人民共和国直辖市。",
      "Attr": {},
      "merge_tag": true
    }
  ],
  "mapping_table": {
    "上海市": ["上海市", "Shanghai"]
  }
}
```

精确合并阶段的 LLM 可以只输出合并后的语义字段：

```json
{
  "entity": {
    "name": "上海市",
    "type": "城市",
    "subtype": "直辖市",
    "description": "上海市是中华人民共和国直辖市。",
    "attributes": {
      "所属国家": "中华人民共和国"
    }
  }
}
```

框架负责生成新的 `entity_id`、聚合 `source_chunk_ids`，并保存完整的 `ExtractedEntity`：

```json
{
  "entity_id": "entity_...",
  "chunk_id": "dedup_...",
  "document_id": "doc_...",
  "name": "上海市",
  "type": "城市",
  "subtype": "直辖市",
  "description": "上海市是中华人民共和国直辖市。",
  "attributes": {
    "所属国家": "中华人民共和国"
  },
  "source_chunk_ids": ["chunk_1", "chunk_2"]
}
```

`entity_id_mapping` 是旧实体 ID 到新实体 ID 的映射：

```python
{
    "entity_old_a": "entity_new",
    "entity_old_b": "entity_new",
}
```

后续关系去重或建图可以用这个映射把关系端点对齐到去重后的实体。

## Configuration

```python
DeduplicateEntitiesConfig(
    deduplicated_entities_prefix="deduplicated_entities",
    entity_keys_artifact="entity_keys",
    semantic_merge=True,
    similarity_threshold=0.9,
    max_rounds=10,
    llm_batch_size=20,
    semantic_batch_size=100,
    semantic_batch_count=4,
    max_attempts=3,
    temperature=0.0,
)
```

| 参数 | 说明 |
| --- | --- |
| `deduplicated_entities_prefix` | 去重实体 JSON 写入 ObjectStore 的前缀。 |
| `entity_keys_artifact` | 输入 entity key artifact 名称。 |
| `semantic_merge` | 是否启用 embedding 相似合并。 |
| `similarity_threshold` | 语义合并候选的余弦相似度阈值。 |
| `max_rounds` | 精确合并阶段最多迭代轮数。 |
| `llm_batch_size` | 同一实体名称下单次交给 LLM 合并的最大实体数。 |
| `semantic_batch_size` | 语义合并阶段的基础批大小。 |
| `semantic_batch_count` | 每轮语义合并聚合的批数量，用于模拟 HetaDB 的批内/批间归并。 |
| `max_attempts` | 单个合并组的最大 LLM 尝试次数。 |
| `temperature` | 去重请求的模型温度。 |

如果某个合并组多次无法得到合法结果，step 会保留该组原始实体，并在 `failed_group_count` 中记录失败数量。

`DeduplicateEntities` 只处理当前构建批次。和历史 Milvus / PostgreSQL 图谱库的全局融合属于后续 store merge step。

## Issues

非法 LLM 输出不会写入 `deduplicated_entities/`。step 会保留原始实体，并在 result 中记录 issue：

```python
StepIssue(
    step="deduplicate_entities",
    subject=IssueSubject(type="dedup_group", id="上海市"),
    code="deduplication_failed",
    severity="warning",
    message="LLM output is missing a non-empty Description field.",
    resolution=IssueResolution(
        action="kept_original_records",
        outcome="The group was not merged, and original records were kept.",
    ),
)
```

`issues` 属于运行诊断信息，不是主图谱产物。后续 recipe runner 可以统一收集这些 issue 生成 build report。
