# Deduplicate Relations

`DeduplicateRelations` 对当前构建批次中的 `ExtractedRelation` 做关系去重，并继续输出 `ExtractedRelation` JSON。

它只处理当前 batch，不写图谱库，也不和历史图谱库做全局合并。

```text
ExtractedRelation JSON -> ExtractedRelation JSON
```

## Contract

`DeduplicateRelations` 使用 recipe components：

```text
stores.objects
models.language
models.embedding  # semantic_merge=True 时需要
```

默认读取：

```text
relation_keys
entity_id_mapping
```

默认输出：

```text
deduplicate_relations_result
deduplicated_relation_keys
relation_id_mapping
```

写入位置：

```text
deduplicated_relations/{relation_id}.json
```

## Behavior

执行去重前，step 会先应用 `entity_id_mapping`，把关系端点对齐到去重后的实体 ID。这样实体去重后，关系去重仍然可以发现指向同一对实体的重复关系。

去重分两层，对齐 HetaDB 当前批次内关系去重流程：

| 阶段 | 说明 |
| --- | --- |
| 精确合并 | 按起点实体、终点实体、关系名称和关系类型分组，组内交给 LLM 合并；支持多轮迭代、批量合并和 split 输出。 |
| 语义合并 | 对关系端点、类型、名称、描述和属性做 embedding，相似关系交给 LLM 返回 `relation_list` 和 `mapping_table`，再按映射表合并。 |

`semantic_merge` 默认开启。关闭后只做精确合并，不需要 `models.embedding`。

语义合并阶段使用 HetaDB 式映射表：

```json
{
  "relation_list": [
    {
      "Node1": "上海市",
      "Node2": "徐汇区",
      "Relation": "包含行政区",
      "Type": "空间关系",
      "Description": "徐汇区是上海市下辖行政区。",
      "Attr": {},
      "merge_tag": true
    }
  ],
  "mapping_table": {
    "上海市||徐汇区": ["上海市||徐汇区"]
  }
}
```

精确合并阶段的 LLM 可以只输出合并后的关系语义字段：

```json
{
  "relation": {
    "type": "空间关系",
    "name": "包含行政区",
    "description": "徐汇区是上海市下辖行政区。",
    "attributes": {}
  }
}
```

框架负责保留关系端点、生成新的 `relation_id`、聚合 `source_chunk_ids`，并保存完整的 `ExtractedRelation`：

```json
{
  "relation_id": "relation_...",
  "chunk_id": "dedup_...",
  "document_id": "doc_...",
  "source_entity_id": "entity_shanghai",
  "target_entity_id": "entity_xuhui",
  "source_entity_name": "上海市",
  "target_entity_name": "徐汇区",
  "type": "空间关系",
  "name": "包含行政区",
  "description": "徐汇区是上海市下辖行政区。",
  "attributes": {},
  "source_chunk_ids": ["chunk_1", "chunk_2"]
}
```

`relation_id_mapping` 是旧关系 ID 到新关系 ID 的映射：

```python
{
    "relation_old_a": "relation_new",
    "relation_old_b": "relation_new",
}
```

## Configuration

```python
DeduplicateRelationsConfig(
    deduplicated_relations_prefix="deduplicated_relations",
    relation_keys_artifact="relation_keys",
    entity_id_mapping_artifact="entity_id_mapping",
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
| `deduplicated_relations_prefix` | 去重关系 JSON 写入 ObjectStore 的前缀。 |
| `relation_keys_artifact` | 输入 relation key artifact 名称。 |
| `entity_id_mapping_artifact` | 输入实体 ID 映射 artifact 名称。设为 `None` 时不做端点重映射。 |
| `semantic_merge` | 是否启用 embedding 相似合并。 |
| `similarity_threshold` | 语义合并候选的余弦相似度阈值。 |
| `max_rounds` | 精确合并阶段最多迭代轮数。 |
| `llm_batch_size` | 同一关系 key 下单次交给 LLM 合并的最大关系数。 |
| `semantic_batch_size` | 语义合并阶段的基础批大小。 |
| `semantic_batch_count` | 每轮语义合并聚合的批数量，用于模拟 HetaDB 的批内/批间归并。 |
| `max_attempts` | 单个合并组的最大 LLM 尝试次数。 |
| `temperature` | 去重请求的模型温度。 |

如果某个合并组多次无法得到合法结果，step 会保留该组原始关系，并在 `failed_group_count` 中记录失败数量。

`DeduplicateRelations` 只处理当前构建批次。和历史 Milvus / PostgreSQL 图谱库的全局融合属于后续 store merge step。

## Issues

非法 LLM 输出不会写入 `deduplicated_relations/`。step 会保留原始关系，并在 result 中记录 issue：

```python
StepIssue(
    step="deduplicate_relations",
    subject=IssueSubject(
        type="dedup_group",
        id="上海市|徐汇区|包含行政区|空间关系",
    ),
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
