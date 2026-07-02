# Merge Graph Into Store

`MergeGraphIntoStore` 把当前 batch 的实体和关系增量合并进已有图谱库。

它对齐 HetaDB 的动态建库后半段：先用向量召回历史候选，再用 LLM 判断是否合并，最后同步更新 SQL 表、图谱向量索引和 evidence。

```text
deduplicated entities / relations
  -> search historical graph vectors
  -> LLM merge decision
  -> delete old graph facts
  -> insert merged graph facts
  -> update evidence
```

## When To Use

`BuildGraph` 和 `MergeGraphIntoStore` 通常二选一。

`BuildGraph` 是轻量写入 step：

```text
current graph facts -> SQL + vector + evidence
```

它不查历史图谱，不调用 LLM 做历史合并，也不删除旧记录。

`MergeGraphIntoStore` 是动态图谱合并 step：

```text
current graph facts + historical graph store -> updated graph store
```

它适合持续导入、增量更新和需要减少历史重复实体/关系的知识库。如果目标图谱库为空，它会自然退化为首次写入。

## Contract

`MergeGraphIntoStore` 使用 recipe components：

```text
stores.objects
stores.sql
stores.vector
models.embedding
models.language
```

默认读取：

```text
deduplicated_entity_keys
deduplicated_relation_keys
chunk_keys
```

默认输出：

```text
merge_graph_into_store_result
```

完成后提供查询能力：

```text
heta_graph_search
```

默认输入是 batch 内去重后的实体和关系。也可以配置为读取 `entity_keys` 和 `relation_keys`，但不建议作为默认建图路径。

## Storage Names

这个 step 不引入 `dataset` 概念。表名和 collection 名由外部命名策略生成后注入：

```python
from heta_framework.kb.steps import (
    GraphTableNames,
    GraphVectorCollections,
    MergeGraphIntoStoreConfig,
)

config = MergeGraphIntoStoreConfig(
    table_names=GraphTableNames(
        entities="papers_entities",
        relations="papers_relations",
        evidence="papers_graph_evidence",
    ),
    vector_collections=GraphVectorCollections(
        entities="papers_graph_entities",
        relations="papers_graph_relations",
    ),
)
```

默认表名是：

```text
entities
relations
graph_evidence
```

默认向量 collection 是：

```text
graph_entities
graph_relations
```

## Entity Merge

实体合并流程：

```text
ExtractedEntity
  -> embedding(name, type, subtype, description, attributes)
  -> VectorStore.search(graph_entities, top_k)
  -> load candidate rows from SQL
  -> LLM returns entity_list + mapping_table
  -> group overlapping candidates
  -> LLM merges each group
  -> delete old entities
  -> insert merged entities
  -> update entity vectors
  -> retarget evidence
```

LLM 输出遵循 HetaDB-style 结构：

```json
{
  "entity_list": [
    {
      "NodeName": "上海市",
      "Type": "城市",
      "Subtype": "直辖市",
      "Description": "上海市是中国直辖市和重要城市。",
      "Attr": {},
      "merge_tag": true
    }
  ],
  "mapping_table": {
    "上海市": ["上海市", "Shanghai"]
  }
}
```

`mapping_table` 为空时表示不合并，当前实体会作为新实体写入。

## Relation Merge

关系在实体之后处理。原因是实体 merge 可能改变 relation 的端点：

```text
merge entities
  -> entity_id / entity_name mapping
  -> normalize relation endpoints
  -> merge relations
```

关系合并流程：

```text
ExtractedRelation
  -> embedding(source, target, type, name, description, attributes)
  -> VectorStore.search(graph_relations, top_k)
  -> load candidate rows from SQL
  -> LLM returns relation_list + mapping_table
  -> group overlapping candidates
  -> LLM merges each group
  -> delete old relations
  -> insert merged relations
  -> update relation vectors
  -> retarget evidence
```

LLM 输出结构：

```json
{
  "relation_list": [
    {
      "Node1": "上海市",
      "Node2": "徐汇区",
      "Relation": "包含行政区",
      "Type": "空间关系",
      "Description": "徐汇区是上海市下辖区域。",
      "Attr": {},
      "merge_tag": true
    }
  ],
  "mapping_table": {
    "上海市||徐汇区": ["relation_old", "relation_new"]
  }
}
```

`mapping_table` 为空时表示不合并，当前关系会作为新关系写入。

## Evidence

合并后的 evidence 由两部分组成：

```text
current batch evidence
historical evidence retargeted to merged fact id
```

例如旧实体和新实体合并后，旧实体指向的 chunk 证据不会丢失。
旧 evidence 会被改写到合并后的 `entity_id` 或 `relation_id`，再和当前 batch evidence 去重后写回 `graph_evidence`。

## Result

```python
MergeGraphIntoStoreResult(
    input_entity_count=1,
    input_relation_count=1,
    inserted_entity_count=0,
    inserted_relation_count=0,
    merged_entity_count=1,
    merged_relation_count=1,
    deleted_entity_count=1,
    deleted_relation_count=1,
    evidence_count=4,
    issues=(),
)
```

`issues` 只记录非致命问题，例如 LLM 返回无效 JSON 或实体合并后关系端点变成同一个实体。这类问题不会直接中断 pipeline；step 会保留当前可写入的结果并继续执行。
