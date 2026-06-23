# Build Graph

`BuildGraph` 把已经抽取好的 `ExtractedEntity` 和 `ExtractedRelation`
写入 Heta-style PostgreSQL 图谱表和图谱向量索引。
它是当前图谱构建链路的落库 step，只负责写入实体、关系、证据映射和召回索引，
不做实体抽取、关系抽取、去重或历史图谱融合。

```text
ExtractedEntity JSON + ExtractedRelation JSON + ParsedChunk JSON
  -> entities / relations / graph_evidence
  -> graph_entities / graph_relations
```

这一步对齐 HetaDB 的 PostgreSQL 建图模式，但字段命名使用 Heta Framework 的新 schema。
旧 HetaDB 的 `node_id`、`node1`、`node2`、`semantics` 和 `cluster_chunk_relation`
在这里被替换为更明确的 `entity_id`、`source_entity_id`、`target_entity_id`、`relation_name` 和 `graph_evidence`。

## Contract

`BuildGraph` 使用 recipe components：

```text
stores.objects
stores.sql
stores.vector
models.embedding
```

默认读取：

```text
deduplicated_entity_keys
deduplicated_relation_keys
chunk_keys
```

默认输出：

```text
build_graph_result
```

完成后提供查询能力：

```text
heta_graph_search
```

默认输入是去重后的实体和关系。如果需要跳过去重，可以配置为读取 `entity_keys` 和 `relation_keys`。

## Tables

默认表名是：

```text
entities
relations
graph_evidence
```

生产中通常由应用层命名策略生成表名。`BuildGraph` 不理解任何业务命名上下文，
只接收最终表名：

```python
from heta_framework.kb.steps import BuildGraphConfig, GraphTableNames

def graph_tables(prefix: str) -> GraphTableNames:
    return GraphTableNames(
        entities=f"{prefix}_entities",
        relations=f"{prefix}_relations",
        evidence=f"{prefix}_graph_evidence",
    )

config = BuildGraphConfig(table_names=graph_tables("papers"))
```

原始未去重图谱也由外部命名策略决定：

```python
config = BuildGraphConfig(table_names=graph_tables("papers_raw"))
```

## Vector Collections

默认图谱向量 collection 是：

```text
graph_entities
graph_relations
```

它们对齐 HetaDB 的图检索方式：向量库负责召回候选 entity/relation id，
PostgreSQL 表负责根据 id 返回结构化实体、关系和证据 chunk。

collection 命名同样由外部策略生成后注入：

```python
from heta_framework.kb.steps import GraphVectorCollections

def graph_vectors(prefix: str) -> GraphVectorCollections:
    return GraphVectorCollections(
        entities=f"{prefix}_graph_entities",
        relations=f"{prefix}_graph_relations",
    )

config = BuildGraphConfig(
    table_names=graph_tables("papers"),
    vector_collections=graph_vectors("papers"),
)
```

实体向量文本由实体名称、类型、子类型、描述和属性组成。关系向量文本由起点实体、终点实体、
关系类型、关系名称、描述和属性组成。

## Entities

```text
entity_id
entity_name
entity_type
entity_subtype
description
attributes
created_at
updated_at
```

映射关系：

| 字段 | 来源 |
| --- | --- |
| `entity_id` | `ExtractedEntity.entity_id` |
| `entity_name` | `ExtractedEntity.name` |
| `entity_type` | `ExtractedEntity.type` |
| `entity_subtype` | `ExtractedEntity.subtype` |
| `description` | `ExtractedEntity.description` |
| `attributes` | `ExtractedEntity.attributes` JSON |

## Relations

```text
relation_id
source_entity_id
target_entity_id
source_entity_name
target_entity_name
relation_type
relation_name
description
attributes
created_at
updated_at
```

映射关系：

| 字段 | 来源 |
| --- | --- |
| `relation_id` | `ExtractedRelation.relation_id` |
| `source_entity_id` | `ExtractedRelation.source_entity_id` |
| `target_entity_id` | `ExtractedRelation.target_entity_id` |
| `source_entity_name` | `ExtractedRelation.source_entity_name` |
| `target_entity_name` | `ExtractedRelation.target_entity_name` |
| `relation_type` | `ExtractedRelation.type` |
| `relation_name` | `ExtractedRelation.name` |
| `description` | `ExtractedRelation.description` |
| `attributes` | `ExtractedRelation.attributes` JSON |

## Evidence

`graph_evidence` 保存图谱事实到 chunk 的证据映射：

```text
fact_id
fact_type
chunk_id
document_id
source_key
source_name
metadata
created_at
updated_at
```

映射关系：

| 字段 | 来源 |
| --- | --- |
| `fact_id` | `entity_id` 或 `relation_id` |
| `fact_type` | `"entity"` 或 `"relation"` |
| `chunk_id` | `source_chunk_ids` 中的每个 chunk id |
| `document_id` | `ParsedChunk.document_id` |
| `source_key` | `ParsedChunk.source.key` |
| `source_name` | `ParsedChunk.source.name` |
| `metadata` | 当前写入 `page_index` |

`BuildGraph` 会通过 `chunk_keys` 读取 `ParsedChunk`，建立 `chunk_id -> source` 映射。
如果某个 `source_chunk_id` 不在输入 chunk 集合中，实体或关系仍会写入 SQL，
但对应 evidence 行会跳过并记录 issue。

## Configuration

```python
BuildGraphConfig(
    table_names=GraphTableNames(
        entities="papers_entities",
        relations="papers_relations",
        evidence="papers_graph_evidence",
    ),
    vector_collections=GraphVectorCollections(
        entities="papers_graph_entities",
        relations="papers_graph_relations",
    ),
    entity_keys_artifact="deduplicated_entity_keys",
    relation_keys_artifact="deduplicated_relation_keys",
    chunk_keys_artifact="chunk_keys",
    vector_metric="cosine",
    batch_size=128,
    object_store=None,
    sql_store=None,
    vector_store=None,
    embedding_model=None,
)
```

| 参数 | 说明 |
| --- | --- |
| `table_names.entities` | 实体表名。 |
| `table_names.relations` | 关系表名。 |
| `table_names.evidence` | 证据映射表名。 |
| `vector_collections.entities` | 实体向量 collection 名称。 |
| `vector_collections.relations` | 关系向量 collection 名称。 |
| `entity_keys_artifact` | 输入 entity key artifact 名称。 |
| `relation_keys_artifact` | 输入 relation key artifact 名称。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称，用于构建 evidence source。 |
| `vector_metric` | 图谱向量 collection 的距离度量。 |
| `batch_size` | SQL 写入批大小。 |
| `object_store` | ObjectStore 组件名称。`None` 表示默认 `stores.objects`。 |
| `sql_store` | SQLStore 组件名称。`None` 表示默认 `stores.sql`。 |
| `vector_store` | VectorStore 组件名称。`None` 表示默认 `stores.vector`。 |
| `embedding_model` | Embedding model 组件名称。`None` 表示默认 `models.embedding`。 |

直接从原始抽取结果建图：

```python
BuildGraph(
    BuildGraphConfig(
        table_names=graph_tables("papers_raw"),
        entity_keys_artifact="entity_keys",
        relation_keys_artifact="relation_keys",
    )
)
```

## Result

```python
BuildGraphResult(
    entity_count=2,
    relation_count=1,
    evidence_count=3,
    entity_vector_count=2,
    relation_vector_count=1,
    vector_dimension=1536,
    skipped_evidence_count=0,
    issues=(),
)
```

| 字段 | 说明 |
| --- | --- |
| `entity_count` | 写入实体表的实体数量。 |
| `relation_count` | 写入关系表的关系数量。 |
| `evidence_count` | 写入 evidence 表的证据行数量。 |
| `entity_vector_count` | 写入实体向量 collection 的记录数量。 |
| `relation_vector_count` | 写入关系向量 collection 的记录数量。 |
| `vector_dimension` | 图谱向量维度。没有实体和关系时为 `0`。 |
| `skipped_evidence_count` | 因 chunk source 缺失而跳过的 evidence 行数量。 |
| `issues` | 可恢复问题列表。 |

## Issues

证据 chunk 缺失时，step 不写入该 evidence 行，而是在 result 中记录 issue：

```python
StepIssue(
    step="build_graph",
    subject=IssueSubject(type="entity", id="entity_shanghai"),
    code="missing_evidence_chunk",
    severity="warning",
    message="Evidence chunk was not found in the graph build input.",
    resolution=IssueResolution(
        action="skipped_evidence",
        outcome="The graph fact was written, but this evidence row was skipped.",
    ),
)
```

`BuildGraph` 只处理当前输入批次。和历史图谱库做全局实体/关系融合，应作为后续独立 step 实现。
