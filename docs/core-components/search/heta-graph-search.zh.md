# Heta Graph Search

`heta_graph_search` 检索 `BuildGraph` 或 `MergeGraphIntoStore` 写入的 Heta-style graph store。

它对齐 HetaDB 的图检索语义：先从图向量库召回 entity / relation，再回 SQL 图表补全结构化事实和 evidence。

## Required Assets

`BuildGraph` 和 `MergeGraphIntoStore` 会声明两个资产：

```python
SearchAsset(
    kind="graph_tables",
    name="entities",
    store="stores.sql",
    metadata={
        "entities_table": "entities",
        "relations_table": "relations",
        "evidence_table": "graph_evidence",
    },
)

SearchAsset(
    kind="graph_vector_index",
    name="graph_entities",
    store="stores.vector",
    metadata={
        "entity_collection": "graph_entities",
        "relation_collection": "graph_relations",
    },
)
```

只要 KB 的 latest run record 中存在这些资产，默认 query registry 就会启用：

```text
heta_graph_search
```

## Retrieval Flow

```text
query text
  -> models.embedding.embed()
  -> search graph entity vectors
  -> search graph relation vectors
  -> hydrate facts from SQL graph tables
  -> attach evidence from graph_evidence
  -> QueryResponse
```

额外补全逻辑：

```text
entity hit
  -> add matched entity
  -> add one-hop relations where source_entity_name or target_entity_name matches

relation hit
  -> add matched relation
  -> add source / target endpoint entities
```

这让图检索结果不只是孤立向量命中，而是带有局部图上下文。

## Usage

```python
response = await kb.query(
    "上海市和徐汇区是什么关系？",
    mode="heta_graph_search",
    top_k=8,
    options={"evidence_top_k": 3},
)
```

每条 `QueryResult` 表示一个图事实：

```text
kind = "entity" | "relation"
id
text
score
source
metadata
```

`metadata["matched_by"]` 表示事实来源：

```text
entity_vector
entity_one_hop
relation_vector
relation_endpoint
```

`metadata["evidence"]` 会包含相关 chunk 来源。

## Boundary

`heta_graph_search` 只负责图事实召回和局部图上下文补全。
它不做答案生成、rerank、BM25 融合或多跳 ReAct 推理。
这些能力应由更高层的 hybrid / multi-hop query engine 组合实现。
