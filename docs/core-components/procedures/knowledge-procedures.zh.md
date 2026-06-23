# Knowledge Procedures

`Procedure` 是一段可复用的 step composition。
它不执行任务，不读取 context，不访问 store，只负责把一段标准做法展开成真实 steps。

```text
Recipe
  -> Procedure
      -> Step
```

`Step` 仍然是真正的执行单元。`Procedure` 只做静态接线。

## Contract

```python
@runtime_checkable
class KnowledgeProcedureProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        ...
```

协议里不重复声明 `requirements` 和 `capabilities`。
单一真相仍然来自展开后的 steps：

```python
expanded_steps = procedure.steps()
```

Recipe runner 应该基于这些真实 steps 做组件校验、artifact 校验和执行调度。

## HetaGraphProcedure

`HetaGraphProcedure` 打包 `IndexVectors` 之后的 Heta-style graph build 流程。
它覆盖的是 HetaDB-style 建图链路，而不是基础向量检索链路。

基础向量检索到 `IndexVectors` 已经完成：

```text
ParseDocuments
SplitDocuments
EmbedChunks
IndexVectors
```

`IndexVectors` 之后，如果需要构建 Heta graph，可以进入 `HetaGraphProcedure`：

```text
MergeChunks
RechunkDocuments
PersistChunks
ExtractEntities
ExtractRelations
DeduplicateEntities
DeduplicateRelations
BuildGraph / MergeGraphIntoStore
```

其中 `MergeChunks`、`RechunkDocuments`、`PersistChunks` 是可选准备步骤。
它们主要服务于后续图谱抽取、证据查询和溯源，不会重新插入 Milvus 形成另一套 chunk 向量检索库。

一次性落图：

```python
from heta_framework.kb.procedures import HetaGraphProcedure

steps = [
    ParseDocuments(...),
    SplitDocuments(...),
    EmbedChunks(...),
    IndexVectors(...),
    *HetaGraphProcedure.build().steps(),
]
```

展开为：

```text
ExtractEntities
ExtractRelations
DeduplicateEntities
DeduplicateRelations
BuildGraph
```

如果需要 HetaDB-style chunk merge / rechunk / SQL chunk 持久化，可以在 `HetaGraphProcedure`
之前显式插入这些 steps：

```python
steps = [
    ParseDocuments(...),
    SplitDocuments(...),
    EmbedChunks(...),
    IndexVectors(...),
    MergeChunks(...),
    RechunkDocuments(...),
    PersistChunks(...),
    *HetaGraphProcedure.build().steps(),
]
```

动态合并入已有图谱库：

```python
steps = [
    ParseDocuments(...),
    SplitDocuments(...),
    EmbedChunks(...),
    IndexVectors(...),
    *HetaGraphProcedure.merge_into_store().steps(),
]
```

展开为：

```text
ExtractEntities
ExtractRelations
DeduplicateEntities
DeduplicateRelations
MergeGraphIntoStore
```

同样，动态合并也可以在进入图谱抽取前加入 chunk 准备 steps：

```python
steps = [
    ParseDocuments(...),
    SplitDocuments(...),
    EmbedChunks(...),
    IndexVectors(...),
    MergeChunks(...),
    RechunkDocuments(...),
    PersistChunks(...),
    *HetaGraphProcedure.merge_into_store().steps(),
]
```

## Static Wiring

Procedure 可以统一配置 artifact 名称：

```python
procedure = HetaGraphProcedure.build(
    chunk_keys_artifact="chunk_keys",
    entity_keys_artifact="entity_keys",
    relation_keys_artifact="relation_keys",
    deduplicated_entity_keys_artifact="deduplicated_entity_keys",
    deduplicated_relation_keys_artifact="deduplicated_relation_keys",
)
```

这些名称会被写入展开后的 step config。
Procedure 本身不会读取这些 artifacts。

## Skip Deduplication

如果不需要 batch 内图谱去重：

```python
steps = [
    ...,
    *HetaGraphProcedure.build(deduplicate=False).steps(),
]
```

展开为：

```text
ExtractEntities
ExtractRelations
BuildGraph
```

此时 `BuildGraph` 会读取：

```text
entity_keys
relation_keys
```

而不是：

```text
deduplicated_entity_keys
deduplicated_relation_keys
```

## Storage Names

表名和向量 collection 名仍然由外部注入：

```python
from heta_framework.kb.steps import GraphTableNames, GraphVectorCollections

procedure = HetaGraphProcedure.merge_into_store(
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

`Procedure` 不引入 `dataset`，也不创建命名策略。

## Boundaries

Procedure 负责：

```text
静态展开 steps
配置 step 之间的 artifact 接线
选择 build / merge_into_store 这类流程分支
```

Procedure 不负责：

```text
执行 steps
读取 ObjectStore
访问 SQL / VectorStore
校验组件是否存在
管理 artifact 生命周期
```
