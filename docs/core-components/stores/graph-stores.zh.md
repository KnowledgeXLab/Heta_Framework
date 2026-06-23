# Graph Stores

Graph Stores 提供 Heta 与属性图存储系统交互的统一入口。它负责保存实体节点、关系边和属性，用于未来 Neo4j、NebulaGraph、JanusGraph 等属性图模块。

当前实现包含：

- `GraphStoreProtocol`：属性图存储能力协议。
- `InMemoryGraphStore`：内存实现，用于测试、示例和本地小规模 pipeline。
- `GraphNode` / `GraphEdge`：图节点和图边的写入对象。

Neo4j、NebulaGraph、JanusGraph 等真实图数据库后续都可以作为 adapter 实现 `GraphStoreProtocol`。
当前 Heta-style `BuildGraph` 使用 PostgreSQL schema，不依赖 `GraphStoreProtocol`。

## 快速开始

```python
from heta_framework.common.stores import GraphEdge, GraphNode, InMemoryGraphStore

store = InMemoryGraphStore()

await store.upsert_nodes(
    [
        GraphNode(
            id="entity_shanghai",
            labels=("Entity", "客观实体", "行政区划"),
            properties={
                "name": "上海市",
                "description": "上海市是中华人民共和国直辖市。",
            },
        ),
        GraphNode(
            id="entity_xuhui",
            labels=("Entity", "客观实体", "行政区划"),
            properties={"name": "徐汇区"},
        ),
    ]
)

await store.upsert_edges(
    [
        GraphEdge(
            id="relation_contains",
            source_id="entity_shanghai",
            target_id="entity_xuhui",
            type="包含行政区",
            properties={
                "type": "空间关系",
                "description": "徐汇区是上海市下辖行政区。",
            },
        )
    ]
)
```

## 核心对象

| 对象 | 说明 |
| --- | --- |
| `GraphStoreProtocol` | 属性图存储能力协议，用于 Recipe、构建步骤和自定义 store 的类型约束。 |
| `InMemoryGraphStore` | 内存图存储实现，不持久化，适合测试和 demo。 |
| `GraphNode` | 要写入的节点记录，包含稳定 id、labels 和 properties。 |
| `GraphEdge` | 要写入的有向边记录，包含稳定 id、起点、终点、关系类型和 properties。 |

## 协议

```python
class GraphStoreProtocol:
    async def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None: ...
    async def upsert_edges(self, edges: Sequence[GraphEdge]) -> None: ...
    async def delete_nodes(self, node_ids: Sequence[str]) -> None: ...
    async def delete_edges(self, edge_ids: Sequence[str]) -> None: ...
    async def get_node(self, node_id: str) -> GraphNode | None: ...
    async def get_edge(self, edge_id: str) -> GraphEdge | None: ...
    async def count_nodes(self) -> int: ...
    async def count_edges(self) -> int: ...
    async def aclose(self) -> None: ...
```

`GraphStoreProtocol` 是结构化协议，不要求用户继承某个父类。自定义图数据库只要实现这些方法，就可以被后续属性图步骤接收。

## 节点

```python
GraphNode(
    id="entity_...",
    labels=("Entity", "客观实体", "行政区划"),
    properties={
        "name": "上海市",
        "type": "客观实体",
        "subtype": "行政区划",
        "description": "...",
        "source_chunk_ids": ["chunk_1", "chunk_2"],
    },
)
```

`id` 是 upsert 和 delete 的稳定主键。Heta 默认使用 `ExtractedEntity.entity_id` 作为节点 id；
实体名称保存在 `properties["name"]` 中。这样同名实体经过去重后可以稳定映射到同一个节点，
也不会把显示名称和数据库主键耦合在一起。

## 边

```python
GraphEdge(
    id="relation_...",
    source_id="entity_shanghai",
    target_id="entity_xuhui",
    type="包含行政区",
    properties={
        "name": "包含行政区",
        "type": "空间关系",
        "description": "...",
        "source_chunk_ids": ["chunk_1"],
    },
)
```

`type` 表示具体图边类型，对应 Heta 关系协议里的 `name`，也就是旧版 HetaDB 的 `Relation` 字段。
关系一级类型保存在 `properties["type"]` 中，对应旧版 HetaDB 的 `Type` 字段。

具体图数据库如果对边类型有命名限制，应在 adapter 内部处理名称转义或映射，不应改变 Heta Framework 的关系语义。

## 能力范围

Graph Stores 层负责：

- 节点写入和更新
- 边写入和更新
- 按 id 删除节点和边
- 按 id 读取节点和边
- 节点和边计数

Graph Stores 不负责实体抽取、关系抽取、去重、向量召回、SQL 持久化、历史图谱融合或 KnowledgeBase 生命周期管理。
