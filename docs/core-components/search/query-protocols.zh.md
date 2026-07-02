# Query Protocols

Query 模块连接两件事：

```text
build steps 产出的可查询资产
query engines 需要消费的检索资产
```

用户最终通过 `KnowledgeBase.query(...)` 查询。框架内部用 `SearchAsset` 和 `QueryEngine` 让检索方式保持可扩展。

## How It Fits Together

```text
Step
  -> declares SearchAsset

QueryEngine
  -> declares required SearchAssetRef

KnowledgeBase
  -> checks current run assets
  -> enables matching query modes
```

这样，一个 query mode 只有在当前 KB 真正构建出所需资产后才会可用。

## SearchAsset

`SearchAsset` 描述一个 step 构建出的可查询资产：

```python
SearchAsset(
    kind="chunk_vector_index",
    name="chunks",
    store="stores.vector",
    metadata={"collection": "chunks"},
)
```

常见资产包括：

```text
chunk_vector_index
chunk_text_index
chunk_full_text_index
graph_vector_index
graph_tables
```

资产只描述位置和结构，不执行查询。

## SearchAssetRef

`SearchAssetRef` 是 QueryEngine 的资产依赖声明：

```python
SearchAssetRef(kind="chunk_vector_index")
SearchAssetRef(kind="graph_tables", name="default")
```

`name=None` 表示任意同类资产都可以满足依赖。指定 `name` 时，只有同名资产可以满足依赖。

## StepCapabilities

Step 通过 `StepCapabilities` 声明自己完成后提供什么：

```python
StepCapabilities(
    artifacts=frozenset({"index_vectors_result"}),
    queries=frozenset({"vector_search"}),
    search_assets=(
        SearchAsset(
            kind="chunk_vector_index",
            name="chunks",
            store="stores.vector",
            metadata={"collection": "chunks"},
        ),
    ),
)
```

| 字段 | 说明 |
| --- | --- |
| `artifacts` | build 过程中的中间产物。 |
| `queries` | 用户可见的查询能力。 |
| `search_assets` | query engine 实际依赖的存储资产。 |

## QueryEngine

第一版 QueryEngine 协议强制声明 `required_assets`。需要语言模型等组件的 engine 可以额外声明 `required_components`；这是可选属性，不破坏自定义 engine 的最小实现。

```python
class QueryEngineProtocol(Protocol):
    @property
    def mode(self) -> str:
        ...

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        ...

    # Optional:
    # required_components: frozenset[ComponentRef]

    async def query(
        self,
        request: QueryRequest,
        context: QueryContext,
    ) -> QueryResponse:
        ...
```

`required_components` 用于声明 `models.language` 这类运行时组件依赖。例如 `heta_rewrite_search` 和 `heta_multihop_search` 都需要语言模型；`heta_rerank_search` 的 reranker 是可选增强，因此不把 `models.reranker` 作为硬依赖。

## QueryRequest And QueryResponse

`QueryRequest` 接收用户输入：

```text
text
mode
top_k
filters
options
trace
```

`options` 用于承载不同 query mode 的专有参数，例如 rerank、RRF、graph depth、max hops。

`QueryResponse` 支持：

```text
results
answer
citations
trace
metadata
```

所以它可以承载：

```text
普通向量检索
关键词检索
图谱检索
混合 rerank 检索
multi-hop answer
citations / provenance
```

## Registry

`QueryEngineRegistry` 管理 query engines：

```python
registry = QueryEngineRegistry([VectorSearchEngine()])
available = registry.available_modes(kb_assets)
```

`KnowledgeBase.available_queries` 会同时检查资产和组件：

```text
engine.required_assets
  vs
KB run_record.capabilities.search_assets

engine.required_components
  vs
recipe.models / recipe.stores / recipe.parsers
```

`KnowledgeBase.available_queries` 和 `KnowledgeBase.query()` 都基于这个 registry 工作。
