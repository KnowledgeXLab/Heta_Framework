# Vector Stores

Vector Stores 提供 Heta 与向量存储系统交互的统一入口。它负责保存向量记录、执行相似度检索、按 metadata 过滤结果，并为后续 KnowledgeRecipe 的向量索引步骤提供稳定接口。

当前实现包含：

- `VectorStoreProtocol`：向量存储能力协议。
- `InMemoryVectorStore`：内存实现，用于测试、示例和本地小规模 pipeline。
- `MilvusVectorStore`：Milvus adapter，通过 `pymilvus` 连接真实 Milvus 服务。
- `VectorRecord` / `VectorQuery` / `VectorSearchResult`：向量写入、查询和返回对象。

Qdrant、pgvector 等真实数据库后续也会作为 adapter 实现 `VectorStoreProtocol`。Recipe 和构建步骤只依赖协议，不依赖某个具体数据库。

## 快速开始

```python
from heta_framework.common.stores import (
    MilvusVectorStore,
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
)

store = MilvusVectorStore(
    uri="http://localhost:19530",
    token="root:Milvus",
    timeout=10,
)

await store.create_collection(
    VectorCollectionConfig(
        name="chunks",
        dimension=3,
        metric="cosine",
    )
)

await store.upsert(
    "chunks",
    [
        VectorRecord(
            id="chunk-001",
            vector=[0.1, 0.2, 0.3],
            text="Heta 是知识库构建框架。",
            metadata={"document_id": "doc-001", "kind": "paper"},
        )
    ],
)

results = await store.search(
    "chunks",
    VectorQuery(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        filter={"kind": "paper"},
    ),
)
```

Milvus adapter 是可选依赖，使用前需要安装：

```bash
pip install "heta[milvus]"
```

## 核心对象

| 对象 | 说明 |
| --- | --- |
| `VectorStoreProtocol` | 向量存储能力协议，用于 Recipe、构建步骤和自定义 store 的类型约束。 |
| `InMemoryVectorStore` | 内存向量存储实现，不持久化，适合测试和 demo。 |
| `MilvusVectorStore` | Milvus adapter，适合生产向量检索。 |
| `VectorCollectionConfig` | collection 配置，包含名称、维度、距离度量和 metadata schema。 |
| `VectorRecord` | 要写入的向量记录，包含 id、vector、text 和 metadata。 |
| `VectorQuery` | 一次向量查询，包含 query vector、top_k 和 metadata filter。 |
| `VectorSearchResult` | 一条检索结果，包含 id、score、text 和 metadata。 |

## 协议

```python
class VectorStoreProtocol:
    async def create_collection(self, config: VectorCollectionConfig) -> None: ...
    async def drop_collection(self, name: str) -> None: ...
    async def has_collection(self, name: str) -> bool: ...
    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None: ...
    async def search(self, collection: str, query: VectorQuery) -> list[VectorSearchResult]: ...
    async def delete(self, collection: str, ids: Sequence[str]) -> None: ...
    async def count(self, collection: str) -> int: ...
    async def aclose(self) -> None: ...
```

`VectorStoreProtocol` 是结构化协议，不要求用户继承某个父类。自定义向量库只要实现这些方法，就可以被后续 Recipe 或构建步骤接收。

## Milvus

```python
from heta_framework.common.stores import MilvusVectorStore

store = MilvusVectorStore(
    uri="http://10.6.8.115:19531",
    token=None,
    db_name=None,
    timeout=10,
)
```

`MilvusVectorStore` 使用固定字段名：

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，对应 `VectorRecord.id`。 |
| `vector` | FLOAT_VECTOR，对应 `VectorRecord.vector`。 |
| `text` | VARCHAR，对应 `VectorRecord.text`。 |

`VectorRecord.metadata` 会通过 Milvus dynamic fields 写入，可以用于简单 metadata 过滤：

```python
VectorQuery(
    vector=[...],
    top_k=10,
    filter={"document_id": "doc-001", "kind": "paper"},
)
```

当前 filter 会转换为 Milvus 表达式：

```text
document_id == "doc-001" and kind == "paper"
```

## Collection

```python
VectorCollectionConfig(
    name="chunks",
    dimension=1536,
    metric="cosine",
    metadata_schema=None,
)
```

| 字段 | 说明 |
| --- | --- |
| `name` | collection 名称。 |
| `dimension` | 向量维度。写入和查询时会校验维度一致。 |
| `metric` | 距离度量，支持 `cosine`、`dot`、`l2`。 |
| `metadata_schema` | metadata schema，可选。当前由具体 adapter 决定是否使用。 |

## 记录

```python
VectorRecord(
    id="chunk-001",
    vector=[...],
    text="chunk text",
    metadata={"document_id": "doc-001", "page": 3},
)
```

`id` 是 upsert 和 delete 的稳定主键。`text` 和 `metadata` 用于检索结果返回、过滤和后续追踪。

## 查询

```python
VectorQuery(
    vector=[...],
    top_k=10,
    filter={"document_id": "doc-001"},
)
```

当前 `filter` 表示 metadata 等值过滤。更复杂的数据库表达式不会进入协议层；Milvus、Qdrant 等 adapter 后续可以通过 adapter 配置或扩展选项处理。

## 能力范围

Vector Stores 层负责：

- collection 生命周期
- 向量记录写入和更新
- 向量相似度检索
- metadata 等值过滤
- 按 id 删除记录
- collection 记录计数

Vector Stores 不负责 embedding 计算、chunk 切分、rerank、hybrid search、权限控制或 KnowledgeBase 生命周期管理。
