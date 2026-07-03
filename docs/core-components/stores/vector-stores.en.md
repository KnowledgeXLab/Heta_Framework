# Vector Stores

Vector Stores are Heta's unified interface to vector storage systems. They handle collection lifecycle, vector upsert, similarity search, and metadata filtering.

Recipes and steps depend on `VectorStoreProtocol`, not a specific database. Heta currently provides an in-memory store and a Milvus adapter; Qdrant, pgvector, and other stores can implement the same protocol later.

## Quick Start

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
            text="Heta is a knowledge-base framework.",
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

Milvus is optional:

```bash
pip install "heta-framework[milvus]"
```

## Implementations

| Store | Use |
| --- | --- |
| `InMemoryVectorStore` | In-memory implementation for tests, examples, and small local pipelines. |
| `MilvusVectorStore` | Milvus adapter backed by `pymilvus`. |

## Core Objects

| Object | Meaning |
| --- | --- |
| `VectorStoreProtocol` | Vector-store capability protocol for recipes, steps, query engines, and custom stores. |
| `VectorCollectionConfig` | Collection config with name, dimension, metric, and optional metadata schema. |
| `VectorRecord` | Record to write, containing id, vector, text, and metadata. |
| `VectorQuery` | Query vector, top_k, and metadata filter. |
| `VectorSearchResult` | Search result with id, score, text, and metadata. |

## Protocol

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

`VectorStoreProtocol` is structural. Custom vector stores only need to implement these methods.

## Collection

```python
VectorCollectionConfig(
    name="chunks",
    dimension=1536,
    metric="cosine",
    metadata_schema=None,
)
```

Supported metrics are `cosine`, `dot`, and `l2`. Dimension is validated on write and query.

## Records And Query

```python
VectorRecord(
    id="chunk-001",
    vector=[...],
    text="chunk text",
    metadata={"document_id": "doc-001", "page": 3},
)

VectorQuery(
    vector=[...],
    top_k=10,
    filter={"document_id": "doc-001"},
)
```

`id` is the stable upsert/delete key. `text` and `metadata` are returned in search results and used for filtering and provenance.

The base protocol supports metadata equality filters. More advanced database expressions should stay in adapter-specific configuration or future extensions.

## Milvus Adapter

```python
from heta_framework.common.stores import MilvusVectorStore

store = MilvusVectorStore(
    uri="http://10.6.8.115:19531",
    token=None,
    db_name=None,
    timeout=10,
)
```

Milvus uses fixed fields:

| Field | Meaning |
| --- | --- |
| `id` | Primary key from `VectorRecord.id`. |
| `vector` | `FLOAT_VECTOR` from `VectorRecord.vector`. |
| `text` | `VARCHAR` from `VectorRecord.text`. |

`VectorRecord.metadata` is written through Milvus dynamic fields and can be used for simple metadata filtering.

## Scope

Vector Stores handle collections, vector upserts, similarity search, metadata equality filters, deletes by id, and collection counts.

They do not compute embeddings, split chunks, rerank, run hybrid search, manage authorization, or manage the `KnowledgeBase` lifecycle.
