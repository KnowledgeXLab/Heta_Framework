import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (
    MilvusVectorStore,
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
    VectorStoreProtocol,
)


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self):
        self.collections = set()
        self.collection_descriptions = {}
        self.collection_indexes = {}
        self.created = []
        self.upserted = []
        self.deleted = []
        self.search_calls = []
        self.flushed = []
        self.closed = False

    @staticmethod
    def create_schema(**kwargs):
        schema = FakeSchema()
        schema.kwargs = kwargs
        return schema

    @staticmethod
    def prepare_index_params():
        return FakeIndexParams()

    def has_collection(self, *, collection_name):
        return collection_name in self.collections

    def create_collection(self, **kwargs):
        collection_name = kwargs["collection_name"]
        self.collections.add(collection_name)
        self.created.append(kwargs)
        vector_field = kwargs["schema"].fields[1]
        self.collection_descriptions[collection_name] = {
            "fields": [
                {"name": field["field_name"], "params": field}
                for field in kwargs["schema"].fields
            ]
        }
        self.collection_indexes[collection_name] = {
            vector_field["field_name"]: {
                "field_name": vector_field["field_name"],
                "metric_type": kwargs["index_params"].indexes[0]["metric_type"],
            }
        }

    def drop_collection(self, *, collection_name):
        self.collections.discard(collection_name)

    def upsert(self, *, collection_name, data):
        self.upserted.append({"collection_name": collection_name, "data": data})

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [
            [
                {
                    "id": "chunk-001",
                    "distance": 0.9,
                    "entity": {
                        "text": "hello",
                        "document_id": "doc-1",
                        "kind": "paper",
                    },
                }
            ]
        ]

    def delete(self, *, collection_name, ids):
        self.deleted.append({"collection_name": collection_name, "ids": ids})

    def flush(self, *, collection_name):
        self.flushed.append(collection_name)

    def query(self, **kwargs):
        return [{"count(*)": 3}]

    def describe_collection(self, *, collection_name):
        return self.collection_descriptions[collection_name]

    def list_indexes(self, *, collection_name):
        return list(self.collection_indexes.get(collection_name, {}))

    def describe_index(self, *, collection_name, index_name):
        return self.collection_indexes[collection_name][index_name]

    def close(self):
        self.closed = True


def install_fake_pymilvus(monkeypatch):
    fake = SimpleNamespace(
        DataType=SimpleNamespace(VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR"),
        MilvusClient=lambda **kwargs: FakeMilvusClient(),
    )
    monkeypatch.setitem(sys.modules, "pymilvus", fake)


def test_milvus_vector_store_satisfies_protocol(monkeypatch):
    install_fake_pymilvus(monkeypatch)

    assert isinstance(MilvusVectorStore(), VectorStoreProtocol)


def test_milvus_vector_store_creates_collection_and_upserts(monkeypatch):
    install_fake_pymilvus(monkeypatch)

    async def run():
        store = MilvusVectorStore()
        await store.create_collection(
            VectorCollectionConfig(name="chunks", dimension=3, metric="cosine")
        )
        await store.upsert(
            "chunks",
            [
                VectorRecord(
                    id="chunk-001",
                    vector=[0.1, 0.2, 0.3],
                    text="hello",
                    metadata={"document_id": "doc-1"},
                )
            ],
        )
        return store

    store = asyncio.run(run())
    client = store._client

    assert client.created[0]["collection_name"] == "chunks"
    assert client.created[0]["index_params"].indexes == [
        {"field_name": "vector", "index_type": "AUTOINDEX", "metric_type": "COSINE"}
    ]
    assert client.upserted[0]["data"] == [
        {
            "id": "chunk-001",
            "vector": [0.1, 0.2, 0.3],
            "text": "hello",
            "document_id": "doc-1",
        }
    ]
    assert client.flushed == ["chunks"]


def test_milvus_vector_store_searches_with_filter(monkeypatch):
    install_fake_pymilvus(monkeypatch)

    async def run():
        store = MilvusVectorStore()
        await store.create_collection(VectorCollectionConfig(name="chunks", dimension=3))
        results = await store.search(
            "chunks",
            VectorQuery(
                vector=[0.1, 0.2, 0.3],
                filter={"document_id": "doc-1", "kind": "paper"},
            ),
        )
        return store, results

    store, results = asyncio.run(run())
    client = store._client

    assert client.search_calls[0]["filter"] == 'document_id == "doc-1" and kind == "paper"'
    assert client.search_calls[0]["output_fields"] == ["text", "*"]
    assert results[0].id == "chunk-001"
    assert results[0].score == 0.9
    assert results[0].text == "hello"
    assert results[0].metadata == {"document_id": "doc-1", "kind": "paper"}


def test_milvus_vector_store_recovers_existing_collection_config(monkeypatch):
    install_fake_pymilvus(monkeypatch)

    async def run():
        store = MilvusVectorStore()
        await store.create_collection(VectorCollectionConfig(name="chunks", dimension=3))
        client = store._client

        restored_store = MilvusVectorStore(client=client)
        results = await restored_store.search(
            "chunks",
            VectorQuery(vector=[0.1, 0.2, 0.3]),
        )
        return restored_store, results

    store, results = asyncio.run(run())

    assert store._collection_configs["chunks"] == VectorCollectionConfig(
        name="chunks",
        dimension=3,
        metric="cosine",
    )
    assert results[0].id == "chunk-001"


def test_milvus_vector_store_delete_count_and_close(monkeypatch):
    install_fake_pymilvus(monkeypatch)

    async def run():
        store = MilvusVectorStore()
        await store.create_collection(VectorCollectionConfig(name="chunks", dimension=3))
        await store.delete("chunks", ["chunk-001"])
        count = await store.count("chunks")
        await store.aclose()
        return store, count

    store, count = asyncio.run(run())
    client = store._client

    assert client.deleted == [{"collection_name": "chunks", "ids": ["chunk-001"]}]
    assert client.flushed == ["chunks"]
    assert count == 3
    assert client.closed is True
