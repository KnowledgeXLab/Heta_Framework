import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (
    InMemoryVectorStore,
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
    VectorStoreProtocol,
)


def test_in_memory_vector_store_satisfies_protocol():
    assert isinstance(InMemoryVectorStore(), VectorStoreProtocol)


def test_in_memory_vector_store_searches_and_filters():
    async def run():
        store = InMemoryVectorStore()
        await store.create_collection(VectorCollectionConfig(name="chunks", dimension=2))
        await store.upsert(
            "chunks",
            [
                VectorRecord(
                    id="a",
                    vector=[1, 0],
                    text="alpha",
                    metadata={"document_id": "doc-1", "kind": "paper"},
                ),
                VectorRecord(
                    id="b",
                    vector=[0, 1],
                    text="beta",
                    metadata={"document_id": "doc-2", "kind": "note"},
                ),
                VectorRecord(
                    id="c",
                    vector=[0.9, 0.1],
                    text="gamma",
                    metadata={"document_id": "doc-3", "kind": "paper"},
                ),
            ],
        )

        results = await store.search(
            "chunks",
            VectorQuery(vector=[1, 0], top_k=2, filter={"kind": "paper"}),
        )
        return await store.count("chunks"), results

    count, results = asyncio.run(run())

    assert count == 3
    assert [result.id for result in results] == ["a", "c"]
    assert results[0].text == "alpha"
    assert results[0].metadata == {"document_id": "doc-1", "kind": "paper"}


def test_in_memory_vector_store_upsert_replaces_records():
    async def run():
        store = InMemoryVectorStore()
        await store.create_collection(VectorCollectionConfig(name="chunks", dimension=2))
        await store.upsert("chunks", [VectorRecord(id="a", vector=[1, 0], text="old")])
        await store.upsert("chunks", [VectorRecord(id="a", vector=[0, 1], text="new")])
        return await store.search("chunks", VectorQuery(vector=[0, 1], top_k=1))

    results = asyncio.run(run())

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].text == "new"


def test_in_memory_vector_store_deletes_records():
    async def run():
        store = InMemoryVectorStore()
        await store.create_collection(VectorCollectionConfig(name="chunks", dimension=2))
        await store.upsert(
            "chunks",
            [
                VectorRecord(id="a", vector=[1, 0]),
                VectorRecord(id="b", vector=[0, 1]),
            ],
        )
        await store.delete("chunks", ["a"])
        return await store.count("chunks"), await store.search(
            "chunks",
            VectorQuery(vector=[1, 0], top_k=10),
        )

    count, results = asyncio.run(run())

    assert count == 1
    assert [result.id for result in results] == ["b"]


def test_in_memory_vector_store_validates_dimensions():
    async def run():
        store = InMemoryVectorStore()
        await store.create_collection(VectorCollectionConfig(name="chunks", dimension=2))
        await store.upsert("chunks", [VectorRecord(id="bad", vector=[1, 2, 3])])

    with pytest.raises(ValueError, match="dimension mismatch"):
        asyncio.run(run())


def test_in_memory_vector_store_supports_l2_metric():
    async def run():
        store = InMemoryVectorStore()
        await store.create_collection(
            VectorCollectionConfig(name="points", dimension=2, metric="l2")
        )
        await store.upsert(
            "points",
            [
                VectorRecord(id="near", vector=[1, 1]),
                VectorRecord(id="far", vector=[10, 10]),
            ],
        )
        return await store.search("points", VectorQuery(vector=[0, 0], top_k=2))

    results = asyncio.run(run())

    assert [result.id for result in results] == ["near", "far"]
