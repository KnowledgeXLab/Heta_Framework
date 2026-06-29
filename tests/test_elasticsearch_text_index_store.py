import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (  # noqa: E402
    ElasticsearchTextIndexStore,
    ElasticsearchTextIndexStoreConfig,
    TextIndexConfig,
    TextIndexRecord,
    TextIndexStoreProtocol,
    TextQuery,
)


class FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, dict] = {}
        self.deleted: list[str] = []

    async def exists(self, *, index: str) -> bool:
        return index in self.created

    async def create(self, *, index: str, mappings: dict) -> None:
        self.created[index] = mappings

    async def delete(self, *, index: str, ignore_unavailable: bool) -> None:
        self.deleted.append(index)
        self.created.pop(index, None)


class FakeElasticsearch:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.documents: dict[str, dict[str, dict]] = {}
        self.search_requests: list[dict] = []
        self.closed = False

    async def search(self, *, index: str, query: dict, size: int) -> dict:
        self.search_requests.append({"index": index, "query": query, "size": size})
        docs = self.documents.get(index, {})
        query_text = query["bool"]["must"][0]["match"]["content_text"].lower()
        hits = []
        for doc_id, source in docs.items():
            if query_text in source["content_text"].lower():
                hits.append({"_id": doc_id, "_score": 3.5, "_source": source})
        return {"hits": {"hits": hits[:size]}}

    async def count(self, *, index: str) -> dict:
        return {"count": len(self.documents.get(index, {}))}

    async def close(self) -> None:
        self.closed = True


class FakeBulkHelper:
    def __init__(self, client: FakeElasticsearch) -> None:
        self.client = client
        self.calls: list[dict] = []

    async def __call__(self, client: FakeElasticsearch, actions: list[dict], **kwargs) -> tuple[int, list]:
        assert client is self.client
        self.calls.append({"actions": actions, "kwargs": kwargs})
        for action in actions:
            index = action["_index"]
            doc_id = action["_id"]
            if action["_op_type"] == "index":
                client.documents.setdefault(index, {})[doc_id] = action["_source"]
            elif action["_op_type"] == "delete":
                client.documents.setdefault(index, {}).pop(doc_id, None)
        return len(actions), []


def test_elasticsearch_text_index_store_implements_protocol():
    client = FakeElasticsearch()
    store = ElasticsearchTextIndexStore(client=client, bulk_helper=FakeBulkHelper(client))

    assert isinstance(store, TextIndexStoreProtocol)


def test_elasticsearch_text_index_store_indexes_searches_and_deletes():
    async def run():
        client = FakeElasticsearch()
        bulk_helper = FakeBulkHelper(client)
        store = ElasticsearchTextIndexStore(
            ElasticsearchTextIndexStoreConfig(hosts="http://localhost:9200"),
            client=client,
            bulk_helper=bulk_helper,
        )

        await store.create_index(TextIndexConfig(name="chunks"))
        await store.create_index(TextIndexConfig(name="chunks"))
        await store.upsert(
            "chunks",
            [
                TextIndexRecord(
                    id="chunk_1",
                    text="Heta supports Elasticsearch full text search.",
                    metadata={"document_id": "doc_1", "source_key": "raw/a.txt"},
                ),
                TextIndexRecord(
                    id="chunk_2",
                    text="Vector search uses embeddings.",
                    metadata={"document_id": "doc_2", "source_key": "raw/b.txt"},
                ),
            ],
        )

        hits = await store.search("chunks", TextQuery(text="Elasticsearch", top_k=5))
        count_before_delete = await store.count("chunks")
        await store.delete("chunks", ["chunk_1"])
        count_after_delete = await store.count("chunks")
        await store.drop_index("chunks")
        await store.aclose()
        return client, bulk_helper, hits, count_before_delete, count_after_delete

    client, bulk_helper, hits, count_before_delete, count_after_delete = asyncio.run(run())

    assert list(client.indices.created) == []
    assert client.indices.deleted == ["chunks"]
    assert len(bulk_helper.calls) == 2
    assert bulk_helper.calls[0]["kwargs"]["refresh"] is True
    assert bulk_helper.calls[1]["kwargs"]["ignore_status"] == (404,)
    assert hits[0].id == "chunk_1"
    assert hits[0].text == "Heta supports Elasticsearch full text search."
    assert hits[0].score == 3.5
    assert hits[0].metadata["source_key"] == "raw/a.txt"
    assert count_before_delete == 2
    assert count_after_delete == 1
    assert client.closed is False


def test_elasticsearch_text_index_store_closes_owned_client():
    async def run():
        client = FakeElasticsearch()
        store = ElasticsearchTextIndexStore(client=client, bulk_helper=FakeBulkHelper(client))
        store._owns_client = True
        await store.aclose()
        return client.closed

    assert asyncio.run(run()) is True
