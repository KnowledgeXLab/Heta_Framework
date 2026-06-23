import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingRequest, EmbeddingResult  # noqa: E402
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore  # noqa: E402
from heta_framework.common.stores.vector import VectorQuery  # noqa: E402
from heta_framework.kb.chunking import ChunkEmbedding, ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import DocumentParserRegistry, TextParser  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    ChunkVectorCollections,
    EmbedChunks,
    EmbedChunksConfig,
    IndexVectors,
    IndexVectorsConfig,
    ParseDocuments,
    SplitDocuments,
    SplitDocumentsConfig,
)


class FakeContext:
    def __init__(self, components):
        self.components = components
        self.artifacts = {}

    def get_component(self, key):
        return self.components[key]

    def get_artifact(self, key):
        return self.artifacts[key]

    def set_artifact(self, key, value):
        self.artifacts[key] = value


class FakeEmbeddingModel:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = []
        for text in request.texts:
            vectors.append([float(len(text)), float(text.count("a")), 1.0])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


def test_embed_chunks_declares_requirements_and_capabilities():
    step = EmbedChunks()

    assert step.name == "embed_chunks"
    assert {ref.key for ref in step.requirements.components} == {
        "stores.objects",
        "models.embedding",
    }
    assert step.requirements.artifacts == frozenset({"chunk_keys"})
    assert step.capabilities.artifacts == frozenset(
        {"embed_chunks_result", "chunk_embedding_keys"}
    )


def test_index_vectors_declares_requirements_and_capabilities():
    step = IndexVectors()

    assert step.name == "index_vectors"
    assert {ref.key for ref in step.requirements.components} == {
        "stores.objects",
        "stores.vector",
    }
    assert step.requirements.artifacts == frozenset({"chunk_keys", "chunk_embedding_keys"})
    assert step.capabilities.artifacts == frozenset({"index_vectors_result"})
    assert step.capabilities.queries == frozenset({"vector_search"})


def test_parse_split_embed_index_enables_vector_search(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    vector_store = InMemoryVectorStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
            "parsers.documents": DocumentParserRegistry([TextParser()]),
        }
    )

    async def run():
        await object_store.put("raw/doc.txt", b"alpha beta gamma. alpha delta.")
        await ParseDocuments().run(context)
        await SplitDocuments(
            SplitDocumentsConfig(chunk_size=16, overlap=4, encoding_name="unicode")
        ).run(context)
        await EmbedChunks(EmbedChunksConfig(batch_size=2)).run(context)
        await IndexVectors().run(context)
        return await vector_store.search("chunks", VectorQuery(vector=[10.0, 2.0, 1.0], top_k=3))

    results = asyncio.run(run())

    assert context.artifacts["embed_chunks_result"].chunk_count == len(
        context.artifacts["chunk_keys"]
    )
    assert context.artifacts["index_vectors_result"].indexed_count == len(
        context.artifacts["chunk_keys"]
    )
    assert context.artifacts["index_vectors_result"].dimension == 3
    assert len(results) >= 1
    assert results[0].text is not None
    assert results[0].metadata["source_name"] == "doc.txt"
    assert results[0].metadata["embedding_model"] == "fake-embedding"


def test_embed_chunks_writes_chunk_embedding_json(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    context = FakeContext(
        {
            "stores.objects": object_store,
            "models.embedding": FakeEmbeddingModel(),
        }
    )

    async def run():
        await object_store.put("raw/doc.txt", b"alpha")
        context.components["parsers.documents"] = DocumentParserRegistry([TextParser()])
        await ParseDocuments().run(context)
        await SplitDocuments(SplitDocumentsConfig(encoding_name="unicode")).run(context)
        await EmbedChunks().run(context)

    asyncio.run(run())

    embedding_keys = context.artifacts["chunk_embedding_keys"]
    assert len(embedding_keys) == 1

    async def read_embedding():
        return ChunkEmbedding.from_json(await object_store.get(embedding_keys[0]))

    embedding = asyncio.run(read_embedding())

    assert embedding.model_name == "fake-embedding"
    assert embedding.dimension == 3
    assert embedding.vector == [5.0, 2.0, 1.0]


def test_index_vectors_rejects_missing_embedding(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    vector_store = InMemoryVectorStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
            "parsers.documents": DocumentParserRegistry([TextParser()]),
        }
    )

    async def run():
        await object_store.put("raw/doc.txt", b"alpha")
        await ParseDocuments().run(context)
        await SplitDocuments(SplitDocumentsConfig(encoding_name="unicode")).run(context)
        context.set_artifact("chunk_embedding_keys", ())
        await IndexVectors().run(context)

    with pytest.raises(ValueError, match="missing embedding"):
        asyncio.run(run())


def test_index_vectors_config_validates_metric():
    with pytest.raises(ValueError, match="metric"):
        IndexVectorsConfig(metric="bad")


def test_index_vectors_config_validates_collection_names():
    with pytest.raises(ValueError, match="collection_names.chunks"):
        IndexVectorsConfig(collection_names=ChunkVectorCollections(chunks=""))


def test_chunk_embedding_round_trips_json():
    embedding = ChunkEmbedding(
        chunk_id="chunk_1",
        document_id="doc_1",
        model_name="fake",
        vector=[1.0, 2.0],
        dimension=2,
    )

    assert ChunkEmbedding.from_json(embedding.to_json_bytes()) == embedding
