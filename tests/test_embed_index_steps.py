import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingRequest, EmbeddingResult  # noqa: E402
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore  # noqa: E402
from heta_framework.common.stores.vector import VectorQuery  # noqa: E402
from heta_framework.kb.chunking import ChunkEmbedding, ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import DocumentParserRegistry, ParsedSource, TextParser  # noqa: E402
from heta_framework.kb.search import (  # noqa: E402
    QueryEngineRegistry,
    SearchAsset,
    SearchAssetCollection,
)
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
    def __init__(self):
        self.requests = []

    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.requests.append(request)
        vectors = []
        for text in request.texts:
            vectors.append([float(len(text)), float(text.count("a")), 1.0])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


def test_embed_chunks_declares_requirements_and_capabilities():
    step = EmbedChunks()

    assert step.name == "embed_chunks"
    assert step.config.batch_size == 10
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


def test_wiki_vector_preset_declares_separate_artifacts_and_search_asset():
    embed_step = EmbedChunks(EmbedChunksConfig(preset="wiki"))
    index_step = IndexVectors(IndexVectorsConfig(preset="wiki"))

    assert embed_step.config.embeddings_prefix == "wiki_embeddings"
    assert embed_step.requirements.artifacts == frozenset({"wiki_chunk_keys"})
    assert embed_step.capabilities.artifacts == frozenset(
        {"embed_wiki_chunks_result", "wiki_chunk_embedding_keys"}
    )
    assert index_step.config.collection_names.chunks == "wiki_chunks"
    assert index_step.requirements.artifacts == frozenset(
        {"wiki_chunk_keys", "wiki_chunk_embedding_keys"}
    )
    assert index_step.capabilities.artifacts == frozenset({"index_wiki_vectors_result"})
    assert index_step.capabilities.queries == frozenset({"wiki_vector_search"})
    assert index_step.capabilities.search_assets[0].kind == "wiki_chunk_vector_index"
    assert index_step.capabilities.search_assets[0].name == "wiki_chunks"


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


def test_wiki_vector_preset_embeds_and_indexes_separate_artifacts(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    vector_store = InMemoryVectorStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
        }
    )
    source = ParsedSource(
        key="wiki/pages/1-heta.md",
        name="1-heta.md",
        file_type="wiki",
        content_sha256="b" * 64,
    )
    chunk = ParsedChunk(
        chunk_id="wiki_page_1_chunk_0",
        document_id="wiki_page_1",
        source=source,
        page_index=0,
        chunk_index=0,
        text="Heta wiki page",
        token_start=0,
        token_end=14,
        heading_path=("Models", "Tool Calling"),
        origin_source=ParsedSource(
            key="raw/heta.md",
            name="heta.md",
            file_type="md",
            content_sha256="a" * 64,
        ),
    )

    async def run():
        await object_store.put("wiki_chunks/wiki_page_1_chunk_0.json", chunk.to_json_bytes())
        context.set_artifact("wiki_chunk_keys", ("wiki_chunks/wiki_page_1_chunk_0.json",))
        await EmbedChunks(EmbedChunksConfig(preset="wiki")).run(context)
        await IndexVectors(IndexVectorsConfig(preset="wiki")).run(context)
        return await vector_store.search(
            "wiki_chunks",
            VectorQuery(vector=[14.0, 2.0, 1.0], top_k=3),
        )

    results = asyncio.run(run())

    assert "chunk_embedding_keys" not in context.artifacts
    assert "index_vectors_result" not in context.artifacts
    assert context.artifacts["wiki_chunk_embedding_keys"] == (
        "wiki_embeddings/wiki_page_1_chunk_0.json",
    )
    assert context.artifacts["embed_wiki_chunks_result"].chunk_count == 1
    assert context.artifacts["index_wiki_vectors_result"].indexed_count == 1
    assert results[0].metadata["source_file_type"] == "wiki"
    assert results[0].metadata["origin_source_key"] == "raw/heta.md"
    assert results[0].metadata["heading_path"] == "Models > Tool Calling"


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


def test_embed_chunks_uses_portable_default_batch_size(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeEmbeddingModel()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "models.embedding": model,
        }
    )
    source = ParsedSource(
        key="raw/doc.txt",
        name="doc.txt",
        file_type="txt",
        content_sha256="a" * 64,
    )

    async def run():
        chunk_keys = []
        for index in range(21):
            chunk = ParsedChunk(
                chunk_id=f"chunk_{index}",
                document_id="doc_1",
                source=source,
                page_index=0,
                chunk_index=index,
                text=f"chunk text {index}",
                token_start=index,
                token_end=index + 1,
            )
            key = f"chunks/{chunk.chunk_id}.json"
            await object_store.put(key, chunk.to_json_bytes())
            chunk_keys.append(key)
        context.set_artifact("chunk_keys", tuple(chunk_keys))
        await EmbedChunks().run(context)

    asyncio.run(run())

    assert [len(request.texts) for request in model.requests] == [10, 10, 1]


def test_embed_chunks_reuses_existing_embedding_json(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeEmbeddingModel()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "models.embedding": model,
        }
    )
    chunk = ParsedChunk(
        chunk_id="chunk_cached",
        document_id="doc_cached",
        source=ParsedSource(
            key="raw/doc.txt",
            name="doc.txt",
            file_type="txt",
            content_sha256="a" * 64,
        ),
        page_index=0,
        chunk_index=0,
        text="cached embedding",
        token_start=0,
        token_end=16,
    )
    embedding = ChunkEmbedding(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        model_name="fake-embedding",
        vector=[1.0, 2.0, 3.0],
        dimension=3,
    )

    async def run():
        await object_store.put("chunks/chunk_cached.json", chunk.to_json_bytes())
        await object_store.put("embeddings/chunk_cached.json", embedding.to_json_bytes())
        context.set_artifact("chunk_keys", ("chunks/chunk_cached.json",))
        await EmbedChunks().run(context)

    asyncio.run(run())

    assert model.requests == []
    assert context.artifacts["chunk_embedding_keys"] == ("embeddings/chunk_cached.json",)


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


def test_chunk_vector_presets_reject_mixed_artifacts():
    with pytest.raises(ValueError, match="chunk_keys_artifact"):
        EmbedChunksConfig(preset="wiki", chunk_keys_artifact="chunk_keys")

    with pytest.raises(ValueError, match="chunk_embedding_keys_artifact"):
        IndexVectorsConfig(
            preset="wiki",
            chunk_embedding_keys_artifact="chunk_embedding_keys",
        )


def test_chunk_vector_configs_preserve_v0_1_0_constructor_contract():
    embed = EmbedChunksConfig(
        "legacy_embeddings",
        32,
        "objects",
        "embedder",
        "legacy_chunk_keys",
    )
    index = IndexVectorsConfig(
        ChunkVectorCollections(chunks="legacy_chunks"),
        "dot",
        64,
        "objects",
        "vectors",
        "legacy_chunk_keys",
        "legacy_embedding_keys",
    )

    assert embed.embeddings_prefix == "legacy_embeddings"
    assert embed.batch_size == 32
    assert embed.object_store == "objects"
    assert embed.embedding_model == "embedder"
    assert embed.chunk_keys_artifact == "legacy_chunk_keys"
    assert embed.chunk_embedding_keys_artifact == "chunk_embedding_keys"
    assert embed.result_artifact == "embed_chunks_result"
    assert index.collection_names == ChunkVectorCollections(chunks="legacy_chunks")
    assert index.metric == "dot"
    assert index.batch_size == 64
    assert index.object_store == "objects"
    assert index.vector_store == "vectors"
    assert index.chunk_keys_artifact == "legacy_chunk_keys"
    assert index.chunk_embedding_keys_artifact == "legacy_embedding_keys"
    assert index.result_artifact == "index_vectors_result"
    assert index.query_mode == "vector_search"


def test_query_registry_discovers_wiki_vector_search():
    registry = QueryEngineRegistry.defaults()
    assets = SearchAssetCollection(
        (
            SearchAsset(
                kind="wiki_chunk_vector_index",
                name="wiki_chunks",
                store="vector",
                metadata={"collection": "wiki_chunks"},
            ),
            SearchAsset(
                kind="wiki_chunk_full_text_index",
                name="wiki_chunk_full_text",
                store="text_index",
                metadata={"index": "wiki_chunk_full_text"},
            ),
        )
    )
    available_modes = registry.available_modes(assets)

    assert "wiki_vector_search" in available_modes
    assert "wiki_full_text_search" in available_modes


def test_chunk_embedding_round_trips_json():
    embedding = ChunkEmbedding(
        chunk_id="chunk_1",
        document_id="doc_1",
        model_name="fake",
        vector=[1.0, 2.0],
        dimension=2,
    )

    assert ChunkEmbedding.from_json(embedding.to_json_bytes()) == embedding
