import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    EmbeddingRequest,
    EmbeddingResult,
    ModelRequest,
    ModelResult,
)
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore  # noqa: E402
from heta_framework.common.stores import SQLStore  # noqa: E402
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserRegistry,
    ParsedSource,
    TextParser,
)
from heta_framework.kb.steps import (  # noqa: E402
    ChunkTableNames,
    EmbedChunks,
    MergeChunks,
    MergeChunksConfig,
    ParseDocuments,
    PersistChunks,
    PersistChunksConfig,
    RechunkDocuments,
    RechunkDocumentsConfig,
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
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lower else 0.0,
                    1.0 if "beta" in lower else 0.0,
                    0.1,
                ]
            )
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


class FakeLanguageModel:
    @property
    def model_name(self):
        return "fake-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            text='{"text": "alpha beta merged fact.", "merge_id": [1]}',
            model_name=self.model_name,
        )

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    async def stream(self, request):
        if False:
            yield None


class MergeTwoCandidatesLanguageModel(FakeLanguageModel):
    async def invoke(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            text='{"text": "alpha beta gamma merged fact.", "merge_id": [1, 2]}',
            model_name=self.model_name,
        )


def test_parsed_chunk_accepts_parent_chunk_ids():
    chunk = ParsedChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        source=ParsedSource(
            key="raw/doc.txt",
            name="doc.txt",
            file_type="txt",
            content_sha256="a" * 64,
        ),
        page_index=0,
        chunk_index=0,
        text="alpha",
        token_start=0,
        token_end=5,
        parent_chunk_ids=("chunk_a",),
    )

    assert ParsedChunk.from_json(chunk.to_json_bytes()).parent_chunk_ids == ("chunk_a",)


def test_merge_chunks_creates_merge_collection_and_outputs_active_keys(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    vector_store = InMemoryVectorStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
            "models.language": FakeLanguageModel(),
            "parsers.documents": DocumentParserRegistry([TextParser()]),
        }
    )

    async def run():
        await object_store.put("raw/doc.txt", b"alpha beta. alpha beta.")
        await ParseDocuments().run(context)
        await SplitDocuments(
            SplitDocumentsConfig(chunk_size=12, overlap=0, encoding_name="unicode")
        ).run(context)
        await EmbedChunks().run(context)
        await MergeChunks(
            MergeChunksConfig(
                merge_collection="merge_test",
                top_k=4,
                min_similarity=0.1,
                max_rounds=1,
            )
        ).run(context)
        return await vector_store.count("merge_test")

    merge_collection_count = asyncio.run(run())

    result = context.artifacts["merge_chunks_result"]
    merged_keys = context.artifacts["merged_chunk_keys"]
    assert result.collection == "merge_test"
    assert result.input_chunk_count >= 2
    assert result.active_chunk_count == len(merged_keys)
    assert result.merged_count >= 1
    assert merge_collection_count == len(merged_keys)

    async def read_merged():
        merged_only = [key for key in merged_keys if key.startswith("merged_chunks/")]
        return ParsedChunk.from_json(await object_store.get(merged_only[0]))

    merged_chunk = asyncio.run(read_merged())
    assert len(merged_chunk.parent_chunk_ids) == 2


def test_merge_chunks_can_merge_multiple_candidates_in_one_llm_call(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    vector_store = InMemoryVectorStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
            "models.language": MergeTwoCandidatesLanguageModel(),
        }
    )
    source = ParsedSource(
        key="raw/doc.txt",
        name="doc.txt",
        file_type="txt",
        content_sha256="a" * 64,
    )
    chunks = [
        ParsedChunk(
            chunk_id=f"chunk_{index}",
            document_id="doc_1",
            source=source,
            page_index=0,
            chunk_index=index,
            text=text,
            token_start=index * 10,
            token_end=index * 10 + 10,
        )
        for index, text in enumerate(["alpha beta", "alpha beta copy", "alpha beta duplicate"])
    ]

    async def run():
        chunk_keys = []
        embedding_keys = []
        for chunk in chunks:
            chunk_key = f"chunks/{chunk.chunk_id}.json"
            embedding_key = f"embeddings/{chunk.chunk_id}.json"
            await object_store.put(chunk_key, chunk.to_json_bytes())
            await object_store.put(
                embedding_key,
                (
                    (
                        '{"chunk_id":"%s","document_id":"doc_1","model_name":"fake",'
                        '"vector":[1.0,1.0,0.1],"dimension":3}'
                    )
                    % chunk.chunk_id
                ).encode("utf-8"),
            )
            chunk_keys.append(chunk_key)
            embedding_keys.append(embedding_key)
        context.set_artifact("chunk_keys", tuple(chunk_keys))
        context.set_artifact("chunk_embedding_keys", tuple(embedding_keys))
        await MergeChunks(
            MergeChunksConfig(
                merge_collection="merge_multi",
                top_k=4,
                num_topk_candidates=2,
                min_similarity=0.1,
                max_rounds=1,
            )
        ).run(context)

    asyncio.run(run())

    merged_keys = context.artifacts["merged_chunk_keys"]
    merged_only = [key for key in merged_keys if key.startswith("merged_chunks/")]
    assert len(merged_keys) == 1
    assert len(merged_only) == 1

    async def read_merged():
        return ParsedChunk.from_json(await object_store.get(merged_only[0]))

    merged_chunk = asyncio.run(read_merged())
    assert merged_chunk.parent_chunk_ids == ("chunk_0", "chunk_1", "chunk_2")


def test_rechunk_documents_preserves_parent_chunk_ids(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects": object_store})
    source = ParsedSource(
        key="raw/doc.txt",
        name="doc.txt",
        file_type="txt",
        content_sha256="a" * 64,
    )
    chunks = [
        ParsedChunk(
            chunk_id="chunk_a",
            document_id="doc_1",
            source=source,
            page_index=0,
            chunk_index=0,
            text="alpha beta",
            token_start=0,
            token_end=10,
        ),
        ParsedChunk(
            chunk_id="chunk_b",
            document_id="doc_1",
            source=source,
            page_index=0,
            chunk_index=1,
            text="gamma delta",
            token_start=10,
            token_end=20,
            parent_chunk_ids=("chunk_root",),
        ),
    ]

    async def run():
        keys = []
        for chunk in chunks:
            key = f"merged/{chunk.chunk_id}.json"
            await object_store.put(key, chunk.to_json_bytes())
            keys.append(key)
        context.set_artifact("merged_chunk_keys", tuple(keys))
        await RechunkDocuments(
            RechunkDocumentsConfig(
                chunk_size=100,
                overlap=0,
                encoding_name="unicode",
            )
        ).run(context)

    asyncio.run(run())

    keys = context.artifacts["rechunked_chunk_keys"]
    assert len(keys) == 1

    async def read_rechunked():
        return ParsedChunk.from_json(await object_store.get(keys[0]))

    rechunked = asyncio.run(read_rechunked())
    assert rechunked.parent_chunk_ids == ("chunk_a", "chunk_root")
    assert "alpha beta" in rechunked.text
    assert "gamma delta" in rechunked.text


def test_persist_chunks_writes_rechunked_chunks_to_sql(tmp_path):
    object_store = LocalObjectStore(tmp_path / "objects")
    sql_store = SQLStore(f"sqlite:///{tmp_path / 'chunks.db'}")
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.sql": sql_store,
        }
    )
    chunk = ParsedChunk(
        chunk_id="chunk_sql",
        document_id="doc_sql",
        source=ParsedSource(
            key="raw/doc.txt",
            name="doc.txt",
            file_type="txt",
            content_sha256="a" * 64,
        ),
        page_index=0,
        chunk_index=0,
        text="alpha persisted",
        token_start=0,
        token_end=15,
        parent_chunk_ids=("chunk_a", "chunk_b"),
    )

    async def run():
        await object_store.put("rechunked/chunk_sql.json", chunk.to_json_bytes())
        context.set_artifact("rechunked_chunk_keys", ("rechunked/chunk_sql.json",))
        await PersistChunks(
            PersistChunksConfig(table_names=ChunkTableNames(chunks="chunks_test"))
        ).run(context)
        rows = await sql_store.fetch_all(
            "SELECT chunk_id, content_text, source_chunk FROM chunks_test"
        )
        await sql_store.aclose()
        return rows

    rows = asyncio.run(run())

    assert context.artifacts["persist_chunks_result"].chunk_count == 1
    assert rows == [
        {
            "chunk_id": "chunk_sql",
            "content_text": "alpha persisted",
            "source_chunk": '["chunk_a", "chunk_b"]',
        }
    ]


def test_persist_chunks_config_validates_table_names():
    with pytest.raises(ValueError, match="table_names.chunks"):
        PersistChunksConfig(table_names=ChunkTableNames(chunks="bad-name"))


def test_persist_chunks_declares_keyword_search_asset():
    step = PersistChunks(
        PersistChunksConfig(
            table_names=ChunkTableNames(chunks="chunks_test"),
            dialect="postgresql",
            sql_store="primary",
        )
    )

    capabilities = step.capabilities

    assert "keyword_search" in capabilities.queries
    assert capabilities.search_assets[0].kind == "chunk_text_index"
    assert capabilities.search_assets[0].name == "chunks_test"
    assert capabilities.search_assets[0].store == "stores.sql.primary"
    assert capabilities.search_assets[0].metadata["dialect"] == "postgresql"
