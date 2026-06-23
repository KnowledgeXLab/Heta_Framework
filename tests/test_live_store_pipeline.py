import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResult,
    LanguageModel,
    ModelRequest,
    ModelResult,
)
from heta_framework.common.stores import (  # noqa: E402
    LocalObjectStore,
    MilvusVectorStore,
    SQLStore,
)
from heta_framework.common.stores.vector import VectorQuery  # noqa: E402
from heta_framework.kb.chunking import ChunkEmbedding  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    EmbedChunks,
    ChunkTableNames,
    ChunkVectorCollections,
    IndexVectors,
    IndexVectorsConfig,
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
from heta_framework.kb.parsing import DocumentParserRegistry, TextParser  # noqa: E402


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
        return "fake-live-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = []
        for text in request.texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lower else 0.0,
                    1.0 if "beta" in lower else 0.0,
                    1.0 if "gamma" in lower else 0.0,
                ]
            )
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


class FakeLanguageModel:
    @property
    def model_name(self):
        return "fake-live-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            text='{"text": "alpha beta merged evidence.", "merge_id": [1]}',
            model_name=self.model_name,
        )

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    async def stream(self, request):
        if False:
            yield None


def _make_embedding_model():
    model_name = os.getenv("HETA_LIVE_EMBEDDING_MODEL")
    if not model_name:
        return FakeEmbeddingModel()
    return EmbeddingModel(
        model_name=model_name,
        api_key=os.getenv("HETA_LIVE_EMBEDDING_API_KEY") or None,
        api_base=os.getenv("HETA_LIVE_EMBEDDING_API_BASE") or None,
        request_timeout=float(os.getenv("HETA_LIVE_MODEL_TIMEOUT", "120")),
        max_retries=int(os.getenv("HETA_LIVE_MODEL_MAX_RETRIES", "2")),
        max_concurrent_requests=int(os.getenv("HETA_LIVE_MODEL_MAX_CONCURRENCY", "2")),
    )


def _make_language_model():
    model_name = os.getenv("HETA_LIVE_LANGUAGE_MODEL")
    if not model_name:
        return FakeLanguageModel()
    return LanguageModel(
        model_name=model_name,
        api_key=os.getenv("HETA_LIVE_LANGUAGE_API_KEY") or None,
        api_base=os.getenv("HETA_LIVE_LANGUAGE_API_BASE") or None,
        request_timeout=float(os.getenv("HETA_LIVE_MODEL_TIMEOUT", "120")),
        max_retries=int(os.getenv("HETA_LIVE_MODEL_MAX_RETRIES", "2")),
        max_concurrent_requests=int(os.getenv("HETA_LIVE_MODEL_MAX_CONCURRENCY", "2")),
    )


@pytest.mark.live
def test_live_milvus_sql_store_pipeline(tmp_path):
    milvus_uri = os.getenv("HETA_LIVE_MILVUS_URI")
    sql_url = os.getenv("HETA_LIVE_SQL_URL") or os.getenv("HETA_LIVE_POSTGRES_URL")
    if not milvus_uri or not sql_url:
        pytest.skip("set HETA_LIVE_MILVUS_URI and HETA_LIVE_SQL_URL to run live store smoke")
    sql_dialect = os.getenv(
        "HETA_LIVE_SQL_DIALECT",
        (
            "postgresql"
            if os.getenv("HETA_LIVE_POSTGRES_URL") and not os.getenv("HETA_LIVE_SQL_URL")
            else "generic"
        ),
    )

    suffix = uuid.uuid4().hex[:12]
    vector_collection = f"heta_live_chunks_{suffix}"
    merge_collection = f"heta_live_merge_{suffix}"
    sql_table = f"heta_live_chunks_{suffix}"

    object_store = LocalObjectStore(tmp_path / "objects")
    vector_store = MilvusVectorStore(
        uri=milvus_uri,
        token=os.getenv("HETA_LIVE_MILVUS_TOKEN") or None,
        db_name=os.getenv("HETA_LIVE_MILVUS_DB") or None,
        timeout=float(os.getenv("HETA_LIVE_MILVUS_TIMEOUT", "10")),
    )
    sql_store = SQLStore(sql_url)
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.vector": vector_store,
            "stores.sql": sql_store,
            "models.embedding": _make_embedding_model(),
            "models.language": _make_language_model(),
            "parsers.documents": DocumentParserRegistry([TextParser()]),
        }
    )

    async def run():
        try:
            await object_store.put("raw/doc.txt", b"alpha beta. alpha beta copy. gamma.")
            await ParseDocuments().run(context)
            await SplitDocuments(
                SplitDocumentsConfig(chunk_size=14, overlap=0, encoding_name="unicode")
            ).run(context)
            await EmbedChunks().run(context)
            await IndexVectors(
                IndexVectorsConfig(
                    collection_names=ChunkVectorCollections(chunks=vector_collection)
                )
            ).run(context)
            await MergeChunks(
                MergeChunksConfig(
                    merge_collection=merge_collection,
                    top_k=4,
                    num_topk_candidates=2,
                    min_similarity=0.1,
                    max_rounds=1,
                )
            ).run(context)
            await RechunkDocuments(
                RechunkDocumentsConfig(chunk_size=100, overlap=0, encoding_name="unicode")
            ).run(context)
            await PersistChunks(
                PersistChunksConfig(
                    table_names=ChunkTableNames(chunks=sql_table),
                    dialect=sql_dialect,
                )
            ).run(context)
            embedding_key = context.artifacts["chunk_embedding_keys"][0]
            query_embedding = ChunkEmbedding.from_json(await object_store.get(embedding_key))
            search_results = await vector_store.search(
                vector_collection,
                VectorQuery(vector=query_embedding.vector, top_k=2),
            )
            rows = await sql_store.fetch_all(
                f"SELECT chunk_id, content_text, source_chunk FROM {sql_table}"
            )
            return search_results, rows
        finally:
            await vector_store.drop_collection(vector_collection)
            await vector_store.drop_collection(merge_collection)
            await sql_store.execute(f"DROP TABLE IF EXISTS {sql_table}")
            await vector_store.aclose()
            await sql_store.aclose()

    results, rows = asyncio.run(run())

    assert results
    assert rows
    assert context.artifacts["persist_chunks_result"].chunk_count == len(rows)
    assert rows[0]["source_chunk"]
