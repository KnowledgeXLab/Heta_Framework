import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import EmbeddingModel  # noqa: E402
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    DocumentParserRegistry,
    EmbedChunks,
    IndexVectors,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    ParseDocuments,
    SplitDocuments,
    SplitDocumentsConfig,
    TextParser,
)


pytestmark = pytest.mark.live


BUSINESS_DOCUMENTS = {
    "heta_recipe_architecture.txt": (
        "Heta framework uses KnowledgeRecipe to compose parsers, language models, "
        "embedding models, object stores, vector stores, SQL stores, and build steps. "
        "A recipe is the construction plan for a knowledge base. KnowledgeBase.create "
        "executes the recipe and records available query capabilities."
    ),
    "milvus_vector_deployment.txt": (
        "For production vector search, Heta can replace the in-memory vector store "
        "with Milvus. Milvus stores chunk vectors in a persistent collection and "
        "supports scalable nearest-neighbor retrieval for knowledge base queries."
    ),
    "parser_mineru_processing.txt": (
        "Document parsing in Heta is routed by DocumentParserRegistry. TextParser handles "
        "plain text and Markdown, while MinerU-style extractors can be used for PDF, Office, "
        "HTML, tables, and image-rich documents before chunks are embedded."
    ),
}


QUERIES = (
    (
        "How does Heta compose a knowledge base with KnowledgeRecipe?",
        "heta_recipe_architecture.txt",
    ),
    (
        "Which storage should be used for persistent production vector retrieval?",
        "milvus_vector_deployment.txt",
    ),
    (
        "How are PDF, HTML, tables, and image-rich files parsed before embedding?",
        "parser_mineru_processing.txt",
    ),
)


def test_live_vector_search_golden_business_case(tmp_path: Path) -> None:
    if os.getenv("HETA_RUN_LIVE_VECTOR_SMOKE") != "1":
        pytest.skip("set HETA_RUN_LIVE_VECTOR_SMOKE=1 to run live vector search smoke")
    asyncio.run(_run_live_vector_search(tmp_path))


async def _run_live_vector_search(tmp_path: Path) -> None:
    embedding = _load_embedding_model()
    object_store = LocalObjectStore(tmp_path / "objects")
    vector_store = InMemoryVectorStore()

    for name, text in BUSINESS_DOCUMENTS.items():
        await object_store.put(f"raw/{name}", text.encode("utf-8"))

    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(embedding=embedding),
        stores=KnowledgeStores(objects=object_store, vector=vector_store),
        steps=(
            ParseDocuments(),
            SplitDocuments(SplitDocumentsConfig(chunk_size=512, overlap=0)),
            EmbedChunks(),
            IndexVectors(),
        ),
    )
    recipe.require_valid()

    kb = await KnowledgeBase.create(
        recipe=recipe,
        name="live-vector-smoke",
        description="Live vector search smoke test with realistic business documents.",
    )

    assert kb.run_record.status == "succeeded", _format_step_errors(kb.run_record.step_records)
    assert "vector_search" in kb.available_queries

    failures: list[str] = []
    for query, expected_source in QUERIES:
        response = await kb.query(query, mode="vector_search", top_k=3)
        returned_sources = [
            str(result.source.get("source_name", ""))
            for result in response.results
        ]
        if expected_source not in returned_sources:
            failures.append(
                f"query={query!r}; expected={expected_source!r}; got={returned_sources!r}"
            )

    await embedding.aclose()
    await vector_store.aclose()
    await object_store.aclose()

    assert not failures, "\n".join(failures)


def _load_embedding_model() -> EmbeddingModel:
    api_key = os.getenv("HETA_LIVE_EMBEDDING_API_KEY")
    api_base = os.getenv("HETA_LIVE_EMBEDDING_API_BASE")
    model_name = os.getenv("HETA_LIVE_EMBEDDING_MODEL_NAME")

    if not api_key or not api_base or not model_name:
        config = _load_config_embedding()
        api_key = api_key or config.get("api_key")
        api_base = api_base or config.get("base_url")
        model_name = model_name or config.get("model")

    if not api_key or not api_base or not model_name:
        pytest.skip(
        "set HETA_LIVE_EMBEDDING_API_KEY/API_BASE/MODEL_NAME or configure "
            "config.yaml hetadb.embedding / hetadb.embedding_api"
        )

    return EmbeddingModel(
        model_name=_litellm_model_name(model_name),
        api_key=api_key,
        api_base=api_base,
        request_timeout=90,
        max_retries=1,
        max_concurrent_requests=2,
    )


def _load_config_embedding() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    hetadb = data.get("hetadb", {}) if isinstance(data, dict) else {}
    embedding = hetadb.get("embedding", {}) if isinstance(hetadb, dict) else {}
    if not embedding:
        embedding = hetadb.get("embedding_api", {}) if isinstance(hetadb, dict) else {}
    return embedding if isinstance(embedding, dict) else {}


def _litellm_model_name(model_name: str) -> str:
    known_prefixes = (
        "openai/",
        "azure/",
        "dashscope/",
        "gemini/",
        "anthropic/",
        "cohere/",
        "voyage/",
    )
    if model_name.startswith(known_prefixes):
        return model_name
    return f"openai/{model_name}"


def _format_step_errors(step_records: tuple[Any, ...]) -> str:
    return "\n".join(
        f"{record.step_name}: {record.status}: {record.error}"
        for record in step_records
        if record.status != "succeeded"
    )
