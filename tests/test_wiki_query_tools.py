import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    KnowledgeBaseBuilder,
    KnowledgeRecipe,
    KnowledgeStores,
    QueryCitation,
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    QueryResponse,
    QueryResult,
    QueryToolBudget,
    QueryToolContext,
    QueryToolRegistry,
    ReadRawObjectTool,
    ReadWikiIndexTool,
    ReadWikiPageTool,
    SearchAsset,
    SearchAssetCollection,
    SearchAssetRef,
    SearchWikiTool,
)


class FakeWikiSearchEngine:
    mode = "wiki_vector_search"
    required_assets = frozenset({SearchAssetRef(kind="wiki_chunk_vector_index")})

    def __init__(self):
        self.requests = []

    async def query(self, request, context):
        self.requests.append(request)
        result = QueryResult(
            id="wiki_chunk_1",
            text="Page: Heta\nSummary: Heta builds wiki pages.",
            score=0.91,
            kind="chunk",
            source={
                "object_key": "wiki/pages/1-heta.md",
                "object_name": "1-heta.md",
                "object_type": "wiki",
            },
            metadata={
                "source_key": "wiki/pages/1-heta.md",
                "heading_path": "Summary",
            },
        )
        return QueryResponse(
            mode=self.mode,
            results=(result,),
            citations=(
                QueryCitation(
                    id="citation_1",
                    result_id=result.id,
                    source=result.source,
                    text=result.text,
                ),
            ),
        )


async def _tool_context(
    *,
    recipe=None,
    assets=(),
    engines=None,
    initial_artifacts=None,
    request=None,
):
    recipe = recipe or KnowledgeRecipe()
    build_result = await KnowledgeBaseBuilder().build(
        recipe,
        initial_artifacts=initial_artifacts or {},
    )
    query_context = QueryContext(
        recipe=recipe,
        run_record=build_result.record,
        assets=SearchAssetCollection(assets),
        engines=engines or QueryEngineRegistry(()),
    )
    return QueryToolContext(
        query_context=query_context,
        request=request or QueryRequest(text="What is Heta?"),
        budget=QueryToolBudget(max_tool_result_chars=500),
    )


def test_read_wiki_index_reads_artifact_key(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path)
        await object_store.put("wiki/custom_index.md", b"# Wiki Index\n\n- [Heta](pages/1-heta.md)")
        context = await _tool_context(
            recipe=KnowledgeRecipe(stores=KnowledgeStores(objects=object_store)),
            initial_artifacts={"wiki_index_key": "wiki/custom_index.md"},
        )
        return await ReadWikiIndexTool().run({}, context)

    result = asyncio.run(run())

    assert result.content.startswith("Wiki Index (wiki/custom_index.md):")
    assert "[Heta](pages/1-heta.md)" in result.content
    assert result.results == ()
    assert result.citations == ()
    assert result.metadata["citable"] is False


def test_read_wiki_page_returns_citable_page_evidence(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path)
        await object_store.put(
            "wiki/pages/1-heta.md",
            b"# Heta\n\n## Summary\n\nHeta builds wiki pages.",
        )
        context = await _tool_context(
            recipe=KnowledgeRecipe(stores=KnowledgeStores(objects=object_store)),
        )
        return await ReadWikiPageTool().run({"path": "pages/1-heta.md"}, context)

    result = asyncio.run(run())

    assert result.content.startswith("Wiki Page: pages/1-heta.md")
    assert result.results[0].kind == "wiki_page"
    assert result.results[0].source["object_key"] == "wiki/pages/1-heta.md"
    assert result.results[0].source["path"] == "pages/1-heta.md"
    assert result.citations[0].result_id == "wiki/pages/1-heta.md"
    assert result.metadata["citable"] is True


def test_read_wiki_page_rejects_non_page_path(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path)
        context = await _tool_context(
            recipe=KnowledgeRecipe(stores=KnowledgeStores(objects=object_store)),
        )
        return await ReadWikiPageTool().run({"path": "../raw/doc.txt"}, context)

    result = asyncio.run(run())

    assert result.is_error
    assert result.content.startswith("error:")
    assert result.metadata["citable"] is False


def test_read_raw_object_is_context_only(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path)
        await object_store.put("raw/doc.txt", b"Original source text for inspection.")
        context = await _tool_context(
            recipe=KnowledgeRecipe(stores=KnowledgeStores(objects=object_store)),
        )
        return await ReadRawObjectTool(max_chars=50).run({"path": "raw/doc.txt"}, context)

    result = asyncio.run(run())

    assert result.content.startswith("Raw Object: raw/doc.txt")
    assert "[truncated]" in result.content
    assert result.results == ()
    assert result.citations == ()
    assert result.metadata["citable"] is False
    assert result.metadata["raw_context_only"] is True


def test_search_wiki_delegates_to_registered_query_engine():
    async def run():
        engine = FakeWikiSearchEngine()
        context = await _tool_context(
            assets=(
                SearchAsset(
                    kind="wiki_chunk_vector_index",
                    name="wiki_chunks",
                    store="vector",
                ),
            ),
            engines=QueryEngineRegistry((engine,)),
            request=QueryRequest(
                text="initial question",
                top_k=8,
                options={"generate_answer": True},
            ),
        )
        result = await SearchWikiTool(
            query_mode="wiki_vector_search",
            asset_refs=frozenset({SearchAssetRef(kind="wiki_chunk_vector_index")}),
            max_top_k=5,
        ).run(
            {"query": "Heta wiki", "top_k": 9},
            context,
        )
        return engine, result

    engine, result = asyncio.run(run())

    assert engine.requests[0].text == "Heta wiki"
    assert engine.requests[0].top_k == 5
    assert engine.requests[0].options["generate_answer"] is False
    assert "Search mode: wiki_vector_search" in result.content
    assert "wiki_chunk_1" in result.content
    assert result.results[0].id == "wiki_chunk_1"
    assert result.citations[0].id == "citation_1"
    assert result.metadata["citable"] is True


def test_wiki_query_tools_can_be_registered_for_model_definitions():
    registry = QueryToolRegistry(
        (
            ReadWikiIndexTool(),
            SearchWikiTool(),
            ReadWikiPageTool(),
            ReadRawObjectTool(),
        )
    )

    assert registry.names == frozenset(
        {
            "read_wiki_index",
            "search_wiki",
            "read_wiki_page",
            "read_raw_object",
        }
    )
    assert [definition.name for definition in registry.model_tool_definitions()] == [
        "read_wiki_index",
        "search_wiki",
        "read_wiki_page",
        "read_raw_object",
    ]
