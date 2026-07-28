import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    ModelRequest,
    ModelResult,
    TokenUsage,
    ToolCall,
    ToolCallingModelRequest,
    ToolCallingModelResult,
    ToolMessage,
)
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    AgenticQueryEngine,
    KnowledgeBaseBuilder,
    KnowledgeModels,
    KnowledgeRecipe,
    KnowledgeStores,
    QueryCitation,
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    QueryResponse,
    QueryResult,
    QueryToolBudget,
    QueryToolResult,
    SearchAsset,
    SearchAssetCollection,
    SearchAssetRef,
)
from heta_framework.kb.search.engines import WikiHybridSearchEngine  # noqa: E402


class ScriptedToolCallingModel:
    model_name = "test/tool-calling"

    def __init__(self, *, planning_messages, synthesis_results):
        self.planning_messages = list(planning_messages)
        self.synthesis_results = list(synthesis_results)
        self.tool_requests = []
        self.model_requests = []

    async def invoke_with_tools(
        self,
        request: ToolCallingModelRequest,
    ) -> ToolCallingModelResult:
        self.tool_requests.append(request)
        message = self.planning_messages.pop(0)
        return ToolCallingModelResult(
            message=message,
            model_name=self.model_name,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.model_requests.append(request)
        result = self.synthesis_results.pop(0)
        return ModelResult(
            text=result.get("text", ""),
            parsed=result.get("parsed"),
            model_name=self.model_name,
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    def stream(self, request):
        async def iterator():
            if False:
                yield None

        return iterator()


class FakeEmbeddingModel:
    model_name = "test/embedding"

    async def embed(self, request):
        return None

    async def embed_many(self, requests):
        return []


class FakeWikiVectorEngine:
    mode = "wiki_vector_search"
    required_assets = frozenset({SearchAssetRef(kind="wiki_chunk_vector_index")})

    async def query(self, request, context):
        result = QueryResult(
            id="wiki_chunk_plant",
            text=(
                "Page: Plant Cell Metabolism\n"
                "Summary: Plant cells convert light energy.\n"
                "Section: Photosynthesis\n\n"
                "Chloroplasts produce ATP and NADPH during light reactions."
            ),
            score=0.93,
            source={
                "object_key": "wiki/pages/1-plant-cell-metabolism.md",
                "object_name": "1-plant-cell-metabolism.md",
                "object_type": "wiki",
            },
            metadata={
                "source_key": "wiki/pages/1-plant-cell-metabolism.md",
                "heading_path": "Photosynthesis",
            },
        )
        return QueryResponse(
            mode=self.mode,
            results=(result,),
            citations=(
                QueryCitation(
                    id="citation_search_plant",
                    result_id=result.id,
                    source=result.source,
                    text=result.text,
                ),
            ),
        )


class FakeWikiFullTextEngine:
    mode = "wiki_full_text_search"
    required_assets = frozenset({SearchAssetRef(kind="wiki_chunk_full_text_index")})

    async def query(self, request, context):
        result = QueryResult(
            id="wiki_chunk_plant",
            text=(
                "Page: Plant Cell Metabolism\n"
                "Summary: Plant cells convert light energy.\n"
                "Section: Photosynthesis\n\n"
                "Chloroplasts produce ATP and NADPH during light reactions."
            ),
            score=4.2,
            source={
                "object_key": "wiki/pages/1-plant-cell-metabolism.md",
                "object_name": "1-plant-cell-metabolism.md",
                "object_type": "wiki",
            },
            metadata={
                "source_key": "wiki/pages/1-plant-cell-metabolism.md",
                "heading_path": "Photosynthesis",
            },
        )
        return QueryResponse(mode=self.mode, results=(result,))


class ContextOnlyTool:
    name = "inspect_context"
    description = "Inspect context that cannot be cited."
    parameters_schema = {"type": "object", "properties": {}}
    required_assets = frozenset()
    required_components = frozenset()

    async def run(self, arguments, context):
        return QueryToolResult(
            content="Unverified raw context.",
            metadata={"citable": False},
        )


class CountingTool:
    name = "search_counted"
    description = "Return one counted evidence item."
    parameters_schema = {"type": "object", "properties": {}}
    required_assets = frozenset()
    required_components = frozenset()

    def __init__(self):
        self.calls = 0

    async def run(self, arguments, context):
        self.calls += 1
        result = QueryResult(id=f"result_{self.calls}", text="Counted evidence.")
        return QueryToolResult(
            content="Counted evidence.",
            results=(result,),
            citations=(
                QueryCitation(
                    id=f"citation_{self.calls}",
                    result_id=result.id,
                    text=result.text,
                ),
            ),
            metadata={"citable": True},
        )


class RequiredQueryTool:
    name = "search_required"
    description = "Search using a required string argument."
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
        "additionalProperties": False,
    }
    required_assets = frozenset()
    required_components = frozenset()

    def __init__(self):
        self.calls = 0

    async def run(self, arguments, context):
        self.calls += 1
        return QueryToolResult(content="Should not execute.", metadata={"citable": False})


async def _context(*, model, engine, object_store=None, assets=(), other_engines=()):
    recipe = KnowledgeRecipe(
        models=KnowledgeModels(language=model, embedding=object()),
        stores=(
            KnowledgeStores(objects=object_store)
            if object_store is not None
            else KnowledgeStores()
        ),
    )
    build_result = await KnowledgeBaseBuilder().build(recipe)
    return QueryContext(
        recipe=recipe,
        run_record=build_result.record,
        assets=SearchAssetCollection(assets),
        engines=QueryEngineRegistry((*other_engines, engine)),
    )


def test_agentic_query_searches_reads_and_synthesizes_grounded_answer(tmp_path):
    model = ScriptedToolCallingModel(
        planning_messages=(
            ToolMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call_search",
                        name="search_wiki",
                        arguments={"query": "plant cell light reactions", "top_k": 3},
                    ),
                ),
            ),
            ToolMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call_page",
                        name="read_wiki_page",
                        arguments={"path": "pages/1-plant-cell-metabolism.md"},
                    ),
                ),
            ),
            ToolMessage(role="assistant", content="Evidence gathering is complete."),
        ),
        synthesis_results=(
            {
                "parsed": {
                    "answer": "Chloroplast light reactions produce ATP and NADPH.",
                    "insights": [
                        {
                            "text": "Chloroplast light reactions produce ATP and NADPH.",
                            "evidence_ids": [
                                "evidence_001_search_wiki",
                                "evidence_002_read_wiki_page",
                            ],
                        }
                    ],
                }
            },
        ),
    )
    engine = AgenticQueryEngine()
    object_store = LocalObjectStore(tmp_path)

    async def run():
        await object_store.put(
            "wiki/pages/1-plant-cell-metabolism.md",
            (
                b"# Plant Cell Metabolism\n\n"
                b"## Summary\n\nPlant cells convert light energy.\n\n"
                b"## Content\n\nChloroplasts produce ATP and NADPH."
            ),
        )
        context = await _context(
            model=model,
            engine=engine,
            object_store=object_store,
            assets=(
                SearchAsset(
                    kind="wiki_chunk_vector_index",
                    name="wiki_chunks",
                    store="vector",
                ),
                SearchAsset(
                    kind="wiki_chunk_full_text_index",
                    name="wiki_chunk_full_text",
                    store="text_index",
                ),
            ),
            other_engines=(
                FakeWikiVectorEngine(),
                FakeWikiFullTextEngine(),
                WikiHybridSearchEngine(),
            ),
        )
        return await context.query(
            "wiki_agent_search",
            QueryRequest(text="What do chloroplast light reactions produce?", trace=True),
        )

    response = asyncio.run(run())

    assert response.answer == "Chloroplast light reactions produce ATP and NADPH."
    assert response.insights[0].text == (
        "Chloroplast light reactions produce ATP and NADPH."
    )
    assert [result.id for result in response.results] == [
        "wiki_chunk_plant",
        "wiki/pages/1-plant-cell-metabolism.md",
    ]
    assert [citation.id for citation in response.citations] == [
        "citation_1",
        "citation_wiki_pages_1_plant_cell_metabolism_md",
    ]
    assert response.results[0].metadata["retrieval_modes"] == (
        "wiki_vector_search",
        "wiki_full_text_search",
    )
    assert response.metadata["termination_reason"] == "model_completed"
    assert response.metadata["model_steps"] == 3
    assert response.metadata["tool_calls"] == 2
    assert response.metadata["token_usage"] == {
        "prompt_tokens": 50,
        "completion_tokens": 11,
        "total_tokens": 61,
    }
    assert [message.role for message in model.tool_requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert len(response.trace) == 4


def test_agentic_query_retries_when_model_selects_context_only_evidence():
    model = ScriptedToolCallingModel(
        planning_messages=(
            ToolMessage(
                role="assistant",
                tool_calls=(ToolCall(id="call_raw", name="inspect_context"),),
            ),
            ToolMessage(role="assistant", content="No citable evidence is available."),
        ),
        synthesis_results=(
            {
                "parsed": {
                    "answer": "Unsupported answer.",
                    "insights": [
                        {
                            "text": "Unsupported claim.",
                            "evidence_ids": ["evidence_001_inspect_context"],
                        }
                    ],
                }
            },
            {
                "parsed": {
                    "answer": "The available evidence is insufficient.",
                    "insights": [],
                }
            },
        ),
    )
    engine = AgenticQueryEngine(tools=(ContextOnlyTool(),))

    async def run():
        context = await _context(model=model, engine=engine)
        return await context.query(engine.mode, QueryRequest(text="What happened?"))

    response = asyncio.run(run())

    assert response.answer == "The available evidence is insufficient."
    assert response.results == ()
    assert response.citations == ()
    assert len(model.model_requests) == 2
    assert "non-citable evidence ids" in (model.model_requests[1].prompt or "")
    assert response.metadata["issues"] == ()


def test_agentic_query_enforces_tool_call_budget_before_extra_execution():
    tool = CountingTool()
    model = ScriptedToolCallingModel(
        planning_messages=(
            ToolMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(id="call_1", name=tool.name),
                    ToolCall(id="call_2", name=tool.name),
                ),
            ),
        ),
        synthesis_results=(
            {
                "parsed": {
                    "answer": "One supported fact.",
                    "insights": [
                        {
                            "text": "One supported fact.",
                            "evidence_ids": ["evidence_001_search_counted"],
                        }
                    ],
                }
            },
        ),
    )
    engine = AgenticQueryEngine(
        tools=(tool,),
        tool_budget=QueryToolBudget(max_tool_calls=1),
    )

    async def run():
        context = await _context(model=model, engine=engine)
        return await context.query(engine.mode, QueryRequest(text="Count evidence."))

    response = asyncio.run(run())

    assert tool.calls == 1
    assert response.answer == "One supported fact."
    assert response.metadata["tool_calls"] == 2
    assert response.metadata["executed_tool_calls"] == 1
    assert response.metadata["termination_reason"] == "budget_exhausted"
    assert response.metadata["issues"][0]["code"] == "tool_call_budget_exhausted"


def test_agentic_query_returns_tool_errors_to_model_without_crashing():
    model = ScriptedToolCallingModel(
        planning_messages=(
            ToolMessage(
                role="assistant",
                tool_calls=(ToolCall(id="call_missing", name="missing_tool"),),
            ),
            ToolMessage(role="assistant", content="No evidence was found."),
        ),
        synthesis_results=(
            {
                "parsed": {
                    "answer": "The available evidence is insufficient.",
                    "insights": [],
                }
            },
        ),
    )
    engine = AgenticQueryEngine(tools=(ContextOnlyTool(),))

    async def run():
        context = await _context(model=model, engine=engine)
        return await context.query(engine.mode, QueryRequest(text="Find missing evidence."))

    response = asyncio.run(run())

    tool_message = model.tool_requests[1].messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.content == "error: query tool is not registered: missing_tool"
    assert response.metadata["issues"][0]["code"] == "tool_call_failed"
    assert response.metadata["executed_tool_calls"] == 0


def test_agentic_query_validates_arguments_before_tool_execution():
    tool = RequiredQueryTool()
    model = ScriptedToolCallingModel(
        planning_messages=(
            ToolMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call_invalid",
                        name=tool.name,
                        arguments={"query": 42},
                    ),
                ),
            ),
            ToolMessage(role="assistant", content="The tool arguments were invalid."),
        ),
        synthesis_results=(
            {
                "parsed": {
                    "answer": "The available evidence is insufficient.",
                    "insights": [],
                }
            },
        ),
    )
    engine = AgenticQueryEngine(tools=(tool,))

    async def run():
        context = await _context(model=model, engine=engine)
        return await context.query(engine.mode, QueryRequest(text="Run invalid search."))

    response = asyncio.run(run())

    assert tool.calls == 0
    assert response.metadata["executed_tool_calls"] == 0
    assert "argument query must be string" in (model.tool_requests[1].messages[-1].content or "")


def test_agentic_query_falls_back_after_invalid_synthesis_outputs():
    model = ScriptedToolCallingModel(
        planning_messages=(ToolMessage(role="assistant", content="No evidence."),),
        synthesis_results=(
            {"text": "not json"},
            {"parsed": {"answer": "Still invalid"}},
        ),
    )
    engine = AgenticQueryEngine(tools=(ContextOnlyTool(),))

    async def run():
        context = await _context(model=model, engine=engine)
        return await context.query(engine.mode, QueryRequest(text="Unknown question."))

    response = asyncio.run(run())

    assert response.answer == (
        "The available evidence is insufficient to produce a grounded answer."
    )
    assert response.metadata["answer_generation"] == "fallback"
    assert response.metadata["issues"][0]["code"] == "invalid_grounded_answer"


def test_default_registry_includes_agentic_wiki_mode():
    registry = QueryEngineRegistry.defaults()
    engine = registry.get("wiki_agent_search")

    assert isinstance(engine, AgenticQueryEngine)
    assert engine.required_assets == frozenset(
        {
            SearchAssetRef(kind="wiki_chunk_vector_index"),
            SearchAssetRef(kind="wiki_chunk_full_text_index"),
        }
    )
    assert {ref.key for ref in engine.required_components} == {
        "models.embedding",
        "models.language",
        "stores.objects",
    }


def test_agentic_mode_is_discoverable_only_for_tool_calling_models(tmp_path):
    registry = QueryEngineRegistry.defaults()
    assets = SearchAssetCollection(
        (
            SearchAsset(
                kind="wiki_chunk_vector_index",
                name="wiki_chunks",
                store="vector",
            ),
            SearchAsset(
                kind="wiki_chunk_full_text_index",
                name="wiki_chunk_full_text",
                store="text_index",
            ),
        )
    )

    plain_recipe = KnowledgeRecipe(
        models=KnowledgeModels(language=object(), embedding=object()),
        stores=KnowledgeStores(objects=LocalObjectStore(tmp_path / "plain")),
    )
    tool_recipe = KnowledgeRecipe(
        models=KnowledgeModels(
            language=ScriptedToolCallingModel(
                planning_messages=(),
                synthesis_results=(),
            ),
            embedding=FakeEmbeddingModel(),
        ),
        stores=KnowledgeStores(objects=LocalObjectStore(tmp_path / "tools")),
    )

    assert "wiki_agent_search" not in registry.available_modes_for(plain_recipe, assets)
    assert "wiki_agent_search" in registry.available_modes_for(tool_recipe, assets)
