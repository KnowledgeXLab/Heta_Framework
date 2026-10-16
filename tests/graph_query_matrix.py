"""Build real minimal KBs and run every built-in query through Graph input preparation.

Run directly with --output to retain KB artifacts and a machine-readable matrix.
Only model boundaries are deterministic fixtures; build steps, stores, query engines
and the Graph producer are the production implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from heta_framework.common.models import (
    EmbeddingResult,
    ModelResult,
    RerankItem,
    RerankResult,
    ToolCall,
    ToolCallingModelResult,
    ToolMessage,
)
from heta_framework.common.stores import (
    ClusterableInMemoryGraphStore,
    InMemoryTextIndexStore,
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
)
from heta_framework.generation import (
    HetaGenTaskProducerGraph,
    HetaGenTaskProducerGraphConfig,
    TaskRequest,
)
from heta_framework.kb import (
    DocumentParserRegistry,
    GraphRAGProcedure,
    HetaGraphProcedure,
    HetaWikiProcedure,
    HiRAGProcedure,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    LeanRAGProcedure,
    LightRAGProcedure,
    TextParser,
)
from heta_framework.kb.search import QueryEngineRegistry
from heta_framework.kb.steps import (
    EmbedChunks,
    IndexFullText,
    IndexVectors,
    ParseDocuments,
    PersistChunks,
    PersistChunksConfig,
    SplitDocuments,
    SplitDocumentsConfig,
)

ENTITIES = (
    "heat",
    "Euler",
    "CrankNicolson",
    "grid",
    "timestep",
    "diffusivity",
    "temperature",
    "boundary",
    "stability",
    "residual",
)
CORPUS = """# Heat diffusion methods — functional test fixture

This is synthetic smoke-test material, not a published paper.
heat diffusion is computed by Euler on a grid using timestep and diffusivity.
CrankNicolson is compared with Euler for heat diffusion.
temperature is evaluated with boundary constraints and residual measurements.
stability depends on timestep and diffusivity on the grid.
Euler uses grid. grid supports timestep. timestep controls diffusivity.
diffusivity affects temperature. temperature obeys boundary.
boundary controls stability. stability affects residual. residual measures heat.
"""
QUERY = "heat"
TOP_K = 8


class MatrixEmbedding:
    model_name = "fixture/content-derived-embedding"

    async def embed(self, request):
        return EmbeddingResult(
            vectors=[
                [1.0, *(float(name.casefold() in text.casefold()) for name in ENTITIES)]
                for text in request.texts
            ],
            model_name=self.model_name,
        )

    async def embed_many(self, requests):
        return [await self.embed(request) for request in requests]


class MatrixReranker:
    model_name = "fixture/lexical-reranker"

    async def rerank(self, request):
        return RerankResult(
            rankings=[
                RerankItem(index=index, score=float(QUERY in text.casefold()))
                for index, text in enumerate(request.documents)
            ],
            model_name=self.model_name,
        )

    async def rerank_many(self, requests):
        return [await self.rerank(request) for request in requests]


class MatrixLanguage:
    model_name = "fixture/scientific-graph-language"

    def __init__(self):
        self.calls = []
        self.aggregation_index = 0

    async def invoke(self, request):
        trace = dict(request.trace_context or {})
        self.calls.append(trace)
        stage = trace.get("stage", "")
        prompt = request.prompt or ""
        if trace.get("step") == "extract_universal_graph" and stage == "entity_extraction":
            data = {
                "entities": [
                    {
                        "name": name,
                        "type": "DOMAIN_TERM",
                        "subtype": None,
                        "description": f"{name} occurs in the heat diffusion fixture.",
                        "attributes": {},
                    }
                    for name in ENTITIES
                ]
            }
        elif trace.get("step") == "extract_universal_graph" and stage == "relation_extraction":
            data = {
                "relations": [
                    {
                        "source": ENTITIES[i],
                        "target": ENTITIES[(i + 1) % len(ENTITIES)],
                        "type": ("USES", "SUPPORTS", "EVALUATES")[i % 3],
                        "name": "related to",
                        "description": f"{ENTITIES[i]} relates to the next heat fixture concept.",
                        "attributes": {"weight": "1.0"},
                    }
                    for i in range(len(ENTITIES))
                ]
            }
        elif stage == "summary_clusters":
            return ModelResult(
                text=(
                    '("entity"<|>"heat methods"<|>"DOMAIN_TERM"<|>"Heat diffusion methods.")'
                    "<|COMPLETE|>"
                ),
                model_name=self.model_name,
            )
        elif trace.get("step") == "build_wiki_pages" and stage == "summary":
            return ModelResult(
                text="Heat diffusion uses Euler, grids and boundary conditions.",
                model_name=self.model_name,
            )
        elif stage == "community_report":
            data = {
                "title": "Heat diffusion",
                "summary": "Heat methods use grids and boundary constraints.",
                "findings": [
                    {"summary": "Heat", "explanation": "Euler and CrankNicolson compute heat."}
                ],
                "rating": 8,
            }
        elif stage == "aggregate_entities":
            self.aggregation_index += 1
            data = {
                "entity_name": f"Heat aggregate {self.aggregation_index}",
                "entity_description": "Heat methods, grids and stability.",
                "findings": [{"summary": "Heat", "explanation": "Heat fixture concepts."}],
            }
        elif stage == "keyword_extraction":
            data = {"high_level_keywords": ["heat"], "low_level_keywords": ["Euler"]}
        elif stage == "global_community_map":
            data = {"points": [{"description": "Euler computes heat on a grid.", "score": 90}]}
        elif "Generate alternative search queries" in prompt:
            data = {"queries": ["heat", "Euler heat"]}
        elif stage == "extract" and trace.get("query_mode") == "heta_multihop_search":
            data = {"usefulness": True, "information": "Euler computes heat on a grid."}
        elif stage == "judge" and trace.get("query_mode") == "heta_multihop_search":
            data = {"judge": True, "answer": "Euler computes heat on a grid."}
        else:
            raise AssertionError(f"unhandled fixture model request: {trace}")
        return ModelResult(text=json.dumps(data), parsed=data, model_name=self.model_name)

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    async def stream(self, request):
        if False:
            yield None

    async def invoke_with_tools(self, request):
        self.calls.append({"stage": "agentic_plan"})
        observed = sum(message.role == "tool" for message in request.messages)
        if observed == 0:
            message = ToolMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="search-heat",
                        name="search_wiki",
                        arguments={"query": QUERY, "top_k": TOP_K},
                    ),
                ),
            )
        elif observed == 1:
            message = ToolMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="read-heat",
                        name="read_wiki_page",
                        arguments={
                            "path": "pages/1-heat-diffusion-methods-functional-test-fixture.md"
                        },
                    ),
                ),
            )
        else:
            message = ToolMessage(role="assistant", content="Evidence collected.")
        return ToolCallingModelResult(message=message, model_name=self.model_name)


def families():
    common = (ParseDocuments(), SplitDocuments(SplitDocumentsConfig(chunk_size=2048, overlap=0)))
    return {
        "heta": (
            common
            + (
                EmbedChunks(),
                IndexVectors(),
                PersistChunks(PersistChunksConfig(chunk_keys_artifact="chunk_keys")),
                IndexFullText(),
                HetaGraphProcedure(deduplicate=False),
            ),
            {},
        ),
        "graphrag": (
            common + (GraphRAGProcedure(graph_cluster_algorithm="connected_components"),),
            {"relations_artifact": "graph_rag_relation_keys"},
        ),
        "lightrag": (
            common + (LightRAGProcedure(entity_extract_max_gleaning=0),),
            {
                "entities_artifact": "light_rag_entity_keys",
                "relations_artifact": "light_rag_relation_keys",
            },
        ),
        "hirag": (
            (
                HiRAGProcedure(
                    chunk_token_size=2048,
                    chunk_overlap_token_size=0,
                    entity_extract_max_gleaning=0,
                    hierarchical_layers=2,
                    graph_cluster_algorithm="connected_components",
                    clustering_backend="deterministic",
                ),
            ),
            {
                "entities_artifact": "hi_rag_universal_entity_keys",
                "relations_artifact": "hi_rag_universal_relation_keys",
            },
        ),
        "leanrag": (
            (
                LeanRAGProcedure(
                    chunk_token_size=2048,
                    chunk_overlap_token_size=0,
                    entity_extract_max_gleaning=0,
                    aggregation_cluster_size=2,
                    aggregation_clustering_backend="deterministic",
                ),
            ),
            {
                "entities_artifact": "lean_rag_universal_entity_keys",
                "relations_artifact": "lean_rag_universal_relation_keys",
            },
        ),
        "wiki": (common + (HetaGraphProcedure(deduplicate=False), HetaWikiProcedure()), {}),
    }


async def run_matrix(output: Path, *, run_producer=False):
    output.mkdir(parents=True, exist_ok=True)
    registry = QueryEngineRegistry.defaults()
    report = {
        "execution": "real_build_steps_and_query_engines_with_fixture_models",
        "answer_generation": False,
        "run_producer": run_producer,
        "storage": "local_objects_sqlite_and_in_memory_graph_vector_text_indexes",
        "registered_modes": sorted(registry.modes),
        "knowledge_bases": {},
        "queries": [],
    }
    covered = set()
    for family, (steps, bindings) in families().items():
        directory = output / family
        directory.mkdir()
        objects = LocalObjectStore(directory / "objects")
        destination = LocalObjectStore(directory / "outputs")
        sql = SQLStore(f"sqlite:///{directory.resolve() / 'kb.sqlite'}")
        vectors = InMemoryVectorStore()
        graph = ClusterableInMemoryGraphStore()
        text = InMemoryTextIndexStore()
        language = MatrixLanguage()
        for number in (1, 2):
            await objects.put(
                f"raw/heat-{number}.txt",
                (CORPUS + f"\nCase {number}: grid size {number * 16}.\n").encode(),
            )
        recipe = KnowledgeRecipe(
            parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
            models=KnowledgeModels(
                language=language, embedding=MatrixEmbedding(), reranker=MatrixReranker()
            ),
            stores=KnowledgeStores(
                objects=objects, sql=sql, vector=vectors, graph=graph, text_index=text
            ),
            steps=steps,
        )
        try:
            recipe.require_valid()
            kb = await KnowledgeBase.create(recipe=recipe, name=f"query-matrix-{family}")
            if kb.run_record.status != "succeeded":
                raise RuntimeError(
                    str(
                        [
                            (step.step_name, step.error)
                            for step in kb.run_record.step_records
                            if step.status == "failed"
                        ]
                    )
                )
            modes = [
                mode
                for mode in sorted(registry.modes)
                if mode not in covered
                and not kb.search_assets.missing(registry.get(mode).required_assets)
            ]
            report["knowledge_bases"][family] = {
                "status": kb.run_record.status,
                "run_id": kb.run_record.run_id,
                "available_queries": sorted(kb.available_queries),
                "step_count": len(kb.run_record.step_records),
                "artifact_names": sorted(kb.run_record.artifacts),
            }
            for mode in modes:
                entry = {"family": family, "mode": mode}
                started = time.monotonic()
                try:
                    response = await kb.query(
                        QUERY,
                        mode=mode,
                        top_k=TOP_K,
                        options={"generate_answer": False},
                        trace=True,
                    )
                    entry.update(
                        result_count=len(response.results), citation_count=len(response.citations)
                    )
                    assert response.results, "query returned no results for matching source text"
                    (directory / f"{mode}.response.json").write_text(
                        json.dumps(asdict(response), indent=2)
                    )
                    producer = HetaGenTaskProducerGraph(
                        HetaGenTaskProducerGraphConfig(
                            query_mode=mode, top_k=TOP_K, max_documents=2, **bindings
                        )
                    )
                    prepared = await producer.prepare_inputs(
                        TaskRequest(f"prepared/{mode}", {"query": QUERY}),
                        object_store=destination,
                        knowledge_base=kb,
                    )
                    entry.update(
                        document_count=prepared.document_count,
                        inputs_manifest_key=prepared.manifest_key,
                    )
                    assert prepared.document_count > 0, (
                        "query did not resolve to any input documents"
                    )
                    snapshot = json.loads(await destination.get(prepared.manifest_key))
                    assert all(
                        doc["document_id"].startswith("doc_") for doc in snapshot["documents"]
                    )
                    if mode == "wiki_agent_search":
                        assert any(item.kind == "wiki_page" for item in response.results)
                    if run_producer:
                        from dataclasses import replace

                        assert len(ENTITIES) < 20  # Must fail the unchanged native structural gate.
                        fixture_env = directory / "fixture.env"
                        fixture_env.write_text("OPENAI_API_KEY=offline-unused\n")
                        executor = HetaGenTaskProducerGraph(
                            replace(
                                producer.config,
                                model="offline-unused",
                                env_file=fixture_env,
                                api_base_url="http://127.0.0.1:1/v1",
                                max_concurrency=2,
                            )
                        )
                        result = await executor.produce(
                            TaskRequest(f"production/{mode}", {"query": QUERY}),
                            object_store=destination,
                            knowledge_base=kb,
                        )
                        run_report = json.loads(await destination.get(result.report_key))
                        manifest = json.loads(await destination.get(result.tasks_manifest_key))
                        assert result.status == "succeeded" and manifest["tasks"] == [], run_report
                        assert run_report["native"]["stages"]["screen"] == "completed_empty"
                        entry.update(
                            production_status=result.status,
                            tasks_manifest_key=result.tasks_manifest_key,
                            report_key=result.report_key,
                            task_count=0,
                            native_stages=run_report["native"]["stages"],
                        )
                    entry["status"] = "passed"
                    covered.add(mode)
                except Exception as error:
                    entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                entry["elapsed_sec"] = round(time.monotonic() - started, 3)
                report["queries"].append(entry)
                print(
                    f"{entry['status'].upper()} {family}/{mode}: {entry.get('error', '')}",
                    flush=True,
                )
        except Exception as error:
            report["knowledge_bases"][family] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"BUILD FAILED {family}: {error}", flush=True)
        finally:
            for store in (sql, vectors, graph, text, objects, destination):
                await store.aclose()
        report["uncovered_modes"] = sorted(registry.modes - covered)
        report["status"] = (
            "passed"
            if (
                not report["uncovered_modes"]
                and all(kb["status"] == "succeeded" for kb in report["knowledge_bases"].values())
                and all(query["status"] == "passed" for query in report["queries"])
            )
            else "failed"
        )
        (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-producer", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(run_matrix(args.output, run_producer=args.run_producer))
    raise SystemExit(summary["status"] != "passed")
