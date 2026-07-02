import asyncio
import json
import os
import shutil
from pathlib import Path

from heta_framework.common.models import EmbeddingModel
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore
from heta_framework.evaluation import (
    BenchmarkManifest,
    BenchmarkRunConfig,
    BenchmarkRunner,
    BenchmarkWorkspace,
    EvidenceRecallAtK,
    JsonlBenchmark,
)
from heta_framework.kb import (
    DocumentParserRegistry,
    EmbedChunks,
    IndexVectors,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    ParseDocuments,
    SplitDocuments,
    SplitDocumentsConfig,
    TextParser,
)


async def main() -> None:
    # 0. Prepare a minimal benchmark workspace.
    workspace = Path("heta-demo-benchmark")
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)

    # 1. Benchmark data: JsonlBenchmark needs documents.jsonl and cases.jsonl.
    documents_jsonl = workspace / "documents.jsonl"
    cases_jsonl = workspace / "cases.jsonl"
    documents_jsonl.write_text(
        json.dumps(
            {
                "document_id": "doc_heta",
                "name": "heta.txt",
                "media_type": "text/plain",
                "text": (
                    "Heta builds KnowledgeBase objects from Recipe definitions. "
                    "Vector search retrieves chunks by semantic similarity."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cases_jsonl.write_text(
        json.dumps(
            {
                "case_id": "case_vector",
                "query": "What retrieves chunks by semantic similarity?",
                "expected": {
                    "evidence": [
                        {"text": "Vector search retrieves chunks by semantic similarity."}
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # 2. Stores + model: BenchmarkRunner builds the KB from the same Recipe, then runs queries.
    objects = LocalObjectStore(workspace / "objects")
    vectors = InMemoryVectorStore()
    embedding = EmbeddingModel(
        model_name=os.getenv("HETA_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # 3. Recipe: this benchmark evaluates a minimal vector_search KB.
    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
        models=KnowledgeModels(embedding=embedding),
        stores=KnowledgeStores(objects=objects, vector=vectors),
        steps=(
            ParseDocuments(),
            SplitDocuments(SplitDocumentsConfig(encoding_name="unicode")),
            EmbedChunks(),
            IndexVectors(),
        ),
    )

    # 4. Benchmark: use evidence_recall@1 to check whether top-1 evidence matches.
    benchmark = JsonlBenchmark(
        manifest=BenchmarkManifest(
            name="home_jsonl",
            version="v1",
            split="demo",
            task_type="retrieval",
        ),
        documents_jsonl=documents_jsonl,
        cases_jsonl=cases_jsonl,
        evaluator_list=(EvidenceRecallAtK(k=1),),
    )
    result = await BenchmarkRunner().run(
        benchmark=benchmark,
        recipe=recipe,
        knowledge_base_name="home-benchmark",
        query_modes=("vector_search",),
        workspace=BenchmarkWorkspace(root_dir=workspace, cache_dir=workspace / "cache"),
        config=BenchmarkRunConfig(top_k=1, report_id="home_demo"),
    )
    _raise_if_benchmark_failed(result)

    print(result.report.score_summary)
    print(result.report_key)

    await embedding.aclose()
    await vectors.aclose()
    await objects.aclose()


def _raise_if_benchmark_failed(result) -> None:
    failed_build = next(
        (kb for kb in result.knowledge_bases if kb.run_record.status != "succeeded"),
        None,
    )
    if failed_build is not None:
        failed_step = next(
            (
                record
                for record in reversed(failed_build.run_record.step_records)
                if record.status == "failed"
            ),
            None,
        )
        if failed_step is None:
            raise RuntimeError(f"knowledge base build failed: {failed_build.run_record.status}")
        raise RuntimeError(f"{failed_step.step_name} failed: {failed_step.error}")

    failed_case = next((case for case in result.report.case_results if case.error), None)
    if failed_case is not None:
        raise RuntimeError(f"{failed_case.case_id} failed: {failed_case.error.message}")


asyncio.run(main())
