import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.evaluation import (  # noqa: E402
    AnswerContains,
    BenchmarkCase,
    BenchmarkDocument,
    BenchmarkEvaluatorProtocol,
    BenchmarkEvidence,
    BenchmarkExpected,
    BenchmarkManifest,
    BenchmarkProtocol,
    BenchmarkRunConfig,
    BenchmarkRunUnit,
    BenchmarkRunner,
    BenchmarkWorkspace,
    EvidenceRecallAtK,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationScore,
    JsonlBenchmark,
    PreparedBenchmark,
    default_report_key,
)
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb import (  # noqa: E402
    KnowledgeRecipe,
    KnowledgeStores,
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    QueryResponse,
    QueryResult,
)


class FakeEvaluator:
    name = "evidence_recall@1"

    async def evaluate(self, *, case, response):
        return EvaluationScore(
            name=self.name,
            value=1.0 if response.results else 0.0,
            metadata={"case_id": case.case_id},
        )


class FakeBenchmark:
    @property
    def manifest(self):
        return BenchmarkManifest(
            name="fake_benchmark",
            version="v1",
            split="test",
            task_type="rag_qa",
        )

    def resources(self):
        return ()

    async def prepare(self, workspace):
        return PreparedBenchmark(manifest=self.manifest, root_dir=workspace.cache_dir)

    async def documents(self, prepared):
        yield BenchmarkDocument(
            document_id="doc_1",
            name="paper.txt",
            media_type="text/plain",
            data=b"Marine biodiversity.",
        )

    async def cases(self, prepared):
        yield BenchmarkCase(
            case_id="case_1",
            query="What is discussed?",
            expected=BenchmarkExpected(
                answers=("Marine biodiversity",),
                evidence=(
                    BenchmarkEvidence(
                        reference_id="doc_1_p0",
                        locator={"document_id": "doc_1", "page_index": 0},
                        text="Marine biodiversity.",
                    ),
                ),
            ),
        )

    async def run_units(self, prepared):
        yield BenchmarkRunUnit(unit_id="corpus")

    def evaluators(self):
        return (FakeEvaluator(),)


class MultiUnitBenchmark(FakeBenchmark):
    @property
    def manifest(self):
        return BenchmarkManifest(
            name="multi_unit",
            version="v1",
            split="test",
            task_type="rag_qa",
            build_scope="case",
        )

    async def documents(self, prepared):
        yield BenchmarkDocument(
            document_id="doc_1",
            name="doc_1.txt",
            media_type="text/plain",
            data=b"Marine biodiversity.",
        )
        yield BenchmarkDocument(
            document_id="doc_2",
            name="doc_2.txt",
            media_type="text/plain",
            data=b"Coral reef monitoring.",
        )

    async def cases(self, prepared):
        yield BenchmarkCase(case_id="case_1", query="What is in document one?")
        yield BenchmarkCase(case_id="case_2", query="What is in document two?")

    async def run_units(self, prepared):
        yield BenchmarkRunUnit(
            unit_id="doc_1",
            document_ids=("doc_1",),
            case_ids=("case_1",),
        )
        yield BenchmarkRunUnit(
            unit_id="doc_2",
            document_ids=("doc_2",),
            case_ids=("case_2",),
        )


def test_benchmark_protocol_accepts_custom_benchmark(tmp_path):
    async def run():
        benchmark = FakeBenchmark()
        assert isinstance(benchmark, BenchmarkProtocol)
        assert isinstance(benchmark.evaluators()[0], BenchmarkEvaluatorProtocol)

        prepared = await benchmark.prepare(
            BenchmarkWorkspace(
                root_dir=tmp_path,
                cache_dir=tmp_path / "cache",
            )
        )
        documents = [document async for document in benchmark.documents(prepared)]
        cases = [case async for case in benchmark.cases(prepared)]

        assert documents[0].raw_key(benchmark.manifest) == (
            "raw/benchmarks/fake_benchmark/test/doc_1/paper.txt"
        )
        assert cases[0].expected.answers == ("Marine biodiversity",)
        assert benchmark.manifest.to_dict()["build_scope"] == "corpus"
        run_units = [unit async for unit in benchmark.run_units(prepared)]
        assert run_units == [BenchmarkRunUnit(unit_id="corpus")]

    asyncio.run(run())


def test_evaluator_scores_query_response():
    async def run():
        score = await FakeEvaluator().evaluate(
            case=BenchmarkCase(case_id="case_1", query="What?"),
            response=QueryResponse(
                mode="vector_search",
                results=(QueryResult(id="chunk_1", text="Marine biodiversity."),),
            ),
        )

        assert score == EvaluationScore(
            name="evidence_recall@1",
            value=1.0,
            metadata={"case_id": "case_1"},
        )

    asyncio.run(run())


def test_evaluation_report_is_json_friendly():
    manifest = BenchmarkManifest(
        name="fake_benchmark",
        version="v1",
        split="test",
        task_type="rag_qa",
    )
    response = QueryResponse(
        mode="vector_search",
        results=(
            QueryResult(
                id="chunk_1",
                text="Marine biodiversity.",
                score=0.9,
                source={"document_id": "doc_1", "page_index": 0},
            ),
        ),
        answer="Marine biodiversity is discussed.",
    )
    report = EvaluationReport(
        report_id="eval_1",
        benchmark=manifest,
        knowledge_base_name="Marine KB",
        knowledge_base_manifest={"name": "Marine KB"},
        recipe_manifest={"steps": []},
        query_modes=("vector_search",),
        score_summary={"evidence_recall@1": 1.0},
        case_results=(
            EvaluationCaseResult(
                case_id="case_1",
                query="What is discussed?",
                query_mode="vector_search",
                response=response,
                scores=(EvaluationScore(name="evidence_recall@1", value=1.0),),
                latency_ms=12.5,
            ),
        ),
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        report_key=default_report_key("Marine KB", "eval_1"),
    )

    payload = report.to_dict()

    assert payload["report_key"] == (
        "_heta/knowledge_bases/marine_kb/evaluations/eval_1/report.json"
    )
    assert payload["case_results"][0]["response"]["results"][0]["source"] == {
        "document_id": "doc_1",
        "page_index": 0,
    }


def test_benchmark_document_requires_one_location(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        BenchmarkDocument(
            document_id="doc",
            name="doc.txt",
            media_type="text/plain",
            data=b"a",
            path=tmp_path / "doc.txt",
        )

    with pytest.raises(ValueError, match="exactly one"):
        BenchmarkDocument(document_id="doc", name="doc.txt", media_type="text/plain")


def test_benchmark_evidence_requires_matching_signal():
    with pytest.raises(ValueError, match="evidence must set"):
        BenchmarkEvidence()


def test_document_name_must_not_be_path():
    with pytest.raises(ValueError, match="not a path"):
        BenchmarkDocument(
            document_id="doc",
            name="nested/doc.txt",
            media_type="text/plain",
            data=b"text",
        )


class FakeRunnerQueryEngine:
    mode = "fake_search"
    required_assets = frozenset()

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        return QueryResponse(
            mode=self.mode,
            answer="Marine biodiversity is discussed.",
            results=(
                QueryResult(
                    id="chunk_1",
                    text="Marine biodiversity is discussed in the benchmark document.",
                    score=1.0,
                    source={
                        "document_id": "doc_1",
                        "object_key": "raw/benchmarks/fake_benchmark/test/doc_1/doc.txt",
                        "page_index": 0,
                    },
                ),
            ),
        )


def test_benchmark_runner_builds_queries_scores_and_persists_report(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path / "objects")
        recipe = KnowledgeRecipe(stores=KnowledgeStores(objects=object_store))
        result = await BenchmarkRunner().run(
            benchmark=FakeBenchmark(),
            recipe=recipe,
            knowledge_base_name="fake-eval-kb",
            query_modes=("fake_search",),
            workspace=BenchmarkWorkspace(
                root_dir=tmp_path / "workspace",
                cache_dir=tmp_path / "workspace" / "cache",
            ),
            query_engines=QueryEngineRegistry([FakeRunnerQueryEngine()]),
            config=BenchmarkRunConfig(
                report_id="eval_1",
                top_k=3,
                max_concurrent_queries=2,
            ),
        )

        assert len(result.knowledge_bases) == 1
        assert result.knowledge_bases[0].run_record.status == "succeeded"
        assert result.benchmark_document_keys == (
            "raw/benchmarks/fake_benchmark/test/doc_1/paper.txt",
        )
        assert await object_store.exists(result.benchmark_document_keys[0])
        assert result.report.score_summary == {
            "fake_search.evidence_recall@1": 1.0,
        }
        assert result.report.knowledge_base_name == "fake-eval-kb"
        assert result.report.metadata["run_units"] == [BenchmarkRunUnit(unit_id="corpus").to_dict()]
        assert result.report_key == (
            "_heta/knowledge_bases/fake-eval-kb/evaluations/eval_1/report.json"
        )
        assert await object_store.exists(result.report_key)

        await object_store.aclose()

    asyncio.run(run())


def test_benchmark_runner_builds_multiple_run_units(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path / "objects")
        recipe = KnowledgeRecipe(stores=KnowledgeStores(objects=object_store))
        result = await BenchmarkRunner().run(
            benchmark=MultiUnitBenchmark(),
            recipe=recipe,
            knowledge_base_name="multi-eval-kb",
            query_modes=("fake_search",),
            workspace=BenchmarkWorkspace(
                root_dir=tmp_path / "workspace",
                cache_dir=tmp_path / "workspace" / "cache",
            ),
            query_engines=QueryEngineRegistry([FakeRunnerQueryEngine()]),
            config=BenchmarkRunConfig(report_id="eval_multi"),
        )

        assert [kb.name for kb in result.knowledge_bases] == [
            "multi-eval-kb-doc_1",
            "multi-eval-kb-doc_2",
        ]
        assert result.benchmark_document_keys == (
            "raw/benchmarks/multi_unit/test/doc_1/doc_1.txt",
            "raw/benchmarks/multi_unit/test/doc_2/doc_2.txt",
        )
        assert result.report.knowledge_base_name == "multi-eval-kb"
        assert len(result.report.case_results) == 2
        assert {
            item.metadata["benchmark_run_unit"] for item in result.report.case_results
        } == {"doc_1", "doc_2"}
        assert result.report_key == (
            "_heta/knowledge_bases/multi-eval-kb/evaluations/eval_multi/report.json"
        )
        await object_store.aclose()

    asyncio.run(run())


def test_jsonl_benchmark_reads_documents_cases_and_default_evaluators(tmp_path):
    documents_path = tmp_path / "documents.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    documents_path.write_text(
        '{"document_id":"doc_1","name":"doc.txt","media_type":"text/plain",'
        '"text":"Marine biodiversity."}\n',
        encoding="utf-8",
    )
    cases_path.write_text(
        '{"case_id":"case_1","query":"What is discussed?",'
        '"expected":{"answers":["Marine biodiversity"],'
        '"evidence":[{"locator":{"document_id":"doc_1"},"text":"Marine biodiversity."}]}}\n',
        encoding="utf-8",
    )
    benchmark = JsonlBenchmark(
        manifest=BenchmarkManifest(
            name="jsonl_benchmark",
            version="v1",
            split="test",
            task_type="rag_qa",
        ),
        documents_jsonl=documents_path,
        cases_jsonl=cases_path,
    )

    async def run():
        prepared = await benchmark.prepare(
            BenchmarkWorkspace(root_dir=tmp_path, cache_dir=tmp_path / "cache")
        )
        documents = [document async for document in benchmark.documents(prepared)]
        cases = [case async for case in benchmark.cases(prepared)]

        assert documents[0].data == b"Marine biodiversity."
        assert cases[0].expected.answers == ("Marine biodiversity",)
        assert tuple(type(evaluator) for evaluator in benchmark.evaluators()) == (
            EvidenceRecallAtK,
            AnswerContains,
        )

    asyncio.run(run())
