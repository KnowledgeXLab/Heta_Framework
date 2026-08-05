import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (  # noqa: E402
    InMemoryGraphStore,
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
)
from heta_framework.evaluation import (  # noqa: E402
    AnswerContains,
    BenchmarkCase,
    BenchmarkEvidence,
    BenchmarkExpected,
    BenchmarkRunConfig,
    BenchmarkRunner,
    BenchmarkRunUnit,
    BenchmarkWorkspace,
    EvidenceRecallAtK,
    MultiHopRagBenchmark,
)
from heta_framework.kb import (  # noqa: E402
    DocumentParserRegistry,
    HiRAGProcedure,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    QueryResponse,
    QueryResult,
    TextParser,
)
from test_hirag_procedure_smoke import SmokeEmbeddingModel, SmokeLanguageModel  # noqa: E402


def test_multihop_rag_benchmark_maps_corpus_and_cases(tmp_path: Path):
    corpus_path = tmp_path / "corpus.json"
    queries_path = tmp_path / "MultiHopRAG.json"
    corpus_path.write_text(
        """
        [
          {
            "title": "The FTX trial is bigger than Sam Bankman-Fried",
            "author": "Elizabeth Lopatto",
            "source": "The Verge",
            "published_at": "2023-09-28T12:00:00+00:00",
            "category": "technology",
            "url": "https://www.theverge.com/ftx-trial",
            "body": "Before his fall, Bankman-Fried was a major figure in crypto."
          },
          {
            "title": "SBF trial starts soon",
            "author": "Jacquelyn Melinek",
            "source": "TechCrunch",
            "published_at": "2023-10-01T14:00:29+00:00",
            "category": "technology",
            "url": "https://techcrunch.com/sbf-trial",
            "body": "Sam Bankman-Fried faced fraud and conspiracy charges."
          }
        ]
        """,
        encoding="utf-8",
    )
    queries_path.write_text(
        """
        [
          {
            "query": "Who faced fraud and conspiracy charges?",
            "answer": "Sam Bankman-Fried",
            "question_type": "inference_query",
            "evidence_list": [
              {
                "title": "The FTX trial is bigger than Sam Bankman-Fried",
                "author": "Elizabeth Lopatto",
                "url": "https://www.theverge.com/ftx-trial",
                "source": "The Verge",
                "category": "technology",
                "published_at": "2023-09-28T12:00:00+00:00",
                "fact": "Before his fall, Bankman-Fried was a major figure in crypto."
              },
              {
                "title": "SBF trial starts soon",
                "author": "Jacquelyn Melinek",
                "url": "https://techcrunch.com/sbf-trial",
                "source": "TechCrunch",
                "category": "technology",
                "published_at": "2023-10-01T14:00:29+00:00",
                "fact": "Sam Bankman-Fried faced fraud and conspiracy charges."
              }
            ]
          }
        ]
        """,
        encoding="utf-8",
    )
    benchmark = MultiHopRagBenchmark(
        corpus_path=corpus_path,
        queries_path=queries_path,
    )

    async def run():
        prepared = await benchmark.prepare(
            BenchmarkWorkspace(root_dir=tmp_path, cache_dir=tmp_path / "cache")
        )
        documents = [document async for document in benchmark.documents(prepared)]
        cases = [case async for case in benchmark.cases(prepared)]
        run_units = [unit async for unit in benchmark.run_units(prepared)]

        assert benchmark.manifest.build_scope == "corpus"
        assert run_units == [BenchmarkRunUnit(unit_id="corpus")]
        assert len(documents) == 2
        assert documents[0].media_type == "text/plain"
        assert documents[0].name.endswith(".txt")
        assert "Title: The FTX trial" in documents[0].data.decode("utf-8")

        case = cases[0]
        assert case.case_id == "multihop_rag_0000"
        assert case.expected.answers == ("Sam Bankman-Fried",)
        assert case.labels == {"question_type": "inference_query"}
        assert len(case.expected.evidence) == 2
        assert case.expected.evidence[0].metadata["document_id"] == documents[0].document_id
        assert case.expected.evidence[0].locator["source_key"] == documents[0].raw_key(
            benchmark.manifest
        )
        assert tuple(type(evaluator) for evaluator in benchmark.evaluators()) == (
            EvidenceRecallAtK,
            AnswerContains,
        )
        score = await EvidenceRecallAtK(k=1).evaluate(
            case=case,
            response=QueryResponse(
                mode="vector_search",
                results=(
                    QueryResult(
                        id="hit_1",
                        text="Before his fall, Bankman-Fried was a major figure in crypto.",
                        source={"object_key": documents[0].raw_key(benchmark.manifest)},
                    ),
                ),
            ),
        )
        assert score.value == 0.5

    asyncio.run(run())


def test_multihop_rag_benchmark_declares_official_resources():
    resources = MultiHopRagBenchmark().resources()

    assert [resource.name for resource in resources] == ["corpus", "queries"]
    assert all(resource.kind == "file" for resource in resources)


def test_multihop_rag_evidence_recall_matches_plural_hirag_sources():
    source_key = "raw/benchmarks/multihop_rag/all/article_1/article_1.txt"
    case = _evidence_case(source_key)
    response = QueryResponse(
        mode="hi_rag_query",
        results=(
            QueryResult(
                id="hi_rag_query_context",
                text="Sam Bankman-Fried faced fraud and conspiracy charges.",
                source={
                    "document_ids": ("doc_1",),
                    "object_keys": (source_key,),
                    "source_keys": (source_key,),
                },
            ),
        ),
    )

    score = asyncio.run(EvidenceRecallAtK(k=5).evaluate(case=case, response=response))

    assert score.value == 1.0
    assert score.metadata["matched"] == 1


def test_multihop_rag_benchmark_runs_hirag_end_to_end(tmp_path: Path):
    corpus_path = tmp_path / "corpus.json"
    queries_path = tmp_path / "MultiHopRAG.json"
    corpus_path.write_text(
        """
        [
          {
            "title": "Knowledge graph collaboration",
            "source": "Example News",
            "url": "https://example.com/alice-bob",
            "body": "Alice collaborates with Bob on knowledge graph research at Acme."
          }
        ]
        """,
        encoding="utf-8",
    )
    queries_path.write_text(
        """
        [
          {
            "query": "Who collaborates with Bob?",
            "answer": "Alice",
            "question_type": "inference_query",
            "evidence_list": [
              {
                "title": "Knowledge graph collaboration",
                "source": "Example News",
                "url": "https://example.com/alice-bob",
                "fact": "Alice collaborates with Bob on knowledge graph research at Acme."
              }
            ]
          }
        ]
        """,
        encoding="utf-8",
    )

    async def run():
        object_store = LocalObjectStore(tmp_path / "objects")
        recipe = KnowledgeRecipe(
            parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
            models=KnowledgeModels(
                language=SmokeLanguageModel(),
                embedding=SmokeEmbeddingModel(),
            ),
            stores=KnowledgeStores(
                objects=object_store,
                graph=InMemoryGraphStore(),
                vector=InMemoryVectorStore(),
                sql=SQLStore(f"sqlite:///{tmp_path / 'hirag-multihop.db'}"),
            ),
            steps=(
                HiRAGProcedure(
                    raw_prefix="raw/benchmarks/multihop_rag/all",
                    chunk_token_size=128,
                    chunk_overlap_token_size=16,
                    entity_extract_max_gleaning=0,
                    hierarchical_layers=2,
                    clustering_backend="deterministic",
                ),
            ),
        )
        result = await BenchmarkRunner().run(
            benchmark=MultiHopRagBenchmark(
                corpus_path=corpus_path,
                queries_path=queries_path,
            ),
            recipe=recipe,
            knowledge_base_name="hirag-multihop-smoke",
            query_modes=("hi_rag_query",),
            workspace=BenchmarkWorkspace(
                root_dir=tmp_path / "workspace",
                cache_dir=tmp_path / "workspace" / "cache",
            ),
            evaluators=(EvidenceRecallAtK(k=5),),
            config=BenchmarkRunConfig(
                max_concurrent_queries=1,
                persist_report=False,
                query_options={"generate_answer": False, "top_m": 3},
            ),
        )

        assert result.knowledge_bases[0].run_record.status == "succeeded"
        assert result.report.score_summary == {
            "hi_rag_query.evidence_recall@5": 1.0,
        }
        assert result.report.case_results[0].error is None
        assert result.report.case_results[0].response is not None
        assert result.report.case_results[0].response.results[0].source["source_keys"]

    asyncio.run(run())


def _evidence_case(source_key: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="multihop_rag_0000",
        query="Who faced fraud and conspiracy charges?",
        expected=BenchmarkExpected(
            evidence=(
                BenchmarkEvidence(
                    reference_id="article_1",
                    locator={"source_key": source_key},
                    text="Sam Bankman-Fried faced fraud and conspiracy charges.",
                ),
            ),
        ),
    )
