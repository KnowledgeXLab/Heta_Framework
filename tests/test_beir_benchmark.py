import asyncio
import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.evaluation import (  # noqa: E402
    BEIR_DATASET_BASE_URL,
    BEIR_RECOMMENDED_DATASETS,
    BeirBenchmark,
    BeirRetrievalMetric,
    BenchmarkCase,
    BenchmarkEvidence,
    BenchmarkExpected,
    BenchmarkRunUnit,
    BenchmarkWorkspace,
)
from heta_framework.evaluation.benchmarks import beir as beir_module  # noqa: E402
from heta_framework.kb import QueryResponse, QueryResult  # noqa: E402


def test_beir_benchmark_maps_documents_cases_and_default_metrics(tmp_path: Path):
    data_root = _write_beir_fixture(tmp_path / "scifact")
    benchmark = BeirBenchmark(dataset="scifact", data_root=data_root, download=False)

    async def run():
        prepared = await benchmark.prepare(
            BenchmarkWorkspace(root_dir=tmp_path, cache_dir=tmp_path / "cache")
        )
        documents = [document async for document in benchmark.documents(prepared)]
        cases = [case async for case in benchmark.cases(prepared)]
        run_units = [unit async for unit in benchmark.run_units(prepared)]

        assert benchmark.manifest.name == "beir_scifact"
        assert benchmark.manifest.split == "test"
        assert benchmark.manifest.metadata["recommended"] is True
        assert BEIR_RECOMMENDED_DATASETS == ("scifact", "nfcorpus", "fiqa", "hotpotqa")
        assert benchmark.resources()[0].uri == f"{BEIR_DATASET_BASE_URL}/scifact.zip"

        assert len(documents) == 3
        assert documents[0].document_id == "D1"
        assert documents[0].name == "D1.txt"
        assert documents[0].media_type == "text/plain"
        assert documents[0].data == b"Paper One\n\nalpha beta evidence"
        assert documents[0].raw_key(benchmark.manifest) == (
            "raw/benchmarks/beir_scifact/test/D1/D1.txt"
        )

        assert len(cases) == 1
        case = cases[0]
        assert case.case_id == "Q1"
        assert case.query == "which paper discusses alpha?"
        assert tuple(evidence.reference_id for evidence in case.expected.evidence) == (
            "D1",
            "D2",
        )
        assert case.expected.evidence[0].locator == {
            "source_key_prefix": "raw/benchmarks/beir_scifact/test/D1/"
        }
        assert case.expected.evidence[0].metadata["relevance"] == 2
        assert case.expected.metadata["positive_qrels"] == 2
        assert len(benchmark.evaluators()) == 25
        assert run_units == [
            BenchmarkRunUnit(
                unit_id="corpus",
                metadata={"dataset": "scifact", "split": "test"},
            )
        ]

    asyncio.run(run())


def test_beir_retrieval_metrics_deduplicate_chunk_hits_by_document():
    case = _case()
    response = QueryResponse(
        mode="vector_search",
        results=(
            _result("hit_0", "D3"),
            _result("hit_1", "D1"),
            _result("hit_2", "D1"),
            _result("hit_3", "D2"),
        ),
    )

    async def run():
        assert (
            await BeirRetrievalMetric(metric="recall", k=3).evaluate(
                case=case,
                response=response,
            )
        ).value == 1.0
        assert (
            await BeirRetrievalMetric(metric="precision", k=3).evaluate(
                case=case,
                response=response,
            )
        ).value == pytest.approx(2 / 3)
        assert (
            await BeirRetrievalMetric(metric="mrr", k=3).evaluate(
                case=case,
                response=response,
            )
        ).value == 0.5
        assert (
            await BeirRetrievalMetric(metric="map", k=3).evaluate(
                case=case,
                response=response,
            )
        ).value == pytest.approx((1 / 2 + 2 / 3) / 2)
        assert (
            await BeirRetrievalMetric(metric="ndcg", k=3).evaluate(
                case=case,
                response=response,
            )
        ).value == pytest.approx(0.6590018048)

    asyncio.run(run())


def test_beir_download_extracts_official_zip_layout(tmp_path: Path, monkeypatch):
    archive_bytes = _beir_zip_bytes()

    def fake_read_url(url: str) -> bytes:
        assert url == f"{BEIR_DATASET_BASE_URL}/scifact.zip"
        return archive_bytes

    monkeypatch.setattr(beir_module, "_read_url", fake_read_url)
    benchmark = BeirBenchmark(dataset="scifact")

    async def run():
        prepared = await benchmark.prepare(
            BenchmarkWorkspace(root_dir=tmp_path, cache_dir=tmp_path / "cache")
        )
        assert prepared.root_dir == tmp_path / "cache" / "beir_scifact" / "scifact"
        documents = [document async for document in benchmark.documents(prepared)]
        assert len(documents) == 1
        assert documents[0].document_id == "D1"

    asyncio.run(run())


def test_beir_zip_rejects_unsafe_member_paths(tmp_path: Path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("../escape.jsonl", "{}")

    try:
        beir_module._extract_zip(archive_path, tmp_path / "out")
    except ValueError as exc:
        assert "unsafe zip member path" in str(exc)
    else:
        raise AssertionError("expected unsafe zip member path error")


def _write_beir_fixture(root: Path) -> Path:
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(
            (
                '{"_id":"D1","title":"Paper One","text":"alpha beta evidence"}',
                '{"_id":"D2","title":"Paper Two","text":"supporting alpha context"}',
                '{"_id":"D3","title":"Paper Three","text":"unrelated text"}',
            )
        ),
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        '{"_id":"Q1","text":"which paper discusses alpha?"}\n',
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "\n".join(
            (
                "query-id\tcorpus-id\tscore",
                "Q1\tD1\t2",
                "Q1\tD2\t1",
                "Q1\tD3\t0",
            )
        ),
        encoding="utf-8",
    )
    return root


def _beir_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(
            "scifact/corpus.jsonl",
            '{"_id":"D1","title":"Paper One","text":"alpha beta evidence"}\n',
        )
        archive.writestr("scifact/queries.jsonl", '{"_id":"Q1","text":"alpha?"}\n')
        archive.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\nQ1\tD1\t1\n")
    return buffer.getvalue()


def _case():
    return BenchmarkCase(
        case_id="Q1",
        query="which paper discusses alpha?",
        expected=BenchmarkExpected(
            evidence=(
                BenchmarkEvidence(
                    reference_id="D1",
                    metadata={"relevance": 2},
                ),
                BenchmarkEvidence(
                    reference_id="D2",
                    metadata={"relevance": 1},
                ),
            ),
        ),
    )


def _result(result_id: str, document_id: str) -> QueryResult:
    return QueryResult(
        id=result_id,
        text=f"text from {document_id}",
        source={
            "object_key": f"raw/benchmarks/beir_scifact/test/{document_id}/{document_id}.txt",
        },
    )
