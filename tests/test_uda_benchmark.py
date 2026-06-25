import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.evaluation import (  # noqa: E402
    AnswerContains,
    BenchmarkWorkspace,
    EvidenceRecallAtK,
    UDA_GITHUB_URL,
    UDA_SOURCE_DOCS_URL,
    BenchmarkRunUnit,
    UdaBenchmark,
)
from heta_framework.evaluation.benchmarks import uda as uda_module  # noqa: E402
from heta_framework.kb import QueryResponse, QueryResult  # noqa: E402


def test_uda_benchmark_maps_fin_subset_documents_and_cases(tmp_path: Path):
    qa_path = tmp_path / "fin_qa.csv"
    extended_path = tmp_path / "bench_fin_qa.json"
    source_root = tmp_path / "src_doc_files"
    source_doc = source_root / "fin_docs" / "ADI_2009.pdf"
    source_doc.parent.mkdir(parents=True)
    source_doc.write_bytes(b"%PDF-1.4\n% test fixture\n")
    qa_path.write_text(
        "\n".join(
            (
                "doc_name|q_uid|question|answer_1|answer_2",
                "ADI_2009|ADI/2009/page_49.pdf-1|what is interest expense?|380|3.8",
            )
        ),
        encoding="utf-8",
    )
    extended_path.write_text(
        json.dumps(
            {
                "ADI_2009": [
                    {
                        "q_uid": "ADI/2009/page_49.pdf-1",
                        "answers": {"str_answer": "380", "exe_answer": 3.8},
                        "evidence": {
                            "table_1": "interest expense was 380",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark = UdaBenchmark(
        subset="fin",
        source_root=source_root,
        qa_path=qa_path,
        extended_info_path=extended_path,
    )

    async def run():
        prepared = await benchmark.prepare(
            BenchmarkWorkspace(root_dir=tmp_path, cache_dir=tmp_path / "cache")
        )
        documents = [document async for document in benchmark.documents(prepared)]
        cases = [case async for case in benchmark.cases(prepared)]
        run_units = [unit async for unit in benchmark.run_units(prepared)]

        assert benchmark.manifest.name == "uda_fin"
        assert benchmark.manifest.build_scope == "case"
        assert benchmark.manifest.homepage == UDA_GITHUB_URL
        assert benchmark.resources()[2].uri == UDA_SOURCE_DOCS_URL
        assert benchmark.resources()[2].metadata["source_archives"] == [
            "https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/main/"
            "src_doc_files/fin_docs.zip?download=true"
        ]
        assert len(documents) == 1
        assert documents[0].document_id == "ADI_2009"
        assert documents[0].name == "ADI_2009.pdf"
        assert documents[0].media_type == "application/pdf"
        assert documents[0].path == source_doc
        assert documents[0].raw_key(benchmark.manifest) == (
            "raw/benchmarks/uda_fin/all/ADI_2009/ADI_2009.pdf"
        )

        case = cases[0]
        assert case.case_id == "ADI/2009/page_49.pdf-1"
        assert case.query == "what is interest expense?"
        assert case.expected.answers == ("380", "3.8")
        assert case.expected.value == 3.8
        assert case.expected.evidence[0].reference_id == "table_1"
        assert case.expected.evidence[0].locator == {
            "source_key_prefix": "raw/benchmarks/uda_fin/all/ADI_2009/"
        }
        assert tuple(type(evaluator) for evaluator in benchmark.evaluators()) == (
            AnswerContains,
        )
        assert run_units == [
            BenchmarkRunUnit(
                unit_id="ADI_2009",
                document_ids=("ADI_2009",),
                case_ids=("ADI/2009/page_49.pdf-1",),
                metadata={"doc_name": "ADI_2009", "subset": "fin"},
            )
        ]

        score = await EvidenceRecallAtK(k=1).evaluate(
            case=case,
            response=QueryResponse(
                mode="vector_search",
                results=(
                    QueryResult(
                        id="hit_1",
                        text="interest expense was 380",
                        source={
                            "object_key": documents[0].raw_key(benchmark.manifest),
                        },
                    ),
                ),
            ),
        )
        assert score.value == 1.0

        missed_score = await EvidenceRecallAtK(k=1).evaluate(
            case=case,
            response=QueryResponse(
                mode="vector_search",
                results=(
                    QueryResult(
                        id="hit_2",
                        text="the same document contains unrelated content",
                        source={
                            "object_key": documents[0].raw_key(benchmark.manifest),
                        },
                    ),
                ),
            ),
        )
        assert missed_score.value == 0.0

    asyncio.run(run())


def test_uda_source_zip_rejects_unsafe_member_paths(tmp_path: Path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("../escape.pdf", b"bad")

    try:
        uda_module._extract_zip(archive_path, tmp_path / "out")
    except ValueError as exc:
        assert "unsafe zip member path" in str(exc)
    else:
        raise AssertionError("expected unsafe zip member path error")


def test_uda_benchmark_reports_missing_source_documents(tmp_path: Path):
    qa_path = tmp_path / "fin_qa.csv"
    source_root = tmp_path / "src_doc_files"
    source_root.mkdir()
    qa_path.write_text(
        "\n".join(
            (
                "doc_name|q_uid|question|answer_1",
                "ADI_2009|case-1|what is interest expense?|380",
            )
        ),
        encoding="utf-8",
    )
    benchmark = UdaBenchmark(
        subset="fin",
        source_root=source_root,
        qa_path=qa_path,
        download_metadata=False,
    )

    async def run():
        prepared = await benchmark.prepare(
            BenchmarkWorkspace(root_dir=tmp_path, cache_dir=tmp_path / "cache")
        )
        try:
            [document async for document in benchmark.documents(prepared)]
        except FileNotFoundError as exc:
            assert "ADI_2009" in str(exc)
        else:
            raise AssertionError("expected missing source document error")

    asyncio.run(run())


def test_uda_benchmark_downloads_source_documents_when_source_root_is_not_set(
    tmp_path: Path,
    monkeypatch,
):
    qa_path = tmp_path / "fin_qa.csv"
    qa_path.write_text(
        "\n".join(
            (
                "doc_name|q_uid|question|answer_1",
                "ADI_2009|case-1|what is interest expense?|380",
            )
        ),
        encoding="utf-8",
    )

    def fake_read_url(url: str) -> bytes:
        assert url == (
            "https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/main/"
            "src_doc_files/fin_docs.zip?download=true"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as archive:
            archive.writestr("fin_docs/ADI_2009.pdf", b"%PDF-1.4\n% downloaded fixture\n")
        return buffer.getvalue()

    monkeypatch.setattr(uda_module, "_read_url", fake_read_url)
    benchmark = UdaBenchmark(subset="fin", qa_path=qa_path, download_metadata=False)

    async def run():
        prepared = await benchmark.prepare(
            BenchmarkWorkspace(root_dir=tmp_path, cache_dir=tmp_path / "cache")
        )
        documents = [document async for document in benchmark.documents(prepared)]

        assert prepared.metadata["source_root"] == str(
            tmp_path / "cache" / "uda_fin" / "source_documents"
        )
        assert len(documents) == 1
        assert documents[0].path == (
            tmp_path / "cache" / "uda_fin" / "source_documents" / "fin_docs" / "ADI_2009.pdf"
        )

    asyncio.run(run())
