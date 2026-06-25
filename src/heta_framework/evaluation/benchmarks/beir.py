"""BEIR benchmark adapter."""

from __future__ import annotations

import json
import zipfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

from heta_framework.evaluation.evaluators.retrieval import (
    BeirRetrievalMetric,
    beir_default_metrics,
)
from heta_framework.evaluation.protocols import BenchmarkEvaluatorProtocol
from heta_framework.evaluation.types import (
    BenchmarkCase,
    BenchmarkDocument,
    BenchmarkEvidence,
    BenchmarkExpected,
    BenchmarkManifest,
    BenchmarkResource,
    BenchmarkRunUnit,
    BenchmarkWorkspace,
    PreparedBenchmark,
)

BEIR_GITHUB_URL = "https://github.com/beir-cellar/beir"
BEIR_DATASET_BASE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
)
BEIR_RECOMMENDED_DATASETS = ("scifact", "nfcorpus", "fiqa", "hotpotqa")

_DATASET_TASKS: Mapping[str, str] = {
    "scifact": "scientific_fact_retrieval",
    "nfcorpus": "biomedical_information_retrieval",
    "fiqa": "financial_question_answering_retrieval",
    "hotpotqa": "multi_hop_question_answering_retrieval",
}

_DATASET_DESCRIPTIONS: Mapping[str, str] = {
    "scifact": "Scientific claim verification retrieval. Small and fast for smoke tests.",
    "nfcorpus": "Biomedical and medical information retrieval with compact corpus size.",
    "fiqa": "Financial question-answering retrieval with stronger domain shift.",
    "hotpotqa": "Multi-hop question-answering retrieval in BEIR's standard IR format.",
}


@dataclass(frozen=True)
class BeirBenchmark:
    """Adapter for one BEIR dataset split.

    BEIR qrels are document-level. The adapter writes each corpus item as one text
    document and keeps the stable benchmark document id in the raw object key so
    chunk-level Heta search results can be mapped back to BEIR labels.
    """

    dataset: str
    split: str = "test"
    data_root: Path | None = None
    download: bool = True
    dataset_url: str | None = None
    evaluator_list: tuple[BenchmarkEvaluatorProtocol, ...] = field(
        default_factory=beir_default_metrics
    )

    def __post_init__(self) -> None:
        dataset = _required_id(self.dataset, "dataset")
        split = _required_id(self.split, "split")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "split", split)
        object.__setattr__(
            self,
            "data_root",
            Path(self.data_root) if self.data_root is not None else None,
        )
        object.__setattr__(self, "evaluator_list", tuple(self.evaluator_list))

    @property
    def manifest(self) -> BenchmarkManifest:
        """Return stable benchmark identity metadata."""
        return BenchmarkManifest(
            name=f"beir_{self.dataset}",
            version="official",
            split=self.split,
            task_type=_DATASET_TASKS.get(self.dataset, "information_retrieval"),
            build_scope="corpus",
            homepage=BEIR_GITHUB_URL,
            license=None,
            citation="BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
            metadata={
                "dataset": self.dataset,
                "recommended": self.dataset in BEIR_RECOMMENDED_DATASETS,
                "description": _DATASET_DESCRIPTIONS.get(self.dataset),
                "dataset_url": self._dataset_url(),
            },
        )

    def resources(self) -> tuple[BenchmarkResource, ...]:
        """Return the BEIR archive resource."""
        return (
            BenchmarkResource(
                name=f"beir_{self.dataset}",
                uri=self._dataset_url(),
                kind="archive",
                metadata={
                    "dataset": self.dataset,
                    "split": self.split,
                    "recommended_datasets": list(BEIR_RECOMMENDED_DATASETS),
                },
            ),
        )

    async def prepare(self, workspace: BenchmarkWorkspace) -> PreparedBenchmark:
        """Locate or download the BEIR dataset directory."""
        root_dir = workspace.cache_dir / self.manifest.name
        root_dir.mkdir(parents=True, exist_ok=True)
        data_root = self.data_root or root_dir / self.dataset
        if self.data_root is None and self.download:
            _download_and_extract(self._dataset_url(), root_dir, expected_dir=self.dataset)
        _validate_beir_root(data_root, self.split)
        return PreparedBenchmark(
            manifest=self.manifest,
            root_dir=data_root,
            resources=self.resources(),
            metadata={
                "dataset": self.dataset,
                "split": self.split,
                "data_root": str(data_root),
                "recommended_datasets": list(BEIR_RECOMMENDED_DATASETS),
            },
        )

    async def documents(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkDocument]:
        """Yield BEIR corpus entries as text documents."""
        corpus_path = Path(prepared.root_dir) / "corpus.jsonl"
        for item in _read_jsonl(corpus_path):
            original_id = _required_text(item.get("_id"), "_id")
            document_id = _benchmark_document_id(original_id)
            title = str(item.get("title") or "").strip()
            text = str(item.get("text") or "").strip()
            content = "\n\n".join(part for part in (title, text) if part).strip()
            if not content:
                content = title or original_id
            yield BenchmarkDocument(
                document_id=document_id,
                name=f"{document_id}.txt",
                media_type="text/plain",
                data=content.encode("utf-8"),
                metadata={
                    "beir_doc_id": original_id,
                    "beir_document_id": document_id,
                    "dataset": self.dataset,
                    "title": title,
                },
            )

    async def cases(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkCase]:
        """Yield BEIR queries with qrels as expected evidence."""
        queries = {
            _required_text(item.get("_id"), "_id"): _required_text(item.get("text"), "text")
            for item in _read_jsonl(Path(prepared.root_dir) / "queries.jsonl")
        }
        qrels = _read_qrels(Path(prepared.root_dir) / "qrels" / f"{self.split}.tsv")
        for query_id, query in queries.items():
            labels = qrels.get(query_id)
            if not labels:
                continue
            yield BenchmarkCase(
                case_id=query_id,
                query=query,
                expected=BenchmarkExpected(
                    evidence=tuple(
                        BenchmarkEvidence(
                            reference_id=_benchmark_document_id(document_id),
                            locator={
                                "source_key_prefix": _source_key_prefix(
                                    self.manifest,
                                    _benchmark_document_id(document_id),
                                ),
                            },
                            metadata={
                                "beir_doc_id": document_id,
                                "beir_document_id": _benchmark_document_id(document_id),
                                "relevance": relevance,
                            },
                        )
                        for document_id, relevance in labels.items()
                        if relevance > 0
                    ),
                    metadata={
                        "beir_query_id": query_id,
                        "positive_qrels": sum(1 for value in labels.values() if value > 0),
                    },
                ),
                labels={"dataset": self.dataset, "split": self.split},
                metadata={"beir_query_id": query_id, "dataset": self.dataset},
            )

    async def run_units(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkRunUnit]:
        """Yield one corpus-level run unit for standard BEIR retrieval."""
        yield BenchmarkRunUnit(
            unit_id="corpus",
            metadata={"dataset": self.dataset, "split": self.split},
        )

    def evaluators(self) -> tuple[BenchmarkEvaluatorProtocol, ...]:
        """Return default BEIR retrieval metrics."""
        return self.evaluator_list

    def _dataset_url(self) -> str:
        if self.dataset_url is not None:
            return self.dataset_url
        return f"{BEIR_DATASET_BASE_URL}/{self.dataset}.zip"


def _download_and_extract(url: str, target_root: Path, *, expected_dir: str) -> None:
    target_dir = target_root / expected_dir
    if target_dir.exists():
        return
    archive_path = target_root / "_downloads" / f"{expected_dir}.zip"
    if not archive_path.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(_read_url(url))
    _extract_zip(archive_path, target_root)


def _validate_beir_root(root: Path, split: str) -> None:
    required = (
        root / "corpus.jsonl",
        root / "queries.jsonl",
        root / "qrels" / f"{split}.tsv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"BEIR dataset is missing required files: {missing}")


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return tuple(rows)


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if line_number == 1 and _is_qrels_header(parts):
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number} must contain query-id, corpus-id, score")
            query_id, document_id, score = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not query_id or not document_id:
                raise ValueError(f"{path}:{line_number} has an empty qrels id")
            try:
                relevance = int(float(score))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} has invalid relevance score: {score}") from exc
            qrels.setdefault(query_id, {})[document_id] = relevance
    return qrels


def _is_qrels_header(parts: list[str]) -> bool:
    normalized = [part.strip().lower() for part in parts]
    return len(normalized) >= 3 and normalized[:3] in (
        ["query-id", "corpus-id", "score"],
        ["query_id", "corpus_id", "score"],
    )


def _source_key_prefix(manifest: BenchmarkManifest, document_id: str) -> str:
    return f"raw/benchmarks/{manifest.name}/{manifest.split}/{document_id}/"


def _benchmark_document_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    if not safe:
        safe = "document"
    if safe == value and len(safe) <= 96:
        return safe
    digest = sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:83].strip('._-')}_{digest}" if safe.strip("._-") else f"document_{digest}"


def _extract_zip(archive_path: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            destination = _safe_zip_destination(target_root, member.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as output:
                output.write(source.read())


def _safe_zip_destination(target_root: Path, member_name: str) -> Path:
    destination = (target_root / member_name).resolve()
    root = target_root.resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"unsafe zip member path: {member_name}")
    return destination


def _read_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "heta-framework-evaluation/0.1"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def _required_id(value: str, field_name: str) -> str:
    normalized = _required_text(value, field_name).lower()
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"{field_name} must be a dataset id, not a path")
    return normalized


def _required_text(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must not be empty")
    normalized = str(value).strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized


__all__ = [
    "BEIR_DATASET_BASE_URL",
    "BEIR_GITHUB_URL",
    "BEIR_RECOMMENDED_DATASETS",
    "BeirBenchmark",
]
