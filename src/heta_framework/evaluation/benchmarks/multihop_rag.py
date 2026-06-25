"""MultiHop-RAG benchmark adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlopen

from heta_framework.evaluation.evaluators.answer import AnswerContains
from heta_framework.evaluation.evaluators.retrieval import EvidenceRecallAtK
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

MULTIHOP_RAG_CORPUS_URL = (
    "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json"
)
MULTIHOP_RAG_QUERIES_URL = (
    "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"
)


@dataclass(frozen=True)
class MultiHopRagBenchmark:
    """Adapter for the official MultiHop-RAG benchmark."""

    corpus_path: Path | None = None
    queries_path: Path | None = None
    download: bool = False
    evaluator_list: tuple[BenchmarkEvaluatorProtocol, ...] = field(
        default_factory=lambda: (EvidenceRecallAtK(k=5), AnswerContains())
    )

    @property
    def manifest(self) -> BenchmarkManifest:
        """Return stable benchmark identity metadata."""
        return BenchmarkManifest(
            name="multihop_rag",
            version="official",
            split="all",
            task_type="multi_hop_qa",
            build_scope="corpus",
            homepage="https://github.com/yixuantt/MultiHop-RAG",
            citation=(
                "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation "
                "for Multi-Hop Queries"
            ),
            metadata={
                "corpus_url": MULTIHOP_RAG_CORPUS_URL,
                "queries_url": MULTIHOP_RAG_QUERIES_URL,
            },
        )

    def resources(self) -> tuple[BenchmarkResource, ...]:
        """Return official resources used by this adapter."""
        return (
            BenchmarkResource(
                name="corpus",
                uri=MULTIHOP_RAG_CORPUS_URL,
                kind="file",
                metadata={"filename": "corpus.json"},
            ),
            BenchmarkResource(
                name="queries",
                uri=MULTIHOP_RAG_QUERIES_URL,
                kind="file",
                metadata={"filename": "MultiHopRAG.json"},
            ),
        )

    async def prepare(self, workspace: BenchmarkWorkspace) -> PreparedBenchmark:
        """Locate or download the official corpus and query files."""
        root_dir = workspace.cache_dir / "multihop_rag"
        root_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = self.corpus_path or root_dir / "corpus.json"
        queries_path = self.queries_path or root_dir / "MultiHopRAG.json"
        if self.download:
            _download_if_missing(MULTIHOP_RAG_CORPUS_URL, corpus_path)
            _download_if_missing(MULTIHOP_RAG_QUERIES_URL, queries_path)
        if not corpus_path.exists():
            raise FileNotFoundError(
                "MultiHop-RAG corpus.json not found. Provide corpus_path or set download=True."
            )
        if not queries_path.exists():
            raise FileNotFoundError(
                "MultiHop-RAG MultiHopRAG.json not found. "
                "Provide queries_path or set download=True."
            )
        return PreparedBenchmark(
            manifest=self.manifest,
            root_dir=root_dir,
            resources=self.resources(),
            metadata={
                "corpus_path": str(corpus_path),
                "queries_path": str(queries_path),
            },
        )

    async def documents(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkDocument]:
        """Yield corpus articles as text documents."""
        corpus = _load_json_array(Path(str(prepared.metadata["corpus_path"])))
        for row in corpus:
            article = _article(row)
            yield BenchmarkDocument(
                document_id=article.document_id,
                name=f"{article.document_id}.txt",
                media_type="text/plain",
                data=_document_text(article).encode("utf-8"),
                metadata=article.metadata,
            )

    async def cases(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkCase]:
        """Yield MultiHop-RAG query cases."""
        queries = _load_json_array(Path(str(prepared.metadata["queries_path"])))
        for index, row in enumerate(queries):
            yield _case(row, index, self.manifest)

    async def run_units(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkRunUnit]:
        """Yield one corpus-level unit for the official MultiHop-RAG retrieval setup."""
        yield BenchmarkRunUnit(unit_id="corpus")

    def evaluators(self) -> tuple[BenchmarkEvaluatorProtocol, ...]:
        """Return default evaluators for MultiHop-RAG."""
        return tuple(self.evaluator_list)


@dataclass(frozen=True)
class _Article:
    document_id: str
    title: str
    body: str
    metadata: Mapping[str, Any]


def _article(row: object) -> _Article:
    if not isinstance(row, Mapping):
        raise ValueError("corpus row must be an object")
    title = _required_string(row.get("title"), "title")
    url = _optional_string(row.get("url"))
    document_id = _article_id(title=title, url=url)
    return _Article(
        document_id=document_id,
        title=title,
        body=_required_string(row.get("body"), "body"),
        metadata={
            "title": title,
            "author": row.get("author"),
            "source": row.get("source"),
            "published_at": row.get("published_at"),
            "category": row.get("category"),
            "url": url,
        },
    )


def _case(row: object, index: int, manifest: BenchmarkManifest) -> BenchmarkCase:
    if not isinstance(row, Mapping):
        raise ValueError(f"query row {index} must be an object")
    evidence_rows = row.get("evidence_list", ())
    if not isinstance(evidence_rows, list):
        raise ValueError("MultiHop-RAG evidence_list must be a list")
    return BenchmarkCase(
        case_id=f"multihop_rag_{index:04d}",
        query=_required_string(row.get("query"), "query"),
        expected=BenchmarkExpected(
            answers=(_required_string(row.get("answer"), "answer"),),
            evidence=tuple(_evidence(item, manifest) for item in evidence_rows),
            metadata={"question_type": row.get("question_type")},
        ),
        labels={"question_type": row.get("question_type")},
        metadata={"original_index": index},
    )


def _evidence(row: object, manifest: BenchmarkManifest) -> BenchmarkEvidence:
    if not isinstance(row, Mapping):
        raise ValueError("MultiHop-RAG evidence item must be an object")
    title = _required_string(row.get("title"), "title")
    url = _optional_string(row.get("url"))
    document_id = _article_id(title=title, url=url)
    source_key = (
        f"raw/benchmarks/{manifest.name}/{manifest.split}/{document_id}/{document_id}.txt"
    )
    return BenchmarkEvidence(
        reference_id=document_id,
        locator={"source_key": source_key},
        text=_optional_string(row.get("fact")),
        metadata={
            "document_id": document_id,
            "title": title,
            "author": row.get("author"),
            "source": row.get("source"),
            "category": row.get("category"),
            "published_at": row.get("published_at"),
            "url": url,
        },
    )


def _document_text(article: _Article) -> str:
    metadata = article.metadata
    return (
        f"Title: {article.title}\n"
        f"Source: {metadata.get('source') or ''}\n"
        f"Published at: {metadata.get('published_at') or ''}\n"
        f"URL: {metadata.get('url') or ''}\n\n"
        f"{article.body}"
    )


def _article_id(*, title: str, url: str | None) -> str:
    import hashlib

    identity = url or title
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"article_{suffix}"


def _load_json_array(path: Path) -> tuple[object, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return tuple(value)


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def _required_string(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{field_name} must not be empty")
    return text


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "MULTIHOP_RAG_CORPUS_URL",
    "MULTIHOP_RAG_QUERIES_URL",
    "MultiHopRagBenchmark",
]
