"""Shared data types for Heta benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from heta_framework.kb.search import (
    QueryCitation,
    QueryResponse,
    QueryResult,
    QueryTraceEvent,
)

EVALUATION_SCHEMA_VERSION = "1"

BenchmarkBuildScope = Literal["corpus", "group", "case"]
BenchmarkResourceKind = Literal["file", "archive", "dataset", "repository", "manual"]


@dataclass(frozen=True)
class BenchmarkManifest:
    """Stable identity and citation metadata for one benchmark split."""

    name: str
    version: str
    split: str
    task_type: str
    build_scope: BenchmarkBuildScope = "corpus"
    homepage: str | None = None
    license: str | None = None
    citation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(self, "version", _required_text(self.version, "version"))
        object.__setattr__(self, "split", _required_text(self.split, "split"))
        object.__setattr__(self, "task_type", _required_text(self.task_type, "task_type"))
        if self.build_scope not in {"corpus", "group", "case"}:
            raise ValueError("build_scope must be one of: corpus, group, case")
        object.__setattr__(self, "homepage", _optional_text(self.homepage, "homepage"))
        object.__setattr__(self, "license", _optional_text(self.license, "license"))
        object.__setattr__(self, "citation", _optional_text(self.citation, "citation"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        """Return a stable key for cache and report paths."""
        return f"{self.name}:{self.version}:{self.split}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "name": self.name,
            "version": self.version,
            "split": self.split,
            "task_type": self.task_type,
            "build_scope": self.build_scope,
            "homepage": self.homepage,
            "license": self.license,
            "citation": self.citation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkResource:
    """One external resource needed to prepare a benchmark."""

    name: str
    uri: str
    kind: BenchmarkResourceKind = "file"
    checksum: str | None = None
    size_bytes: int | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(self, "uri", _required_text(self.uri, "uri"))
        if self.kind not in {"file", "archive", "dataset", "repository", "manual"}:
            raise ValueError("kind must be one of: file, archive, dataset, repository, manual")
        object.__setattr__(self, "checksum", _optional_text(self.checksum, "checksum"))
        if self.size_bytes is not None and self.size_bytes <= 0:
            raise ValueError("size_bytes must be greater than zero")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "name": self.name,
            "uri": self.uri,
            "kind": self.kind,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "required": self.required,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkWorkspace:
    """Local directories used while preparing and running a benchmark."""

    root_dir: Path
    cache_dir: Path
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_dir", Path(self.root_dir))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir) if self.output_dir is not None else None,
        )


@dataclass(frozen=True)
class PreparedBenchmark:
    """Prepared local benchmark state returned by BenchmarkProtocol.prepare."""

    manifest: BenchmarkManifest
    root_dir: Path
    resources: tuple[BenchmarkResource, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_dir", Path(self.root_dir))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class BenchmarkDocument:
    """One source document that should be written to a KB ObjectStore raw prefix."""

    document_id: str
    name: str
    media_type: str
    data: bytes | None = None
    path: Path | None = None
    source_uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _required_text(self.document_id, "document_id"))
        object.__setattr__(self, "name", _document_name(self.name))
        object.__setattr__(self, "media_type", _required_text(self.media_type, "media_type"))
        object.__setattr__(self, "path", Path(self.path) if self.path is not None else None)
        object.__setattr__(self, "source_uri", _optional_text(self.source_uri, "source_uri"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        locations = sum(
            item is not None
            for item in (
                self.data,
                self.path,
                self.source_uri,
            )
        )
        if locations != 1:
            raise ValueError("exactly one of data, path, or source_uri must be set")

    def raw_key(self, manifest: BenchmarkManifest) -> str:
        """Return the default ObjectStore raw key for this benchmark document."""
        return (
            f"raw/benchmarks/{manifest.name}/{manifest.split}/"
            f"{self.document_id}/{self.name}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return metadata for a report without embedding document bytes."""
        return {
            "document_id": self.document_id,
            "name": self.name,
            "media_type": self.media_type,
            "has_data": self.data is not None,
            "path": str(self.path) if self.path is not None else None,
            "source_uri": self.source_uri,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Gold evidence for a benchmark case.

    locator intentionally stays open-ended. Built-in evaluators understand common
    keys such as document_id, source_key, page_index, chunk_id, table_id, row_index,
    and column, while custom benchmarks may add their own fields.
    """

    reference_id: str | None = None
    locator: Mapping[str, Any] = field(default_factory=dict)
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _optional_text(self.reference_id, "reference_id"))
        object.__setattr__(self, "locator", dict(self.locator))
        object.__setattr__(self, "text", _optional_text(self.text, "text"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.reference_id is None and not self.locator and self.text is None:
            raise ValueError("evidence must set at least one of reference_id, locator, or text")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "reference_id": self.reference_id,
            "locator": dict(self.locator),
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkExpected:
    """Expected output and evidence labels for a benchmark case."""

    answers: tuple[str, ...] = ()
    evidence: tuple[BenchmarkEvidence, ...] = ()
    value: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "answers",
            tuple(_required_text(answer, "answer") for answer in self.answers),
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "answers": list(self.answers),
            "evidence": [item.to_dict() for item in self.evidence],
            "value": _json_safe(self.value),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkCase:
    """One query case from a benchmark."""

    case_id: str
    query: str
    expected: BenchmarkExpected = field(default_factory=BenchmarkExpected)
    labels: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text(self.case_id, "case_id"))
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        object.__setattr__(self, "labels", dict(self.labels))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "case_id": self.case_id,
            "query": self.query,
            "expected": self.expected.to_dict(),
            "labels": dict(self.labels),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkRunUnit:
    """One independent KB build-and-query unit within a benchmark run.

    Empty ``document_ids`` or ``case_ids`` means all benchmark documents or cases.
    This keeps corpus-level benchmarks compact while allowing document-scoped
    benchmarks to ask the runner to build several smaller KBs.
    """

    unit_id: str
    document_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _required_text(self.unit_id, "unit_id"))
        object.__setattr__(
            self,
            "document_ids",
            tuple(_required_text(item, "document_id") for item in self.document_ids),
        )
        object.__setattr__(
            self,
            "case_ids",
            tuple(_required_text(item, "case_id") for item in self.case_ids),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_corpus_level(self) -> bool:
        """Return whether this unit uses all documents and all cases."""
        return not self.document_ids and not self.case_ids

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "unit_id": self.unit_id,
            "document_ids": list(self.document_ids),
            "case_ids": list(self.case_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationScore:
    """Score produced by one benchmark evaluator for one query response."""

    name: str
    value: float | bool | str
    passed: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        if isinstance(self.value, str):
            object.__setattr__(self, "value", _required_text(self.value, "value"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationError:
    """Non-fatal error for one benchmark case and query mode."""

    message: str
    error_type: str
    retryable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", _required_text(self.message, "message"))
        object.__setattr__(self, "error_type", _required_text(self.error_type, "error_type"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "message": self.message,
            "error_type": self.error_type,
            "retryable": self.retryable,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Evaluation result for one benchmark case and one query mode."""

    case_id: str
    query: str
    query_mode: str
    scores: tuple[EvaluationScore, ...] = ()
    response: QueryResponse | None = None
    latency_ms: float | None = None
    error: EvaluationError | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text(self.case_id, "case_id"))
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        object.__setattr__(self, "query_mode", _required_text(self.query_mode, "query_mode"))
        object.__setattr__(self, "scores", tuple(self.scores))
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.response is None and self.error is None:
            raise ValueError("either response or error must be set")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "case_id": self.case_id,
            "query": self.query,
            "query_mode": self.query_mode,
            "scores": [score.to_dict() for score in self.scores],
            "response": _query_response_to_dict(self.response) if self.response else None,
            "latency_ms": self.latency_ms,
            "error": self.error.to_dict() if self.error else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable report for one recipe evaluation against one benchmark."""

    report_id: str
    benchmark: BenchmarkManifest
    knowledge_base_name: str
    knowledge_base_manifest: Mapping[str, Any]
    recipe_manifest: Mapping[str, Any]
    query_modes: tuple[str, ...]
    score_summary: Mapping[str, float]
    case_results: tuple[EvaluationCaseResult, ...]
    started_at: str
    finished_at: str
    schema_version: str = EVALUATION_SCHEMA_VERSION
    report_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_id", _required_text(self.report_id, "report_id"))
        object.__setattr__(
            self,
            "knowledge_base_name",
            _required_text(self.knowledge_base_name, "knowledge_base_name"),
        )
        object.__setattr__(
            self,
            "query_modes",
            tuple(_required_text(mode, "query_mode") for mode in self.query_modes),
        )
        object.__setattr__(self, "score_summary", dict(self.score_summary))
        object.__setattr__(self, "case_results", tuple(self.case_results))
        object.__setattr__(self, "started_at", _required_text(self.started_at, "started_at"))
        object.__setattr__(self, "finished_at", _required_text(self.finished_at, "finished_at"))
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, "schema_version"),
        )
        object.__setattr__(self, "report_key", _optional_text(self.report_key, "report_key"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation suitable for ObjectStore."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "report_key": self.report_key,
            "benchmark": self.benchmark.to_dict(),
            "knowledge_base_name": self.knowledge_base_name,
            "knowledge_base_manifest": _json_safe(self.knowledge_base_manifest),
            "recipe_manifest": _json_safe(self.recipe_manifest),
            "query_modes": list(self.query_modes),
            "score_summary": dict(self.score_summary),
            "case_results": [result.to_dict() for result in self.case_results],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
        }


def default_report_key(knowledge_base_name: str, report_id: str) -> str:
    """Return the default ObjectStore key for an evaluation report."""
    return (
        f"_heta/knowledge_bases/{_slug(knowledge_base_name)}"
        f"/evaluations/{_slug(report_id)}/report.json"
    )


def _query_response_to_dict(response: QueryResponse) -> dict[str, Any]:
    return {
        "mode": response.mode,
        "results": [_query_result_to_dict(result) for result in response.results],
        "answer": response.answer,
        "citations": [_query_citation_to_dict(citation) for citation in response.citations],
        "trace": [_query_trace_to_dict(event) for event in response.trace],
        "metadata": _json_safe(response.metadata),
    }


def _query_result_to_dict(result: QueryResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "text": result.text,
        "score": result.score,
        "kind": result.kind,
        "source": _json_safe(result.source),
        "metadata": _json_safe(result.metadata),
    }


def _query_citation_to_dict(citation: QueryCitation) -> dict[str, Any]:
    return {
        "id": citation.id,
        "result_id": citation.result_id,
        "source": _json_safe(citation.source),
        "text": citation.text,
        "metadata": _json_safe(citation.metadata),
    }


def _query_trace_to_dict(event: QueryTraceEvent) -> dict[str, Any]:
    return {
        "stage": event.stage,
        "message": event.message,
        "metadata": _json_safe(event.metadata),
    }


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _document_name(value: str) -> str:
    name = _required_text(value, "name")
    if Path(name).name != name or "\\" in name:
        raise ValueError("name must be a file name, not a path")
    return name


def _slug(value: str) -> str:
    return "_".join(_required_text(value, "value").lower().split())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_json_safe(item) for item in value]
    return {
        "type": type(value).__name__,
        "manifest_note": "value omitted because it is not JSON-safe",
    }


__all__ = [
    "BenchmarkCase",
    "BenchmarkBuildScope",
    "BenchmarkDocument",
    "BenchmarkEvidence",
    "BenchmarkExpected",
    "BenchmarkManifest",
    "BenchmarkResource",
    "BenchmarkResourceKind",
    "BenchmarkRunUnit",
    "BenchmarkWorkspace",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationCaseResult",
    "EvaluationError",
    "EvaluationReport",
    "EvaluationScore",
    "PreparedBenchmark",
    "default_report_key",
]
