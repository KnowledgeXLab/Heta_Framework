"""JSONL benchmark adapter."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

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


@dataclass(frozen=True)
class JsonlBenchmark:
    """Benchmark adapter backed by documents.jsonl and cases.jsonl files."""

    manifest: BenchmarkManifest
    documents_jsonl: Path
    cases_jsonl: Path
    root_dir: Path | None = None
    resources_value: tuple[BenchmarkResource, ...] = ()
    evaluator_list: tuple[BenchmarkEvaluatorProtocol, ...] = field(
        default_factory=lambda: (EvidenceRecallAtK(k=5), AnswerContains())
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "documents_jsonl", Path(self.documents_jsonl))
        object.__setattr__(self, "cases_jsonl", Path(self.cases_jsonl))
        object.__setattr__(
            self,
            "root_dir",
            Path(self.root_dir) if self.root_dir is not None else self.documents_jsonl.parent,
        )
        object.__setattr__(self, "resources_value", tuple(self.resources_value))
        object.__setattr__(self, "evaluator_list", tuple(self.evaluator_list))

    def resources(self) -> tuple[BenchmarkResource, ...]:
        """Return external resources needed to prepare this benchmark."""
        return self.resources_value

    async def prepare(self, workspace: BenchmarkWorkspace) -> PreparedBenchmark:
        """Validate local JSONL files and return prepared state."""
        if not self.documents_jsonl.exists():
            raise FileNotFoundError(f"documents_jsonl does not exist: {self.documents_jsonl}")
        if not self.cases_jsonl.exists():
            raise FileNotFoundError(f"cases_jsonl does not exist: {self.cases_jsonl}")
        return PreparedBenchmark(
            manifest=self.manifest,
            root_dir=self.root_dir or workspace.cache_dir,
            resources=self.resources_value,
            metadata={
                "documents_jsonl": str(self.documents_jsonl),
                "cases_jsonl": str(self.cases_jsonl),
            },
        )

    async def documents(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkDocument]:
        """Yield benchmark documents from JSONL."""
        for row in _read_jsonl(self.documents_jsonl):
            yield _document_from_row(row, prepared.root_dir)

    async def cases(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkCase]:
        """Yield benchmark cases from JSONL."""
        for row in _read_jsonl(self.cases_jsonl):
            yield _case_from_row(row)

    async def run_units(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkRunUnit]:
        """Yield one corpus-level run unit."""
        yield BenchmarkRunUnit(unit_id="corpus")

    def evaluators(self) -> tuple[BenchmarkEvaluatorProtocol, ...]:
        """Return default evaluators for this benchmark."""
        return self.evaluator_list


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() == "":
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{index}: JSONL row must be an object")
        rows.append(value)
    return tuple(rows)


def _document_from_row(row: Mapping[str, Any], root_dir: Path) -> BenchmarkDocument:
    data = None
    path = None
    source_uri = _optional_string(row.get("source_uri"), "source_uri")
    if "text" in row:
        data = str(row["text"]).encode("utf-8")
    if "data_base64" in row:
        if data is not None:
            raise ValueError("document row must not set both text and data_base64")
        data = base64.b64decode(_required_string(row.get("data_base64"), "data_base64"))
    if "path" in row:
        path = root_dir / _required_string(row.get("path"), "path")
    return BenchmarkDocument(
        document_id=_required_string(row.get("document_id"), "document_id"),
        name=_required_string(row.get("name"), "name"),
        media_type=_required_string(row.get("media_type"), "media_type"),
        data=data,
        path=path,
        source_uri=source_uri,
        metadata=_mapping(row.get("metadata")),
    )


def _case_from_row(row: Mapping[str, Any]) -> BenchmarkCase:
    expected = row.get("expected", {})
    if expected is None:
        expected = {}
    if not isinstance(expected, Mapping):
        raise ValueError("case expected must be an object")
    return BenchmarkCase(
        case_id=_required_string(row.get("case_id"), "case_id"),
        query=_required_string(row.get("query"), "query"),
        expected=BenchmarkExpected(
            answers=tuple(str(answer) for answer in expected.get("answers", ())),
            evidence=tuple(_evidence_from_row(item) for item in expected.get("evidence", ())),
            value=expected.get("value"),
            metadata=_mapping(expected.get("metadata")),
        ),
        labels=_mapping(row.get("labels")),
        metadata=_mapping(row.get("metadata")),
    )


def _evidence_from_row(row: object) -> BenchmarkEvidence:
    if not isinstance(row, Mapping):
        raise ValueError("evidence row must be an object")
    return BenchmarkEvidence(
        reference_id=_optional_string(row.get("reference_id"), "reference_id"),
        locator=_mapping(row.get("locator")),
        text=_optional_string(row.get("text"), "text"),
        metadata=_mapping(row.get("metadata")),
    )


def _required_string(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{field_name} must not be empty")
    return text


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{field_name} must not be empty")
    return text


def _mapping(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("value must be an object")
    return dict(value)


__all__ = ["JsonlBenchmark"]
