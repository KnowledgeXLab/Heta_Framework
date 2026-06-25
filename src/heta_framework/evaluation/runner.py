"""Runner for benchmark-driven recipe evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.evaluation.protocols import (
    BenchmarkEvaluatorProtocol,
    BenchmarkProtocol,
)
from heta_framework.evaluation.types import (
    BenchmarkCase,
    BenchmarkDocument,
    BenchmarkRunUnit,
    BenchmarkWorkspace,
    EvaluationCaseResult,
    EvaluationError,
    EvaluationReport,
    EvaluationScore,
    default_report_key,
)
from heta_framework.kb import KnowledgeBase, KnowledgeRecipe, QueryEngineRegistry


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Configuration for one benchmark run."""

    top_k: int = 10
    query_options: Mapping[str, Any] = field(default_factory=dict)
    trace: bool = False
    persist_report: bool = True
    max_concurrent_queries: int = 8
    report_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.max_concurrent_queries <= 0:
            raise ValueError("max_concurrent_queries must be greater than zero")
        if self.report_id is not None and self.report_id.strip() == "":
            raise ValueError("report_id must not be empty")
        object.__setattr__(self, "query_options", dict(self.query_options))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Result returned after evaluating one recipe against one benchmark."""

    knowledge_bases: tuple[KnowledgeBase, ...]
    report: EvaluationReport
    benchmark_document_keys: tuple[str, ...]
    report_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "knowledge_bases", tuple(self.knowledge_bases))
        if not self.knowledge_bases:
            raise ValueError("knowledge_bases must not be empty")
        object.__setattr__(self, "benchmark_document_keys", tuple(self.benchmark_document_keys))


class BenchmarkRunner:
    """Build a knowledge base from a recipe and evaluate it on a benchmark."""

    async def run(
        self,
        *,
        benchmark: BenchmarkProtocol,
        recipe: KnowledgeRecipe,
        knowledge_base_name: str,
        query_modes: tuple[str, ...],
        workspace: BenchmarkWorkspace | None = None,
        evaluators: tuple[BenchmarkEvaluatorProtocol, ...] | None = None,
        config: BenchmarkRunConfig | None = None,
        query_engines: QueryEngineRegistry | None = None,
        initial_artifacts: Mapping[str, Any] | None = None,
    ) -> BenchmarkRunResult:
        """Build a benchmark KB with a recipe, run queries, and return a report."""
        active_config = config or BenchmarkRunConfig()
        _validate_query_modes(query_modes)
        object_store = _require_object_store(recipe)
        active_workspace = workspace or _default_workspace()
        _ensure_workspace(active_workspace)

        started_at = _utc_now()
        prepared = await benchmark.prepare(active_workspace)
        documents = tuple([document async for document in benchmark.documents(prepared)])
        cases = tuple([case async for case in benchmark.cases(prepared)])
        run_units = tuple([unit async for unit in benchmark.run_units(prepared)])
        _validate_prepared_inputs(documents=documents, cases=cases, run_units=run_units)
        document_by_id = _documents_by_id(documents)
        case_by_id = _cases_by_id(cases)
        active_evaluators = evaluators if evaluators is not None else benchmark.evaluators()

        knowledge_bases: list[KnowledgeBase] = []
        case_results: list[EvaluationCaseResult] = []
        all_document_keys: list[str] = []
        for unit in run_units:
            unit_documents = _select_documents(unit, document_by_id)
            unit_cases = _select_cases(unit, case_by_id)
            document_keys = await _write_benchmark_documents(
                manifest=benchmark.manifest,
                documents=unit_documents,
                object_store=object_store,
            )
            all_document_keys.extend(document_keys)
            build_artifacts = dict(initial_artifacts or {})
            build_artifacts["benchmark_document_keys"] = document_keys
            build_artifacts["source_keys"] = document_keys

            knowledge_base = await KnowledgeBase.create(
                recipe=recipe,
                name=_knowledge_base_name(
                    base_name=knowledge_base_name,
                    unit=unit,
                    unit_count=len(run_units),
                ),
                initial_artifacts=build_artifacts,
            )
            if query_engines is not None:
                knowledge_base = _with_query_engines(knowledge_base, query_engines)
            knowledge_bases.append(knowledge_base)
            case_results.extend(
                await _evaluate_cases(
                    cases=unit_cases,
                    knowledge_base=knowledge_base,
                    query_modes=query_modes,
                    evaluators=active_evaluators,
                    config=active_config,
                    unit=unit,
                )
            )

        finished_at = _utc_now()
        report_id = active_config.report_id or f"eval_{uuid4().hex}"
        report_key = (
            default_report_key(knowledge_base_name, report_id)
            if active_config.persist_report
            else None
        )
        report = EvaluationReport(
            report_id=report_id,
            report_key=report_key,
            benchmark=benchmark.manifest,
            knowledge_base_name=knowledge_base_name,
            knowledge_base_manifest=_knowledge_base_manifest(knowledge_bases),
            recipe_manifest=recipe.manifest().to_dict(),
            query_modes=query_modes,
            score_summary=_score_summary(tuple(case_results)),
            case_results=tuple(case_results),
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                **dict(active_config.metadata),
                "benchmark_document_keys": list(dict.fromkeys(all_document_keys)),
                "run_units": [unit.to_dict() for unit in run_units],
            },
        )
        if report_key is not None:
            await object_store.put(
                report_key,
                json.dumps(report.to_dict(), ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )
        return BenchmarkRunResult(
            knowledge_bases=tuple(knowledge_bases),
            report=report,
            benchmark_document_keys=tuple(dict.fromkeys(all_document_keys)),
            report_key=report_key,
        )


async def _write_benchmark_documents(
    *,
    manifest: object,
    documents: tuple[BenchmarkDocument, ...],
    object_store: ObjectStoreProtocol,
) -> tuple[str, ...]:
    keys: list[str] = []
    for document in documents:
        key = document.raw_key(manifest)
        data = _document_bytes(document)
        await object_store.put(key, data)
        keys.append(key)
    if not keys:
        raise ValueError("benchmark produced no documents")
    return tuple(keys)


async def _evaluate_cases(
    *,
    cases: tuple[BenchmarkCase, ...],
    knowledge_base: KnowledgeBase,
    query_modes: tuple[str, ...],
    evaluators: tuple[BenchmarkEvaluatorProtocol, ...],
    config: BenchmarkRunConfig,
    unit: BenchmarkRunUnit,
) -> tuple[EvaluationCaseResult, ...]:
    results: list[EvaluationCaseResult] = []
    tasks: list[tuple[BenchmarkCase, str]] = []
    for case in cases:
        tasks.extend((case, query_mode) for query_mode in query_modes)
    if not tasks:
        raise ValueError("benchmark produced no cases")
    if config.max_concurrent_queries == 1:
        for case, query_mode in tasks:
            results.append(
                await _evaluate_one_case(
                    case=case,
                    knowledge_base=knowledge_base,
                    query_mode=query_mode,
                    evaluators=evaluators,
                    config=config,
                    unit=unit,
                )
            )
    else:
        semaphore = asyncio.Semaphore(config.max_concurrent_queries)
        results = list(
            await asyncio.gather(
                *(
                    _evaluate_one_case_with_semaphore(
                        semaphore=semaphore,
                        case=case,
                        knowledge_base=knowledge_base,
                        query_mode=query_mode,
                        evaluators=evaluators,
                        config=config,
                        unit=unit,
                    )
                    for case, query_mode in tasks
                )
            )
        )
    return tuple(results)


async def _evaluate_one_case_with_semaphore(
    *,
    semaphore: asyncio.Semaphore,
    case: BenchmarkCase,
    knowledge_base: KnowledgeBase,
    query_mode: str,
    evaluators: tuple[BenchmarkEvaluatorProtocol, ...],
    config: BenchmarkRunConfig,
    unit: BenchmarkRunUnit,
) -> EvaluationCaseResult:
    async with semaphore:
        return await _evaluate_one_case(
            case=case,
            knowledge_base=knowledge_base,
            query_mode=query_mode,
            evaluators=evaluators,
            config=config,
            unit=unit,
        )


async def _evaluate_one_case(
    *,
    case: BenchmarkCase,
    knowledge_base: KnowledgeBase,
    query_mode: str,
    evaluators: tuple[BenchmarkEvaluatorProtocol, ...],
    config: BenchmarkRunConfig,
    unit: BenchmarkRunUnit,
) -> EvaluationCaseResult:
    started = perf_counter()
    try:
        response = await knowledge_base.query(
            case.query,
            mode=query_mode,
            top_k=config.top_k,
            options=config.query_options,
            trace=config.trace,
        )
        scores = [
            await evaluator.evaluate(case=case, response=response)
            for evaluator in evaluators
        ]
        return EvaluationCaseResult(
            case_id=case.case_id,
            query=case.query,
            query_mode=query_mode,
            response=response,
            scores=tuple(scores),
            latency_ms=_elapsed_ms(started),
            metadata={
                "benchmark_run_unit": unit.unit_id,
                "knowledge_base_name": knowledge_base.name,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return EvaluationCaseResult(
            case_id=case.case_id,
            query=case.query,
            query_mode=query_mode,
            error=EvaluationError(
                message=str(exc) or exc.__class__.__name__,
                error_type=exc.__class__.__name__,
            ),
            latency_ms=_elapsed_ms(started),
            metadata={
                "benchmark_run_unit": unit.unit_id,
                "knowledge_base_name": knowledge_base.name,
            },
        )


def _score_summary(case_results: tuple[EvaluationCaseResult, ...]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for result in case_results:
        for score in result.scores:
            numeric = _numeric_score(score)
            if numeric is None:
                continue
            key = f"{result.query_mode}.{score.name}"
            values.setdefault(key, []).append(numeric)
    return {
        key: sum(items) / len(items)
        for key, items in sorted(values.items())
        if items
    }


def _numeric_score(score: EvaluationScore) -> float | None:
    if isinstance(score.value, bool):
        return 1.0 if score.value else 0.0
    if isinstance(score.value, (int, float)):
        return float(score.value)
    return None


def _document_bytes(document: object) -> bytes:
    data = getattr(document, "data", None)
    if data is not None:
        return data
    path = getattr(document, "path", None)
    if path is not None:
        return Path(path).read_bytes()
    source_uri = getattr(document, "source_uri", None)
    if source_uri is not None:
        raise NotImplementedError(
            "BenchmarkRunner cannot fetch source_uri documents yet; prepare them as data or path"
        )
    raise ValueError("benchmark document must provide data or path")


def _with_query_engines(
    knowledge_base: KnowledgeBase,
    query_engines: QueryEngineRegistry,
) -> KnowledgeBase:
    return KnowledgeBase(
        name=knowledge_base.name,
        description=knowledge_base.description,
        recipe=knowledge_base.recipe,
        run_record=knowledge_base.run_record,
        created_at=knowledge_base.created_at,
        updated_at=knowledge_base.updated_at,
        metadata=knowledge_base.metadata,
        query_engines=query_engines,
    )


def _validate_prepared_inputs(
    *,
    documents: tuple[BenchmarkDocument, ...],
    cases: tuple[BenchmarkCase, ...],
    run_units: tuple[BenchmarkRunUnit, ...],
) -> None:
    if not documents:
        raise ValueError("benchmark produced no documents")
    if not cases:
        raise ValueError("benchmark produced no cases")
    if not run_units:
        raise ValueError("benchmark produced no run units")
    _require_unique("document_id", [document.document_id for document in documents])
    _require_unique("case_id", [case.case_id for case in cases])
    _require_unique("unit_id", [unit.unit_id for unit in run_units])


def _documents_by_id(
    documents: tuple[BenchmarkDocument, ...],
) -> dict[str, BenchmarkDocument]:
    return {document.document_id: document for document in documents}


def _cases_by_id(cases: tuple[BenchmarkCase, ...]) -> dict[str, BenchmarkCase]:
    return {case.case_id: case for case in cases}


def _select_documents(
    unit: BenchmarkRunUnit,
    documents: Mapping[str, BenchmarkDocument],
) -> tuple[BenchmarkDocument, ...]:
    if not unit.document_ids:
        return tuple(documents.values())
    missing = [document_id for document_id in unit.document_ids if document_id not in documents]
    if missing:
        raise ValueError(f"benchmark run unit {unit.unit_id!r} references unknown documents: {missing}")
    return tuple(documents[document_id] for document_id in unit.document_ids)


def _select_cases(
    unit: BenchmarkRunUnit,
    cases: Mapping[str, BenchmarkCase],
) -> tuple[BenchmarkCase, ...]:
    if not unit.case_ids:
        return tuple(cases.values())
    missing = [case_id for case_id in unit.case_ids if case_id not in cases]
    if missing:
        raise ValueError(f"benchmark run unit {unit.unit_id!r} references unknown cases: {missing}")
    return tuple(cases[case_id] for case_id in unit.case_ids)


def _knowledge_base_name(
    *,
    base_name: str,
    unit: BenchmarkRunUnit,
    unit_count: int,
) -> str:
    if unit_count == 1 and unit.is_corpus_level:
        return base_name
    suffix = _slug(unit.unit_id)
    candidate = f"{base_name}-{suffix}"
    if len(candidate) <= 80:
        return candidate
    digest = hashlib.sha1(unit.unit_id.encode("utf-8")).hexdigest()[:10]
    max_base_length = max(1, 80 - len(digest) - 2)
    return f"{base_name[:max_base_length].rstrip(' -_')}-{digest}"


def _knowledge_base_manifest(knowledge_bases: list[KnowledgeBase]) -> dict[str, Any]:
    if len(knowledge_bases) == 1:
        return knowledge_bases[0].manifest().to_dict()
    return {
        "knowledge_bases": [
            knowledge_base.manifest().to_dict() for knowledge_base in knowledge_bases
        ]
    }


def _require_unique(field_name: str, values: list[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"benchmark produced duplicate {field_name} values: {duplicates}")


def _require_object_store(recipe: KnowledgeRecipe) -> ObjectStoreProtocol:
    object_store = recipe.stores.objects
    if not isinstance(object_store, ObjectStoreProtocol):
        raise TypeError("recipe.stores.objects must satisfy ObjectStoreProtocol")
    return object_store


def _validate_query_modes(query_modes: tuple[str, ...]) -> None:
    if not query_modes:
        raise ValueError("query_modes must not be empty")
    for mode in query_modes:
        if mode.strip() == "":
            raise ValueError("query mode must not be empty")


def _default_workspace() -> BenchmarkWorkspace:
    root_dir = Path(".heta") / "evaluation"
    return BenchmarkWorkspace(
        root_dir=root_dir,
        cache_dir=root_dir / "cache",
        output_dir=root_dir / "reports",
    )


def _ensure_workspace(workspace: BenchmarkWorkspace) -> None:
    workspace.root_dir.mkdir(parents=True, exist_ok=True)
    workspace.cache_dir.mkdir(parents=True, exist_ok=True)
    if workspace.output_dir is not None:
        workspace.output_dir.mkdir(parents=True, exist_ok=True)


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "unit"


__all__ = [
    "BenchmarkRunConfig",
    "BenchmarkRunResult",
    "BenchmarkRunner",
]
