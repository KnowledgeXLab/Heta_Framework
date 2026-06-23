"""Ranking helpers shared by built-in query engines."""

from __future__ import annotations

from dataclasses import replace

from heta_framework.kb.search.types import QueryResponse, QueryResult


def reciprocal_rank_fusion(
    responses: list[QueryResponse],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> tuple[QueryResult, ...]:
    """Fuse ranked query responses with reciprocal rank fusion."""
    scores: dict[tuple[str, str], float] = {}
    best_results: dict[tuple[str, str], QueryResult] = {}
    source_modes: dict[tuple[str, str], list[str]] = {}

    for response in responses:
        for rank, result in enumerate(response.results, start=1):
            key = _result_key(result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            source_modes.setdefault(key, []).append(response.mode)
            existing = best_results.get(key)
            if existing is None or _result_score(result) > _result_score(existing):
                best_results[key] = result

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    if top_k is not None:
        ordered_keys = ordered_keys[:top_k]

    return tuple(
        replace(
            best_results[key],
            score=scores[key],
            metadata={
                **dict(best_results[key].metadata),
                "rrf_score": scores[key],
                "retrieval_modes": tuple(dict.fromkeys(source_modes[key])),
            },
        )
        for key in ordered_keys
    )


def weighted_reciprocal_rank_fusion(
    responses: list[QueryResponse],
    *,
    weights: dict[str, float],
    k: int = 60,
    top_k: int | None = None,
) -> tuple[QueryResult, ...]:
    """Fuse ranked responses with per-mode weights and rank-only scores."""
    scores: dict[tuple[str, str], float] = {}
    best_results: dict[tuple[str, str], QueryResult] = {}
    source_modes: dict[tuple[str, str], list[str]] = {}

    for response in responses:
        weight = weights.get(response.mode, 1.0)
        if weight <= 0:
            continue
        for rank, result in enumerate(response.results, start=1):
            key = _result_key(result)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            source_modes.setdefault(key, []).append(response.mode)
            existing = best_results.get(key)
            if existing is None or _result_score(result) > _result_score(existing):
                best_results[key] = result

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    if top_k is not None:
        ordered_keys = ordered_keys[:top_k]

    return tuple(
        replace(
            best_results[key],
            score=scores[key],
            metadata={
                **dict(best_results[key].metadata),
                "hybrid_score": scores[key],
                "retrieval_modes": tuple(dict.fromkeys(source_modes[key])),
                "fusion": "weighted_rrf",
            },
        )
        for key in ordered_keys
    )


def result_score(result: QueryResult) -> float:
    """Return a sortable score for a query result."""
    return _result_score(result)


def deduplicate_results(results: list[QueryResult]) -> tuple[QueryResult, ...]:
    """Deduplicate results by kind and id, keeping the highest-scored item."""
    best: dict[tuple[str, str], QueryResult] = {}
    for result in results:
        key = _result_key(result)
        existing = best.get(key)
        if existing is None or _result_score(result) > _result_score(existing):
            best[key] = result
    return tuple(sorted(best.values(), key=_result_score, reverse=True))


def _result_key(result: QueryResult) -> tuple[str, str]:
    return result.kind, result.id


def _result_score(result: QueryResult) -> float:
    return result.score if result.score is not None else 0.0
