"""Reusable retrieval evaluators for benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any, Literal, Mapping

from heta_framework.evaluation.protocols import BenchmarkEvaluatorProtocol
from heta_framework.evaluation.types import BenchmarkCase, BenchmarkEvidence, EvaluationScore
from heta_framework.kb.search import QueryResponse, QueryResult


@dataclass(frozen=True)
class EvidenceRecallAtK:
    """Recall of expected evidence against the top-k query results."""

    k: int = 5

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be greater than zero")

    @property
    def name(self) -> str:
        """Return the stable evaluator name."""
        return f"evidence_recall@{self.k}"

    async def evaluate(
        self,
        *,
        case: BenchmarkCase,
        response: QueryResponse,
    ) -> EvaluationScore:
        """Score top-k evidence recall for one response."""
        expected = case.expected.evidence
        if not expected:
            return EvaluationScore(
                name=self.name,
                value=0.0,
                metadata={"skipped": True, "reason": "case has no expected evidence"},
            )

        results = response.results[: self.k]
        matched = tuple(
            evidence
            for evidence in expected
            if any(_matches_evidence(evidence, result) for result in results)
        )
        return EvaluationScore(
            name=self.name,
            value=len(matched) / len(expected),
            metadata={
                "matched": len(matched),
                "expected": len(expected),
                "matched_reference_ids": [
                    evidence.reference_id for evidence in matched if evidence.reference_id
                ],
            },
        )


BeirMetricName = Literal["ndcg", "map", "recall", "precision", "mrr"]


@dataclass(frozen=True)
class BeirRetrievalMetric:
    """Standard BEIR retrieval metric for one k value.

    Heta query engines return chunks, while BEIR qrels are document-level labels.
    The evaluator therefore maps each result back to a benchmark document id and
    deduplicates documents before scoring.
    """

    metric: BeirMetricName
    k: int = 10

    def __post_init__(self) -> None:
        if self.metric not in {"ndcg", "map", "recall", "precision", "mrr"}:
            raise ValueError("metric must be one of: ndcg, map, recall, precision, mrr")
        if self.k <= 0:
            raise ValueError("k must be greater than zero")

    @property
    def name(self) -> str:
        """Return the stable evaluator name."""
        return f"beir_{self.metric}@{self.k}"

    async def evaluate(
        self,
        *,
        case: BenchmarkCase,
        response: QueryResponse,
    ) -> EvaluationScore:
        """Score one query response against BEIR-style qrels."""
        qrels = _qrels(case)
        if not qrels:
            return EvaluationScore(
                name=self.name,
                value=0.0,
                metadata={"skipped": True, "reason": "case has no BEIR qrels"},
            )
        ranked_doc_ids = _ranked_document_ids(response.results, k=self.k)
        value = _score_beir_metric(self.metric, qrels, ranked_doc_ids, self.k)
        return EvaluationScore(
            name=self.name,
            value=value,
            metadata={
                "relevant_documents": len(qrels),
                "retrieved_documents": len(ranked_doc_ids),
                "matched_documents": [
                    document_id for document_id in ranked_doc_ids if qrels.get(document_id, 0) > 0
                ],
            },
        )


def beir_default_metrics(
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 100),
) -> tuple[BeirRetrievalMetric, ...]:
    """Return the standard BEIR metric set used by the built-in adapter."""
    metrics: list[BeirRetrievalMetric] = []
    for metric in ("ndcg", "map", "recall", "precision", "mrr"):
        metrics.extend(BeirRetrievalMetric(metric=metric, k=k) for k in k_values)
    return tuple(metrics)


def _matches_evidence(evidence: BenchmarkEvidence, result: QueryResult) -> bool:
    if evidence.locator and evidence.text:
        return _locator_matches(evidence.locator, result) and _text_matches(
            evidence.text,
            result.text,
        )
    if evidence.locator:
        return _locator_matches(evidence.locator, result)
    if evidence.text:
        return _text_matches(evidence.text, result.text)
    if evidence.reference_id:
        return _reference_matches(evidence.reference_id, result)
    return False


def _locator_matches(locator: Mapping[str, Any], result: QueryResult) -> bool:
    source = dict(result.source)
    for key, expected in locator.items():
        if key == "source_key":
            actual = source.get("source_key", source.get("object_key"))
        elif key == "source_key_prefix":
            actual = str(source.get("source_key", source.get("object_key", "")))
            if actual.startswith(str(expected)):
                continue
            return False
        elif key == "chunk_id":
            chunk_ids = source.get("chunk_ids", ())
            if expected == result.id or expected in chunk_ids:
                continue
            return False
        else:
            actual = source.get(key)
        if actual != expected:
            return False
    return True


def _reference_matches(reference_id: str, result: QueryResult) -> bool:
    if reference_id == result.id:
        return True
    source = dict(result.source)
    return reference_id in {
        str(source.get("document_id", "")),
        str(source.get("source_key", "")),
        str(source.get("object_key", "")),
    }


def _text_matches(expected_text: str, result_text: str) -> bool:
    expected = _normalize_text(expected_text)
    actual = _normalize_text(result_text)
    return expected in actual or actual in expected


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _qrels(case: BenchmarkCase) -> dict[str, int]:
    qrels: dict[str, int] = {}
    for evidence in case.expected.evidence:
        if not evidence.reference_id:
            continue
        relevance = evidence.metadata.get("relevance", 1)
        try:
            relevance_value = int(relevance)
        except (TypeError, ValueError):
            relevance_value = 1
        if relevance_value > 0:
            qrels[evidence.reference_id] = max(
                relevance_value,
                qrels.get(evidence.reference_id, 0),
            )
    return qrels


def _ranked_document_ids(results: tuple[QueryResult, ...], *, k: int) -> tuple[str, ...]:
    document_ids: list[str] = []
    seen: set[str] = set()
    for result in results:
        for document_id in _beir_document_ids(result):
            if document_id in seen:
                continue
            seen.add(document_id)
            document_ids.append(document_id)
            if len(document_ids) >= k:
                return tuple(document_ids)
    return tuple(document_ids)


def _beir_document_id(result: QueryResult) -> str | None:
    document_ids = _beir_document_ids(result)
    return document_ids[0] if document_ids else None


def _beir_document_ids(result: QueryResult) -> tuple[str, ...]:
    source = dict(result.source)
    document_ids: list[str] = []
    seen: set[str] = set()

    def add(value: object | None) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text not in seen:
            document_ids.append(text)
            seen.add(text)

    for key in ("beir_document_id", "benchmark_document_id"):
        value = source.get(key) or result.metadata.get(key)
        if value is not None and str(value).strip():
            add(value)
    for key in ("beir_document_ids", "benchmark_document_ids"):
        for value in _as_sequence(source.get(key) or result.metadata.get(key)):
            add(value)
    for key in ("source_key", "object_key"):
        parsed = _beir_document_id_from_source_key(source.get(key))
        if parsed is not None:
            add(parsed)
    for key in ("source_keys", "object_keys"):
        for value in _as_sequence(source.get(key)):
            parsed = _beir_document_id_from_source_key(value)
            if parsed is not None:
                add(parsed)
    return tuple(document_ids)


def _beir_document_id_from_source_key(source_key: object | None) -> str | None:
    if source_key is None:
        return None
    parts = [part for part in str(source_key).split("/") if part]
    try:
        index = parts.index("beir")
    except ValueError:
        index = -1
    if index >= 0 and len(parts) > index + 2:
        return parts[index + 2]
    if len(parts) >= 5 and parts[0] == "raw" and parts[1] == "benchmarks":
        return parts[4]
    return None


def _as_sequence(value: object | None) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


def _score_beir_metric(
    metric: BeirMetricName,
    qrels: Mapping[str, int],
    ranked_doc_ids: tuple[str, ...],
    k: int,
) -> float:
    if metric == "ndcg":
        return _ndcg_at_k(qrels, ranked_doc_ids, k)
    if metric == "map":
        return _map_at_k(qrels, ranked_doc_ids, k)
    if metric == "recall":
        return _recall_at_k(qrels, ranked_doc_ids, k)
    if metric == "precision":
        return _precision_at_k(qrels, ranked_doc_ids, k)
    if metric == "mrr":
        return _mrr_at_k(qrels, ranked_doc_ids, k)
    raise AssertionError(f"unsupported BEIR metric: {metric}")


def _ndcg_at_k(qrels: Mapping[str, int], ranked_doc_ids: tuple[str, ...], k: int) -> float:
    dcg = _dcg([qrels.get(document_id, 0) for document_id in ranked_doc_ids[:k]])
    ideal = _dcg(sorted(qrels.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg / ideal


def _dcg(relevances: list[int]) -> float:
    return sum(((2**relevance) - 1) / log2(index + 2) for index, relevance in enumerate(relevances))


def _map_at_k(qrels: Mapping[str, int], ranked_doc_ids: tuple[str, ...], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for index, document_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(document_id, 0) <= 0:
            continue
        hits += 1
        precision_sum += hits / index
    return precision_sum / min(len(qrels), k) if qrels else 0.0


def _recall_at_k(qrels: Mapping[str, int], ranked_doc_ids: tuple[str, ...], k: int) -> float:
    relevant = {document_id for document_id, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def _precision_at_k(qrels: Mapping[str, int], ranked_doc_ids: tuple[str, ...], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for document_id in ranked_doc_ids[:k] if qrels.get(document_id, 0) > 0)
    return hits / k


def _mrr_at_k(qrels: Mapping[str, int], ranked_doc_ids: tuple[str, ...], k: int) -> float:
    for index, document_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(document_id, 0) > 0:
            return 1 / index
    return 0.0


__all__ = [
    "BeirMetricName",
    "BeirRetrievalMetric",
    "BenchmarkEvaluatorProtocol",
    "EvidenceRecallAtK",
    "beir_default_metrics",
]
