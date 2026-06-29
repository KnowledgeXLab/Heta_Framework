"""In-memory full-text index store."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from heta_framework.common.stores.text_index.types import (
    TextIndexConfig,
    TextIndexRecord,
    TextQuery,
    TextSearchResult,
)

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass
class _Index:
    config: TextIndexConfig
    records: dict[str, TextIndexRecord] = field(default_factory=dict)


class InMemoryTextIndexStore:
    """BM25-ranked in-memory text index for tests and local pipelines."""

    def __init__(self) -> None:
        self._indexes: dict[str, _Index] = {}

    async def create_index(self, config: TextIndexConfig) -> None:
        """Create an index if it does not already exist."""
        self._indexes.setdefault(config.name, _Index(config=config))

    async def drop_index(self, name: str) -> None:
        """Drop an index if it exists."""
        self._indexes.pop(name, None)

    async def upsert(self, index: str, records: Sequence[TextIndexRecord]) -> None:
        """Insert or update text records."""
        target = self._get_index(index)
        for record in records:
            target.records[record.id] = record

    async def search(self, index: str, query: TextQuery) -> list[TextSearchResult]:
        """Search one full-text index using BM25 scoring."""
        target = self._get_index(index)
        query_terms = _tokenize(query.text)
        if not query_terms:
            return []

        records = [
            record
            for record in target.records.values()
            if _matches_filter(record.metadata, query.filters)
        ]
        scores = _bm25_scores(records, query_terms)
        results = [
            TextSearchResult(
                id=record.id,
                text=record.text,
                score=score,
                metadata=record.metadata,
            )
            for record, score in scores
            if score > 0
        ]
        results.sort(key=lambda item: (-item.score, item.id))
        return results[: query.top_k]

    async def delete(self, index: str, ids: Sequence[str]) -> None:
        """Delete records by id."""
        target = self._get_index(index)
        for record_id in ids:
            target.records.pop(record_id, None)

    async def count(self, index: str) -> int:
        """Return the number of records in an index."""
        return len(self._get_index(index).records)

    async def aclose(self) -> None:
        """Release resources held by the store."""

    def _get_index(self, name: str) -> _Index:
        try:
            return self._indexes[name]
        except KeyError as exc:
            raise ValueError(f"text index does not exist: {name}") from exc


def _bm25_scores(
    records: Sequence[TextIndexRecord],
    query_terms: list[str],
) -> list[tuple[TextIndexRecord, float]]:
    if not records:
        return []
    tokenized = [_tokenize(record.text) for record in records]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    total_docs = len(records)
    scores: list[tuple[TextIndexRecord, float]] = []
    for record, tokens, length in zip(records, tokenized, lengths, strict=True):
        term_frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            score += _bm25_term_score(
                frequency=frequency,
                document_frequency=document_frequency[term],
                document_count=total_docs,
                document_length=length,
                average_document_length=average_length,
            )
        scores.append((record, score))
    return scores


def _bm25_term_score(
    *,
    frequency: int,
    document_frequency: int,
    document_count: int,
    document_length: int,
    average_document_length: float,
) -> float:
    k1 = 1.5
    b = 0.75
    idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
    denominator = frequency + k1 * (
        1 - b + b * document_length / (average_document_length or 1.0)
    )
    return idf * frequency * (k1 + 1) / denominator


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _matches_filter(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())
