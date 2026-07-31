"""Shared helpers for universal graph steps."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, Literal

from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import GraphStoreProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation


RAGAdapterTarget = Literal["graphrag", "lightrag", "hirag", "leanrag"]


def _entities_by_chunk(
    entities: Iterable[ExtractedEntity],
) -> dict[str, tuple[ExtractedEntity, ...]]:
    grouped: dict[str, list[ExtractedEntity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.chunk_id].append(entity)
    return {key: tuple(value) for key, value in grouped.items()}


def _relations_by_chunk(
    relations: Iterable[ExtractedRelation],
) -> dict[str, tuple[ExtractedRelation, ...]]:
    grouped: dict[str, list[ExtractedRelation]] = defaultdict(list)
    for relation in relations:
        grouped[relation.chunk_id].append(relation)
    return {key: tuple(value) for key, value in grouped.items()}


def _relation_weight(relation: ExtractedRelation) -> float:
    for key in ("weight", "strength", "relationship_strength"):
        value = relation.attributes.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return 1.0


def _most_common(values: Iterable[str]) -> str:
    counts = Counter(value.strip() for value in values if value.strip())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _safe_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return _json(value).encode("utf-8")


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_graph_store(component: object) -> GraphStoreProtocol:
    if not isinstance(component, GraphStoreProtocol):
        raise TypeError("stores.graph must satisfy GraphStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component
