"""Prepared evidence bundles, independent of KB connections and native run paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GraphInputError(ValueError):
    """Source evidence cannot satisfy the native input contract."""


@dataclass(frozen=True)
class PreparedGraphInputs:
    manifest_key: str
    document_count: int


@dataclass(frozen=True)
class GraphDocument:
    document_id: str
    title: str
    text: str
    graph: dict[str, Any]
    provenance: dict[str, Any]
