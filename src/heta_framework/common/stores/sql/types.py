"""Data types for SQL stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SQLParameters = dict[str, Any]
SQLRow = dict[str, Any]


@dataclass(frozen=True)
class SQLResult:
    """Result metadata for a SQL statement."""

    rowcount: int
