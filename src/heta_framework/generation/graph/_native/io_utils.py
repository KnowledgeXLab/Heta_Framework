# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small IO/hash helpers shared by the POC.

Deliberately dependency-light: this package is a re-implementation, not a fork,
so nothing here imports ``omega``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any


def normalize_title(value: str) -> str:
    """NFKC + casefold + collapse every non-alphanumeric run to a single space.

    Lives here rather than in :mod:`omegadesigner.sources` because that module imports pandas:
    the search side pairs KG files with PDFs by title and must stay importable in a minimal
    environment (``pip install omegadesigner`` without the generation extras).
    """
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded).split())


def title_tokens(value: str) -> set[str]:
    return set(normalize_title(value).split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}."
    )
    try:
        with handle:
            handle.write(content)
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def tree_hashes(root: str | Path) -> dict[str, str]:
    """File-relative-path -> sha256 for every regular file under ``root``."""
    root_path = Path(root)
    return {
        str(path.relative_to(root_path)): sha256_file(path) for path in sorted(root_path.rglob("*")) if path.is_file()
    }


# --------------------------------------------------------------------------------------
# values read back from a dataset
# --------------------------------------------------------------------------------------
#
# A value that crosses a stage boundary comes back through parquet/pandas, so it is not the
# plain Python object that was written: a list column is a numpy array, a struct column a dict
# of arrays, a bool a numpy scalar. `value or []` is not safe on any of those --
# `bool(empty_array)` raises "The truth value of an empty array is ambiguous" -- and Data
# Designer drops the row when a column raises. Two real runs were lost that way, so every read
# goes through these.


def _unwrap_scalar(value: Any) -> Any:
    """numpy scalar -> Python scalar; everything else unchanged."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def as_list(value: Any) -> list:
    """Coerce a dataset value into a plain list of plain Python scalars.

    ``None``, a scalar, a numpy array and JSON text all end up as a list; numpy scalars are
    unwrapped so the result survives ``json.dumps`` and prints as ``1.0`` rather than
    ``np.float64(1.0)``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return as_list(json.loads(value))
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, dict):
        return [_unwrap_scalar(item) for item in value.values()]
    if hasattr(value, "item") and not hasattr(value, "__iter__"):
        return [_unwrap_scalar(value)]
    try:
        return [_unwrap_scalar(item) for item in value]
    except TypeError:
        return [_unwrap_scalar(value)]


def as_json_object(value: Any) -> dict:
    """Coerce a dataset value into a JSON-friendly dict (struct column or JSON text)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): value[key] for key in value}


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    """Float, with ``NaN`` treated as "not measured" so it cannot reach ``json.dumps``.

    A numeric column whose producer was skipped comes back from pandas as ``NaN``; serializing
    that emits bare ``NaN``, which is not valid JSON.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN check without importing math


def as_gate(value: Any) -> bool:
    """A gate is a real boolean: numpy scalars unwrap, skip markers and arrays do not pass."""
    if isinstance(value, (dict, list, tuple, set)):
        return False
    if hasattr(value, "item"):  # numpy scalar
        try:
            value = value.item()
        except (TypeError, ValueError):
            return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False


def as_bool_or_none(value: Any) -> bool | None:
    """Tri-state boolean: ``None`` means "not measured", which is not the same as False."""
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        return None
    unwrapped = value.item() if hasattr(value, "item") else value
    if isinstance(unwrapped, bool):
        return unwrapped
    if isinstance(unwrapped, (int, float)):
        return bool(unwrapped)
    return None


def as_text(value: Any) -> str:
    """Coerce a dataset scalar into text without letting an array leak into a path or a name."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)) or hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            return ""
        if isinstance(value, (dict, list, tuple, set)):
            return ""
    return str(value)
