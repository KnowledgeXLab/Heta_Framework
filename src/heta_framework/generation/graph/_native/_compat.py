# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 OmegaDesigner contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library gaps between the Python versions this package supports.

`tomllib` entered the standard library in Python 3.11. This package declares ``>=3.10``, so on
3.10 the ``tomli`` backport is used instead (declared as a conditional dependency in
``pyproject.toml``). Import it from here and nowhere else -- ``tests/test_compat.py`` fails if a
module reaches for ``tomllib`` directly, because that import only breaks on the oldest supported
interpreter, which is exactly the version nobody develops on.
"""

from __future__ import annotations

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only reachable on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = ["tomllib"]
