# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TBS per-task static contract checks, re-implemented from Omega validation.py:154-201.

The 24 shell scripts in :data:`STATIC_CHECKS` are a *vendored external contract*:
they are Terminal-Bench-Science's own per-task validators, copied verbatim into
``omegadesigner/assets/checks`` together with their upstream LICENSE. This module
never reinterprets what a check means — it only stages an isolated copy of a
generated package and executes each script exactly the way Omega does
(``bash <check> <relative_task_path>`` with ``cwd`` set to the staging root), so a
package that passes here would pass the same upstream gate.

Docker and Harbor execution are intentionally *outside* the Data Designer
generation loop: this module only runs host-side ``bash`` checks. Container
runtime evidence lives in :mod:`omegadesigner.domain.harbor` and is invoked by an
explicit caller, never as a side effect of dataset generation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from heta_framework.generation.graph._native.domain.schemas import KEBAB

# Omega ``validation.py:20`` — order is part of the contract, not cosmetic.
STATIC_CHECKS: tuple[str, ...] = (
    "check-canary.sh",
    "check-dockerfile-references.sh",
    "check-dockerfile-sanity.sh",
    "check-dockerfile-platform.sh",
    "check-task-absolute-path.sh",
    "check-test-file-references.sh",
    "check-test-sh-sanity.sh",
    "check-task-fields.sh",
    "check-task-path.sh",
    "check-task-structure.sh",
    "check-task-timeout.sh",
    "check-instruction-suffix.sh",
    "check-gpu-types.sh",
    "check-allow-internet.sh",
    "check-task-slug.sh",
    "check-separate-verifier.sh",
    "check-verifier-tooling-baked.sh",
    "check-trial-network-fetch.sh",
    "check-pip-pinning.sh",
    "check-pytest-version.sh",
    "check-nproc.sh",
    "check-compose-host-binds.sh",
    "check-binary-reward.sh",
    "check-task-readme.sh",
)

# Package-local vendored assets. Kept as module constants so callers can pass them
# explicitly (the frozen signatures stay explicit) without recomputing paths.
_ASSETS_ROOT: Path = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_CHECKS_DIR: Path = _ASSETS_ROOT / "checks"
DEFAULT_REVIEWER_POOL: Path = _ASSETS_ROOT / "tbs_reference" / "reviewer-pool.yml"

#: Staged packages are always written under this root, matching TBS layout.
TASKS_ROOT_NAME: str = "tasks"
FALLBACK_SLUG: str = "generated-task"
_INVALID_SEGMENT: str = "invalid"
_ERROR_TAIL_LINES: int = 40
_MISSING_SCRIPT_RETURNCODE: int = 127
_TIMEOUT_RETURNCODE: int = 124


class StaticCheckOutcome(BaseModel):
    """Result of one vendored TBS check executed against a staged package copy."""

    name: str
    passed: bool
    returncode: int
    log_path: str
    warnings: list[str] = Field(default_factory=list)
    error_tail: str = ""


def _error_tail(text: str, *, lines: int = _ERROR_TAIL_LINES) -> str:
    """Last ``lines`` lines of ``text``; used as bounded failure evidence."""
    stripped = text.rstrip("\n")
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])


def sanitize_slug(value: str | None) -> str:
    """Kebab-case a task slug, falling back to ``generated-task`` (Omega:167-168)."""
    slug = str(value or "").rsplit("/", 1)[-1]
    return slug if KEBAB.fullmatch(slug) else FALLBACK_SLUG


def _path_segment(value: object) -> str:
    """Omega ``validation.py:173-175``: normalize a TBS domain/field path segment."""
    segment = "-".join(str(value or "unspecified").lower().split())
    return segment if KEBAB.fullmatch(segment) else _INVALID_SEGMENT


def stage_package(
    package_dir: Path,
    staging_root: Path,
    *,
    slug: str,
    domain: str,
    field: str,
    reviewer_pool: Path,
) -> Path:
    """Reproduce Omega ``validation.py:154-201`` staging, returning the staged task path.

    A private copy is used because several upstream checks call ``find`` without
    ``-H`` (a symlink would silently skip nested files, including canary checks)
    and because no check may mutate the real draft. Authoring/validation output is
    removed from the copy: keeping it makes later runs scan Harbor logs as task
    files, notably causing false canary failures.
    """
    root = Path(staging_root)
    relative = Path(TASKS_ROOT_NAME) / _path_segment(domain) / _path_segment(field) / sanitize_slug(slug)
    linked = root / relative
    (root / ".github").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(reviewer_pool), root / ".github" / "reviewer-pool.yml")
    if linked.exists():
        # staging_root may be caller-owned and reused between runs; the staged copy
        # must be exact, so a stale tree is replaced rather than merged.
        shutil.rmtree(linked)
    linked.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(package_dir), linked)
    shutil.rmtree(linked / "authoring", ignore_errors=True)
    return linked


def _check_env() -> dict[str, str]:
    """PATH-prefixed environment matching Omega ``validation.py:190-191``."""
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def _staging_failure_outcomes(staging_root: Path, reason: str) -> list[StaticCheckOutcome]:
    """Fail-closed outcomes when staging itself fails; never raises."""
    logs = Path(staging_root) / "logs"
    outcomes: list[StaticCheckOutcome] = []
    for check in STATIC_CHECKS:
        log_path = logs / f"static-{check}.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(reason + "\n", encoding="utf-8")
        except OSError:
            pass
        outcomes.append(
            StaticCheckOutcome(
                name=check,
                passed=False,
                returncode=_MISSING_SCRIPT_RETURNCODE,
                log_path=str(log_path),
                warnings=[],
                error_tail=reason,
            )
        )
    return outcomes


def run_static_checks(
    package_dir: Path,
    *,
    staging_root: Path,
    checks_dir: Path,
    reviewer_pool: Path,
    slug: str,
    domain: str,
    field: str,
    timeout_sec: int = 300,
) -> list[StaticCheckOutcome]:
    """Stage a copy of ``package_dir`` and run every vendored TBS check against it.

    Each check is executed as ``bash <checks_dir>/<check> <relative_task_path>``
    with ``cwd`` set to the staging root. Per-check output is written to
    ``<staging_root>/logs/static-<check>.log`` — deliberately outside the staged
    ``tasks/`` tree so validation output cannot be scanned as task files.

    Robustness is part of the contract: a missing script, a timeout, a staging
    failure, or an ``OSError`` becomes a failed outcome with an explanatory
    ``error_tail``. This function never raises and never returns fewer than
    ``len(STATIC_CHECKS)`` outcomes.
    """
    try:
        staged = stage_package(
            Path(package_dir),
            Path(staging_root),
            slug=slug,
            domain=domain,
            field=field,
            reviewer_pool=Path(reviewer_pool),
        )
    except OSError as error:
        return _staging_failure_outcomes(staging_root, f"staging failed: {type(error).__name__}: {error}")

    relative_task_path = staged.relative_to(Path(staging_root)).as_posix()
    checks_root = Path(checks_dir)
    logs = Path(staging_root) / "logs"
    env = _check_env()
    outcomes: list[StaticCheckOutcome] = []

    for check in STATIC_CHECKS:
        log_path = logs / f"static-{check}.log"
        script = checks_root / check
        warnings: list[str] = []
        error_tail = ""
        if not script.is_file():
            returncode = _MISSING_SCRIPT_RETURNCODE
            combined = f"check script not found: {script}\n"
        else:
            command = ["bash", str(script), relative_task_path]
            try:
                result = subprocess.run(
                    command,
                    cwd=str(staging_root),
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                    env=env,
                )
                returncode = result.returncode
                combined = (result.stdout or "") + (result.stderr or "")
            except subprocess.TimeoutExpired as error:

                def _decode(value: object) -> str:
                    if isinstance(value, bytes):
                        return value.decode("utf-8", errors="replace")
                    return value if isinstance(value, str) else ""

                returncode = _TIMEOUT_RETURNCODE
                combined = _decode(error.stdout) + _decode(error.stderr) + (f"\ntimeout: check exceeded {timeout_sec}s")
            except OSError as error:
                returncode = _MISSING_SCRIPT_RETURNCODE
                combined = f"{type(error).__name__}: {error}\n"

        passed = returncode == 0
        warnings = [line for line in combined.splitlines() if line.startswith("WARN ")]
        if not passed:
            error_tail = _error_tail(combined) or f"check failed with returncode {returncode}"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(combined, encoding="utf-8")
        except OSError as error:
            # A log write failure must not mask the check result.
            error_tail = (error_tail + "\n" if error_tail else "") + (
                f"log write failed: {type(error).__name__}: {error}"
            )
        outcomes.append(
            StaticCheckOutcome(
                name=check,
                passed=passed,
                returncode=returncode,
                log_path=str(log_path),
                warnings=warnings,
                error_tail=error_tail,
            )
        )
    return outcomes


def summarize_static(outcomes: list[StaticCheckOutcome]) -> dict:
    """Roll a run up to ``{"passed", "attempted", "failed", "warnings"}``.

    ``passed`` is fail-closed: an empty outcome list is not a pass. ``warnings``
    maps every attempted check name to its collected ``WARN `` lines (possibly an
    empty list), so callers can look a check up without a membership test.
    """
    attempted = len(outcomes)
    failed = [outcome.name for outcome in outcomes if not outcome.passed]
    return {
        "passed": bool(attempted) and not failed,
        "attempted": attempted,
        "failed": failed,
        "warnings": {outcome.name: list(outcome.warnings) for outcome in outcomes},
    }
