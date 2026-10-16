# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor Oracle/NOP runtime validation, re-implemented from Omega validation.py:335-387.

Harbor execution is a *vendored external runtime contract*: the ``harbor run``
invocation, the ``result.json`` shape, and the reward semantics are owned by
Terminal-Bench-Science, not by this package. Docker/Harbor execution is
intentionally *outside* the Data Designer generation loop — nothing here runs
during dataset generation. It is an explicit, opt-in, fail-closed verification
step that a caller invokes after a package has been generated, and a missing
``harbor`` binary (or a machine without a working Docker daemon) degrades to a
failed outcome rather than an exception.

The scientific claim being reproduced is narrow and non-negotiable:

* Oracle x ``oracle_repeats`` must each complete exactly one trial with reward
  exactly ``1.0`` (the reference solution solves the task).
* NOP x 1 must complete exactly one trial with reward exactly ``0.0`` (the
  initial state does not already solve the task).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from heta_framework.generation.graph._native.domain.process import run_cancellable
from heta_framework.generation.graph._native.io_utils import atomic_write_text, read_json

#: Omega ``validation.py:351`` uses a generous wall-clock budget for a full trial.
HARBOR_TIMEOUT_SEC: int = 86_400
_HARBOR_MISSING_RETURNCODE: int = 127
_TIMEOUT_RETURNCODE: int = 124
DEFAULT_JOB_PREFIX: str = "omega"


class HarborRunResult(BaseModel):
    """One ``harbor run`` invocation plus the trial evidence it produced."""

    agent: str
    index: int
    returncode: int
    completed: bool
    reward: float | None
    passed: bool
    log_path: str
    result_dir: str
    error: str = ""


def _is_binary_reward(value: Any) -> bool:
    """Omega ``validation.py:365``: numeric (``bool`` excluded) and in ``{0, 1}``."""
    return type(value) in (int, float) and value in (0, 1)


def parse_harbor_outputs(output_dir: Path) -> tuple[int, float | None, list[str]]:
    """Reproduce Omega ``validation.py:355-366`` trial parsing.

    Recursively finds every ``result.json`` containing ``verifier_result``. A run
    counts as completed only when there is *exactly one* such trial, it carries no
    ``exception_info``, and every reward is numerically ``0`` or ``1``. Returns
    ``(trial_count, mean_reward, errors)`` where ``mean_reward`` is ``None``
    whenever the run did not complete. Never raises.
    """
    errors: list[str] = []
    root = Path(output_dir)
    trials: list[dict[str, Any]] = []
    try:
        candidates = sorted(root.rglob("result.json"))
    except OSError as error:
        return 0, None, [f"could not scan {root}: {type(error).__name__}: {error}"]

    for result_path in candidates:
        try:
            trial = read_json(result_path)
        except (OSError, ValueError) as error:
            errors.append(f"unreadable result.json {result_path}: {type(error).__name__}: {error}")
            continue
        if isinstance(trial, dict) and "verifier_result" in trial:
            trials.append(trial)

    exceptions = [trial["exception_info"] for trial in trials if trial.get("exception_info")]
    for exception in exceptions:
        errors.append(f"trial exception_info: {exception!r}")

    rewards = [(trial.get("verifier_result") or {}).get("rewards", {}).get("reward") for trial in trials]
    completed = len(trials) == 1 and not exceptions and all(_is_binary_reward(value) for value in rewards)
    if not completed:
        if not trials:
            errors.append(f"no trial with verifier_result found under {root}")
        elif len(trials) != 1:
            errors.append(f"expected exactly one trial, found {len(trials)} under {root}")
        elif not all(_is_binary_reward(value) for value in rewards):
            errors.append(f"rewards are not binary 0/1: {rewards!r}")
        return len(trials), None, errors

    return len(trials), sum(rewards) / len(rewards), errors


def _expected_reward(agent: str) -> float | None:
    """Oracle must score exactly 1.0; NOP must score exactly 0.0."""
    if agent == "oracle":
        return 1.0
    if agent == "nop":
        return 0.0
    return None


def harbor_version() -> str:
    """`harbor --version` (or "" when it cannot be determined).

    Part of the validation cache key: a Harbor upgrade can change how a package scores, so a
    cached verdict from another version must not be reused.
    """
    result = run_cancellable(["harbor", "--version"], timeout=60)
    if result.returncode != 0:
        return ""
    first = result.output.strip().splitlines()
    return first[0].strip() if first else ""


def _run_once(
    package_dir: Path,
    *,
    harbor_env: str,
    agent: str,
    index: int,
    run_output: Path,
    job_name: str,
    log_path: Path,
    cancelled: Callable[[], bool] | None = None,
) -> HarborRunResult:
    """Execute a single ``harbor run``; every failure mode is a result, not a raise."""
    command = [
        sys.executable,
        "-c",
        "from harbor.cli.main import app; app()",
        "run",
        "-p",
        str(package_dir),
        "--agent",
        agent,
        "--env",
        ("heta_framework.generation.derivation.docker:DerivationDockerEnvironment"
         if harbor_env == "docker" else harbor_env),
        "-o",
        str(run_output),
        "--job-name",
        job_name,
    ]
    outcome = run_cancellable(command, cwd=package_dir, timeout=HARBOR_TIMEOUT_SEC, cancelled=cancelled)
    returncode = outcome.returncode
    combined = outcome.output
    if outcome.cancelled:
        error = "harbor run cancelled with the dataset generation run"
    elif outcome.timed_out:
        error = f"harbor run timed out after {HARBOR_TIMEOUT_SEC}s"
    else:
        error = "" if returncode == 0 else f"harbor run exited with returncode {returncode}"
        if outcome.error:
            error = outcome.error

    if combined:
        combined = combined if combined.endswith("\n") else combined + "\n"
    try:
        atomic_write_text(log_path, combined)
    except OSError as error_write:
        error = (error + "; " if error else "") + f"log write failed: {type(error_write).__name__}: {error_write}"

    if returncode == _HARBOR_MISSING_RETURNCODE and not error:
        error = "harbor executable was not found on PATH"

    trials, reward, parse_errors = parse_harbor_outputs(run_output)
    completed = trials == 1 and reward is not None
    if parse_errors and not completed:
        # Diagnostics are only attached to runs that are actually incomplete; a
        # stray unreadable result.json next to a valid trial is not fatal.
        error = "; ".join(filter(None, [error, *parse_errors]))
    expected = _expected_reward(agent)
    passed = returncode == 0 and completed and (expected is None or reward == expected)
    return HarborRunResult(
        agent=agent,
        index=index,
        returncode=returncode,
        completed=completed,
        reward=reward,
        passed=passed,
        log_path=str(log_path),
        result_dir=str(run_output),
        error=error,
    )


def run_harbor(
    package_dir: Path,
    *,
    harbor_env: str,
    agent: str,
    repeats: int,
    output_dir: Path,
    job_prefix: str = DEFAULT_JOB_PREFIX,
    cancelled: Callable[[], bool] | None = None,
) -> list[HarborRunResult]:
    """Run ``harbor run`` serially ``repeats`` times and parse each trial's evidence.

    Mirrors Omega's command construction
    (``harbor run -p <pkg> --agent <agent> --env <env> -o <dir> --job-name <name>``)
    and its ``--job-name omega-<agent>-<index>`` naming. Each run gets its own
    output directory under ``output_dir`` and its own log file. When ``harbor`` is
    absent from ``PATH`` no process is started: every result is a failed outcome
    carrying an explanatory ``error``.
    """
    package = Path(package_dir)
    root = Path(output_dir)
    runs: list[HarborRunResult] = []
    harbor_missing = importlib.util.find_spec("harbor") is None

    for index in range(1, repeats + 1):
        job_name = f"{job_prefix}-{agent}-{index}"
        # A FRESH directory per run: harbor nests <job-name>/ under -o and we parse the
        # run root recursively, so reusing a directory would mix trials from earlier runs
        # and trip the "expected exactly one trial" guard.
        run_output = root / job_name
        run_output.mkdir(parents=True, exist_ok=True)
        log_path = root / f"{job_prefix}-{agent}-{index}.log"
        if harbor_missing:
            message = "harbor executable was not found on PATH"
            try:
                atomic_write_text(log_path, message + "\n")
            except OSError:
                pass
            runs.append(
                HarborRunResult(
                    agent=agent,
                    index=index,
                    returncode=_HARBOR_MISSING_RETURNCODE,
                    completed=False,
                    reward=None,
                    passed=False,
                    log_path=str(log_path),
                    result_dir=str(run_output),
                    error=message,
                )
            )
            continue
        try:
            run_output.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            message = f"could not create harbor output dir {run_output}: {type(error).__name__}: {error}"
            runs.append(
                HarborRunResult(
                    agent=agent,
                    index=index,
                    returncode=_HARBOR_MISSING_RETURNCODE,
                    completed=False,
                    reward=None,
                    passed=False,
                    log_path=str(log_path),
                    result_dir=str(run_output),
                    error=message,
                )
            )
            continue
        runs.append(
            _run_once(
                package,
                harbor_env=harbor_env,
                agent=agent,
                index=index,
                run_output=run_output,
                job_name=job_name,
                log_path=log_path,
                cancelled=cancelled,
            )
        )
    return runs


def _agent_summary(
    agent: str,
    runs: list[HarborRunResult],
    expected_reward: float,
    expected_runs: int,
) -> dict[str, Any]:
    passed = len(runs) == expected_runs and all(
        run.returncode == 0 and run.completed and run.reward == expected_reward for run in runs
    )
    return {
        "requested": True,
        "agent": agent,
        "passed": passed,
        "expected_reward": expected_reward,
        "runs": [run.model_dump(mode="json") for run in runs],
    }


def validate_package_harbor(
    package_dir: Path,
    *,
    harbor_env: str = "docker",
    oracle_repeats: int = 2,
    output_root: Path | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Fail-closed Oracle/NOP runtime validation; never raises.

    Oracle x ``oracle_repeats`` must each exit ``0`` with reward *exactly* ``1.0``,
    and NOP x 1 must exit ``0`` with reward *exactly* ``0.0``. Any deviation —
    including a missing ``harbor`` binary, an incomplete trial, or a non-binary
    reward — yields ``passed=False`` with the concrete errors attached.
    """
    package = Path(package_dir)
    root = Path(output_root) if output_root is not None else package / "authoring" / "validation" / "harbor"
    errors: list[str] = []
    oracle_runs: list[HarborRunResult] = []
    nop_runs: list[HarborRunResult] = []
    try:
        repeats = max(1, int(oracle_repeats))
        oracle_runs = run_harbor(
            package, harbor_env=harbor_env, agent="oracle", repeats=repeats, output_dir=root, cancelled=cancelled
        )
        nop_runs = run_harbor(
            package, harbor_env=harbor_env, agent="nop", repeats=1, output_dir=root, cancelled=cancelled
        )
    except Exception as error:  # noqa: BLE001 - fail-closed boundary, must never raise
        errors.append(f"harbor validation failed to run: {type(error).__name__}: {error}")

    oracle = _agent_summary("oracle", oracle_runs, expected_reward=1.0, expected_runs=max(1, int(oracle_repeats)))
    nop = _agent_summary("nop", nop_runs, expected_reward=0.0, expected_runs=1)
    for summary in (oracle, nop):
        for run in summary["runs"]:
            if run["error"]:
                errors.append(f"{run['agent']}-{run['index']}: {run['error']}")
        if not summary["passed"] and not summary["runs"]:
            errors.append(f"{summary['agent']}: no runs were executed")
    if not oracle["passed"] and oracle["runs"]:
        errors.append("oracle did not complete every repeat with reward exactly 1.0")
    if not nop["passed"] and nop["runs"]:
        errors.append("nop did not complete with reward exactly 0.0")
    return {
        "requested": True,
        "passed": bool(oracle["passed"] and nop["passed"]),
        "oracle": oracle,
        "nop": nop,
        "errors": errors,
    }
