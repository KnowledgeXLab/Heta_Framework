# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime validation columns: Docker materialization, Harbor oracle/nop, final verdict.

Data Designer schedules these like any other column, so the expensive external steps inherit
the run's admission control, its row-group checkpointing and its failure policy. Three things
are deliberate:

* **cell_by_cell, not full_column.** One task per row means a slow or wedged Harbor trial only
  holds its own row; a batch column would also block the write-back of every sibling row in the
  group. (Measured: admission is an ``asyncio.Semaphore`` over row groups, and a group's
  checkpoint only fires once every row in it is finished.)
* **Never raise.** A missing Docker daemon, a failing build, a cancelled trial and a
  non-binary reward are all column values. Raising would count toward the scheduler's
  ``shutdown_error_rate`` and can abort a run whose other packages are perfectly fine.
* **Cancellable processes.** The subprocess helpers poll
  ``data_designer.engine.context.is_run_cancellation_requested`` so cancelling the run tears
  down the Docker/Harbor process group instead of leaving it running behind the run.
"""

from __future__ import annotations

import json
from typing import Any

import data_designer.config as dd

from heta_framework.generation.graph._native.config import Settings
from heta_framework.generation.graph._native.domain import validation_runtime
from heta_framework.generation.graph._native.io_utils import (
    as_bool_or_none,
    as_float,
    as_gate,
    as_int,
    as_json_object,
    as_list,
    as_text,
)


def _cancellation_check() -> Any:
    """Data Designer's run-cancellation predicate, or ``None`` outside a DD run."""
    try:
        from data_designer.engine.context import is_run_cancellation_requested
    except ImportError:  # pragma: no cover - DD always provides it in a run
        return None
    return is_run_cancellation_requested


@dd.custom_column_generator(
    required_columns=["package_dir", "static_passed"],
    side_effect_columns=[
        "materialize_ok",
        "materialize_cache_hit",
        "deterministic",
        "hardening_deviations",
        "materialized_paths",
        "materialize_errors",
    ],
)
def runtime_materialize(row: dict, generator_params: Settings) -> dict:
    """Phase 1 as a column: build the generator image, run it twice, verify reproducibility.

    The heavy output (build/run logs, generated data trees, the image) stays on disk inside the
    package; the row only carries the verdict and the paths, per the project's column-redundancy
    rule.
    """
    settings = generator_params
    package_dir = as_text(row.get("package_dir"))
    row["runtime_materialize"] = "failed"
    row["materialize_ok"] = False
    row["materialize_cache_hit"] = False
    row["deterministic"] = None
    row["hardening_deviations"] = []
    row["materialized_paths"] = []
    row["materialize_errors"] = []
    if not package_dir:
        row["materialize_errors"] = ["no package_dir on this row; nothing to materialize"]
        return row
    if not row.get("static_passed"):
        # Fail closed and keep the cascade as *values*: `harbor_validation` skips on
        # `materialize_ok == false`, which a skip of this column would not trigger.
        row["materialize_errors"] = ["static contract checks failed; materialization not attempted"]
        return row

    def _phase() -> dict:
        return validation_runtime.materialization_phase(
            package_dir,
            oracle_repeats=int(getattr(settings.validation, "oracle_repeats", 2)),
            harbor_env=str(getattr(settings.validation, "harbor_env", "docker")),
            cancelled=_cancellation_check(),
            force=validation_runtime.revalidate_requested(),
        )

    try:
        outcome = _phase()
    except Exception as error:  # noqa: BLE001 - a rejected package must not abort the run
        row["materialize_errors"] = [f"{type(error).__name__}: {error}"]
        return row
    row["materialize_ok"] = bool(outcome["ok"])
    row["materialize_cache_hit"] = bool(outcome.get("cache_hit"))
    row["deterministic"] = outcome.get("deterministic")
    row["hardening_deviations"] = as_list(outcome.get("hardening_deviations"))
    row["materialized_paths"] = as_list(outcome.get("materialized_paths"))
    row["materialize_errors"] = as_list(outcome.get("errors"))
    row["runtime_materialize"] = (
        "ready" if row["materialize_ok"] else ("cached" if row["materialize_cache_hit"] else "failed")
    )
    return row


@dd.custom_column_generator(
    required_columns=["package_dir", "materialize_ok"],
    side_effect_columns=[
        "harbor_passed",
        "harbor_cache_hit",
        "oracle_rewards",
        "nop_reward",
        "harbor_errors",
    ],
)
def harbor_validation(row: dict, generator_params: Settings) -> dict:
    """Phase 2 as a column: Harbor oracle × repeats and nop × 1, judged fail-closed."""
    settings = generator_params
    package_dir = as_text(row.get("package_dir"))
    row["harbor_validation"] = "failed"
    row["harbor_passed"] = False
    row["harbor_cache_hit"] = False
    row["oracle_rewards"] = []
    row["nop_reward"] = None
    row["harbor_errors"] = []
    if not package_dir:
        row["harbor_errors"] = ["no package_dir on this row; nothing to validate"]
        return row

    try:
        outcome = validation_runtime.harbor_phase(
            package_dir,
            oracle_repeats=int(getattr(settings.validation, "oracle_repeats", 2)),
            harbor_env=str(getattr(settings.validation, "harbor_env", "docker")),
            cancelled=_cancellation_check(),
            force=validation_runtime.revalidate_requested(),
        )
    except Exception as error:  # noqa: BLE001 - a failed trial must not abort the run
        row["harbor_errors"] = [f"{type(error).__name__}: {error}"]
        return row
    row["harbor_passed"] = bool(outcome["passed"])
    row["harbor_cache_hit"] = bool(outcome.get("cache_hit"))
    row["oracle_rewards"] = as_list(outcome.get("oracle_rewards"))
    row["nop_reward"] = outcome.get("nop_reward")
    row["harbor_errors"] = as_list(outcome.get("errors"))
    row["harbor_validation"] = "passed" if row["harbor_passed"] else "failed"
    return row


@dd.custom_column_generator(
    required_columns=["package_dir", "static_passed", "static_summary", "materialize_ok", "harbor_passed"],
    side_effect_columns=["terminal_bench_ready", "validation_summary", "validation_report_path"],
)
def terminal_bench_ready(row: dict, generator_params: Settings) -> dict:
    """Compose the three gates into the package's verdict and persist its report.

    This is the only column that writes ``validation_report.json``, so the report always
    describes the same three facts the dataset row does (static, materialized, Harbor).
    """
    settings = generator_params
    return _guarded(
        row,
        ok_column="terminal_bench_ready",
        error_column="validation_summary",
        body=lambda: _compose_verdict(row, settings),
    )


def _guarded(row: dict, *, ok_column: str, error_column: str, body: Any) -> dict:
    """Run a column body with the "a bad row is a value, not an exception" contract enforced.

    Data Designer turns a raised exception into ``CustomColumnGenerationError`` and *drops the
    row*; with a single-row dataset that aborts the whole run. A coercion bug in the verdict
    column did exactly that after both Harbor trials had already passed, so the contract is
    enforced here rather than trusted. The guard writes only the column's own declared fields,
    because an undeclared key would be silently dropped by Data Designer with a warning.
    """
    try:
        return body()
    except Exception as error:  # noqa: BLE001 - the contract of every column in this module
        message = f"{type(error).__name__}: {error}"
        row[ok_column] = False
        row[error_column] = [message]
        if ok_column == "terminal_bench_ready":
            row["terminal_bench_ready"] = False
            row["validation_summary"] = json.dumps({"terminal_bench_ready": False, "errors": [message]})
            row["validation_report_path"] = ""
        return row


def _compose_verdict(row: dict, settings: Settings) -> dict:
    package_dir = as_text(row.get("package_dir"))
    static_summary = as_json_object(row.get("static_summary"))
    static = {
        "passed": as_gate(row.get("static_passed")),
        "failed": as_list(static_summary.get("failed")),
        "attempted": as_int(static_summary.get("attempted")),
    }
    materialize = {
        "ok": as_gate(row.get("materialize_ok")),
        "deterministic": as_bool_or_none(row.get("deterministic")),
        "hardening_deviations": (as_list(row.get("hardening_deviations"))),
        "errors": as_list(row.get("materialize_errors")),
        "materialized_paths": as_list(row.get("materialized_paths")),
        "cache_hit": bool(row.get("materialize_cache_hit")),
    }
    harbor = {
        "passed": as_gate(row.get("harbor_passed")),
        "oracle_rewards": as_list(row.get("oracle_rewards")),
        "nop_reward": as_float(row.get("nop_reward")),
        "errors": as_list(row.get("harbor_errors")),
        "cache_hit": bool(row.get("harbor_cache_hit")),
        # Trial-level evidence lives in the cache next to the package, not in the dataset row.
        "detail": validation_runtime.cached_harbor_detail(package_dir) if package_dir else None,
    }
    report = validation_runtime.build_report(
        package_dir,
        static=static,
        materialize=materialize,
        harbor=harbor,
        settings_domain=settings.task_domain,
        settings_field=settings.task_field,
    )
    summary = validation_runtime.validation_summary(report)
    report_path = ""
    if package_dir:
        try:
            report_path = str(validation_runtime.write_report(package_dir, report))
        except OSError as error:
            summary["errors"] = [*summary["errors"], f"report write failed: {error}"]
    row["terminal_bench_ready"] = bool(summary["terminal_bench_ready"])
    row["validation_summary"] = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    row["validation_report_path"] = report_path
    return row
