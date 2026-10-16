# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellable external process execution.

Every long external step in this project (`docker build`, `docker run`, `harbor run`) used
``subprocess.run``. That is fine while the work happens in a plain Python loop, but once those
steps run *inside* Data Designer's scheduler a run cancellation (Ctrl-C, a failing sibling
task, a workflow abort) can no longer stop them: ``subprocess.run`` blocks in a worker thread
that Data Designer cannot interrupt, so the process outlives the run and its containers keep
consuming the machine.

This module owns that lifecycle instead:

* the child gets its own process group (``start_new_session=True``), so a cancel can take the
  whole tree down rather than just the CLI wrapper;
* the wait loop polls a caller-supplied cancellation predicate (Data Designer's
  ``is_run_cancellation_requested``) between short waits;
* timeouts and missing binaries stay *values* (rc 124 / 127), exactly like the
  ``subprocess.run``-based code they replace, because a rejected package must remain a row
  value and never an exception that aborts a dataset generation run.

The predicate is injected rather than imported so this module stays free of Data Designer
imports; the DD-facing layer passes its own.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Same convention as the previous `subprocess.run` wrappers: 124 = timeout, 127 = no binary.
TIMEOUT_RETURNCODE = 124
MISSING_RETURNCODE = 127
CANCELLED_RETURNCODE = 130

#: How long to wait for a child to die after SIGTERM before escalating to SIGKILL.
_KILL_GRACE_SEC = 5.0
#: Cancellation/timeout poll interval: long enough to be free, short enough to react.
_POLL_SEC = 0.5


@dataclass(frozen=True)
class ProcessResult:
    """Outcome of one external command; never an exception."""

    command: tuple[str, ...]
    returncode: int
    output: str
    timed_out: bool = False
    cancelled: bool = False
    error: str = ""


def _terminate(process: subprocess.Popen[str]) -> None:
    """SIGTERM the child's process group, then SIGKILL if it does not die."""
    try:
        group = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        group = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            if group is not None:
                os.killpg(group, sig)
            else:
                process.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            process.wait(timeout=_KILL_GRACE_SEC)
            return
        except subprocess.TimeoutExpired:
            continue


def run_cancellable(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: float,
    cancelled: Callable[[], bool] | None = None,
    poll_interval: float = _POLL_SEC,
) -> ProcessResult:
    """Run ``command``, honouring an external cancellation predicate and a deadline.

    ``cancelled`` is polled between waits; when it turns true the child's process group is
    terminated and the result carries ``cancelled=True`` with whatever output arrived. A
    container started by the killed CLI may briefly outlive it — Docker/Harbor own that
    cleanup, and the caller records the cancellation instead of pretending the run completed.
    """
    argv = [str(part) for part in command]
    try:
        process = subprocess.Popen(  # noqa: S603 - command comes from this package, not user input
            argv,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            # Heta adaptation: external validators do not need model credentials.
            env={key: value for key, value in os.environ.items() if key in {
                "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH",
                "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "BUILDX_BUILDER",
                "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
            }},
        )
    except OSError as error:
        return ProcessResult(tuple(argv), MISSING_RETURNCODE, "", error=f"{type(error).__name__}: {error}")

    deadline = time.monotonic() + max(0.0, timeout)
    output = ""
    while True:
        if cancelled is not None and cancelled():
            _terminate(process)
            return ProcessResult(tuple(argv), CANCELLED_RETURNCODE, output, cancelled=True, error="cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate(process)
            return ProcessResult(
                tuple(argv), TIMEOUT_RETURNCODE, output, timed_out=True, error=f"timed out after {timeout:g}s"
            )
        try:
            output = process.communicate(timeout=min(poll_interval, remaining))[0] or ""
            return ProcessResult(tuple(argv), process.returncode or 0, output)
        except subprocess.TimeoutExpired:
            continue
