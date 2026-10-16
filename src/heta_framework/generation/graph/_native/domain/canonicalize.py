# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic canonicalization of the two model-authored spec files.

Every function here is a faithful, dependency-light re-implementation of Omega's
``science/task_generation/static_fixes.py`` plus
``PaperTaskGenerationPipeline._canonicalize_spec_time_budget``. Nothing in this
module imports ``omega``; the observable output (comment syntax, key ordering,
regex semantics, wrapper text) is kept equivalent so generated packages still
satisfy the Terminal-Bench-Science static checks.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from heta_framework.generation.graph._native.domain.schemas import TBS_CANARY

__all__ = [
    "CANONICAL_TIMEOUT_SENTENCE",
    "add_canary",
    "absolute_instruction_paths",
    "canonical_instruction_suffix",
    "canonicalize_spec_time_budget",
    "fill_environment_fields",
    "fix_reward_initialization",
    "verifier_entrypoint",
]

# Omega's canary lives in three places as one literal; the shared constant is
# authoritative here.
CANARY: str = TBS_CANARY

# The canonical instruction tail, as a ``str.format`` template so both the
# writer (``canonical_instruction_suffix``) and readers can share one literal.
# Matches the sentence enforced by ``assets/checks/check-instruction-suffix.sh``.
CANONICAL_TIMEOUT_SENTENCE: str = (
    "You have {agent_timeout_sec} seconds to complete this task. "
    "Do not cheat by using online solutions or hints specific to this task."
)

# ``_canonicalize_spec_time_budget`` strips this pre-existing sentence (with any
# stale number, including a fractional one) before re-appending the canonical
# form. ``\s*$`` swallows the trailing newline so ``rstrip`` fully normalizes.
_TIMEOUT_SENTENCE_RE = re.compile(
    r"(?:\n\n)?You have [0-9]+(?:\.[0-9]+)? seconds to complete this task\. "
    r"Do not cheat by using online solutions or hints specific to this task\.\s*$"
)

# Omega ``static_fixes.generator_dockerfile`` pin check.
_SAFE_GENERATOR_PIN_RE = re.compile(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+")

# Omega ``static_fixes.absolute_instruction_paths`` path-shaped tokens.
_ABSOLUTE_PATH_RE = re.compile(r"/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+")
_QUOTED_PATH_RE = re.compile(r"([`\"'])([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)\1")

# Omega ``static_fixes.fix_reward_initialization``.
_EMPTY_REWARD_INIT_RE = re.compile(r"(?m)^([ \t]*):\s*>\s*[\"']?/logs/verifier/reward\.txt[\"']?[ \t]*$")

# Extensions whose comment syntax is ``#`` in Omega ``add_canary``.
_HASH_COMMENT_SUFFIXES: frozenset[str] = frozenset({".py", ".sh", ".toml", ".yaml", ".yml", ".cfg", ".ini", ".conf"})

# Verifier wrapper owning the EXIT trap, verbatim from Omega
# ``static_fixes.verifier_entrypoint`` (canary interpolated).
_VERIFIER_WRAPPER_TEMPLATE: str = """#!/usr/bin/env bash
# {canary}
set -euo pipefail
# reward.txt remains private until submitted execution has finished.
finish_reports() {{
    local result=$?
    trap - EXIT
    python3 -c 'import sys; sys.path.insert(0, "/tests"); from support.omega_reports import publish_reports; publish_reports()' || exit 1
    exit "$result"
}}
trap finish_reports EXIT
bash /tests/omega_test_body.sh
"""


def canonical_instruction_suffix(agent_timeout_sec: int) -> str:
    """Exact trailing paragraph TBS requires on ``instruction.md``."""
    return "\n\n" + CANONICAL_TIMEOUT_SENTENCE.format(agent_timeout_sec=agent_timeout_sec) + "\n"


def _replace_section_number(task_toml: str, section: str, key: str, value: str) -> str:
    """Set ``[section] key = value`` in place, appending the key when absent."""
    section_pattern = re.compile(rf"(?ms)(^\[{re.escape(section)}\][^\n]*\n)(.*?)(?=^\[|\Z)")
    section_match = section_pattern.search(task_toml)
    if section_match is None:
        raise ValueError(f"task.toml must contain [{section}]")
    body = section_match.group(2)
    key_pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    replacement = f"{key} = {value}"
    if key_pattern.search(body):
        body = key_pattern.sub(replacement, body, count=1)
    else:
        body = replacement + "\n" + body
    return task_toml[: section_match.start(2)] + body + task_toml[section_match.end(2) :]


def canonicalize_spec_time_budget(
    files: dict[str, str], *, expert_hours: float, agent_timeout_sec: int
) -> dict[str, str]:
    """Re-derive every duplicated time representation from one expert estimate.

    Rewrites ``task.toml`` ``[metadata].expert_time_estimate_hours`` and
    ``[agent].timeout_sec``, strips any stale canonical timeout sentence from
    ``instruction.md``, and re-appends the exact one. Returns a NEW dict; the
    input mapping is never mutated.
    """
    task_toml = files["task.toml"]
    task_toml = _replace_section_number(task_toml, "metadata", "expert_time_estimate_hours", f"{expert_hours:g}")
    task_toml = _replace_section_number(task_toml, "agent", "timeout_sec", str(agent_timeout_sec))
    instruction = _TIMEOUT_SENTENCE_RE.sub("", files["instruction.md"]).rstrip()
    instruction += canonical_instruction_suffix(agent_timeout_sec)
    return {**files, "instruction.md": instruction, "task.toml": task_toml}


def fill_environment_fields(task_toml: str, runtime: dict[str, Any]) -> str:
    """Copy runtime resource integers into ``[environment]`` when absent.

    Only Omega's five integer fields are backfilled, and only when ``runtime``
    actually carries an ``int`` (never ``bool``) for them.
    """
    from heta_framework.generation.graph._native._compat import tomllib

    parsed = tomllib.loads(task_toml)
    environment = parsed.get("environment")
    if not isinstance(environment, dict):
        return task_toml
    keys = ("gpus", "build_timeout_sec", "cpus", "memory_mb", "storage_mb")
    missing = [key for key in keys if key not in environment and type(runtime.get(key)) is int]
    if not missing:
        return task_toml
    match = re.search(r"(?m)^\[environment\][ \t]*(?:#.*)?$", task_toml)
    if not match:
        return task_toml
    insertion = "\n" + "\n".join(f"{key} = {runtime[key]}" for key in missing)
    return task_toml[: match.end()] + insertion + task_toml[match.end() :]


def absolute_instruction_paths(instruction: str, runtime: dict[str, Any], data_contract: dict[str, Any]) -> str:
    """Resolve only quoted paths with one explicit, contract-backed destination.

    A token is rewritten only when exactly one absolute destination is known.
    Ambiguous or unknown tokens are left byte-for-byte untouched, so the
    function can never invent a path.
    """
    candidates: dict[str, set[str]] = {}

    def add(relative: str, absolute: str) -> None:
        candidates.setdefault(relative, set()).add(absolute)
        candidates.setdefault(PurePosixPath(relative).name, set()).add(absolute)

    for absolute in _ABSOLUTE_PATH_RE.findall(instruction):
        add(PurePosixPath(absolute).name, absolute)
    root = runtime.get("public_data_dir")
    if isinstance(root, str) and root.startswith("/"):
        for entry in data_contract.get("public_files", []):
            rel = entry.get("path") if isinstance(entry, dict) else None
            if (
                isinstance(rel, str)
                and rel
                and not rel.startswith("/")
                and not any(c in rel for c in "*?{}[]")
                and ".." not in PurePosixPath(rel).parts
            ):
                add(rel, str(PurePosixPath(root) / rel))

    def replace(match: re.Match[str]) -> str:
        token = match[2]
        choices = candidates.get(token, set())
        return match[1] + next(iter(choices)) + match[1] if len(choices) == 1 else match[0]

    return _QUOTED_PATH_RE.sub(replace, instruction)


def add_canary(content: str, path: str = "") -> str:
    """Insert the TBS canary comment in a language-appropriate syntax.

    Idempotent: a file already carrying the canary is returned unchanged, and a
    path whose suffix has no recognized comment syntax is also returned
    unchanged. The shebang (and, for ``.py``, a PEP 263 coding line) is
    preserved above the canary.
    """
    if CANARY in content:
        return content
    suffix = PurePosixPath(path).suffix
    if suffix == ".md":
        comment = f"<!-- {CANARY} -->\n"
    elif suffix in _HASH_COMMENT_SUFFIXES or PurePosixPath(path).name == "Dockerfile":
        comment = f"# {CANARY}\n"
    else:
        return content
    lines = content.splitlines(keepends=True)
    position = 1 if content.startswith("#!") else 0
    if suffix == ".py":
        for i, line in enumerate(lines[:2]):
            if re.search(r"coding[:=]\s*[-\w.]+", line):
                position = max(position, i + 1)
    lines.insert(position, comment)
    return "".join(lines)


def fix_reward_initialization(content: str) -> str:
    """Replace only the exact empty-file reward initialization.

    Every scoring branch is left intact: a truncated ``: > reward.txt`` makes
    Harbor read an empty reward.
    """
    return _EMPTY_REWARD_INIT_RE.sub(
        lambda match: match[1] + "printf '0\\n' > /logs/verifier/reward.txt",
        content,
    )


def verifier_entrypoint(test_sh: str) -> dict[str, str]:
    """Split ``tests/test.sh`` into a trap-owning wrapper plus its original body.

    The model-written scoring stays intact in a child shell
    (``tests/omega_test_body.sh``); the wrapper owns the EXIT trap so reports are
    published — and ``reward.txt`` made readable — only after scoring finishes.
    Making both files executable is the caller's job.
    """
    wrapper = _VERIFIER_WRAPPER_TEMPLATE.format(canary=CANARY)
    return {"tests/test.sh": wrapper, "tests/omega_test_body.sh": test_sh}
