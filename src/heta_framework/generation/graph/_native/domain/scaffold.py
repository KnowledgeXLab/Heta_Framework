# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Docker/execution scaffolding for a generated task package.

Re-implements Omega's ``_render_scaffold`` and ``_safe_pip_install_block`` plus
``static_fixes.generator_dockerfile``. Rendering is pure: nothing here touches
the filesystem except reading the vendored template assets that ship inside this
package. The two verifier support modules are read from
``omegadesigner/assets/templates`` at render time rather than being duplicated as
Python string literals.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

from heta_framework.generation.graph._native.domain.schemas import PYTHON_VERSIONS, TBS_CANARY

__all__ = [
    "generator_dockerfile",
    "render_scaffold",
    "safe_pip_install_block",
]

# ``scaffold.py`` lives at ``omegadesigner/domain/scaffold.py``; the vendored
# assets sit beside the ``omegadesigner`` package directory.
_TEMPLATES_DIR: Path = Path(__file__).resolve().parents[1] / "assets" / "templates"

# Omega ``static_fixes.generator_dockerfile`` pin check (exact ``name==version``).
_SAFE_GENERATOR_PIN_RE = re.compile(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+")

# Verifier-only tooling pinned by Omega's scaffold; kept identical so the
# generated tests image matches the pinned versions the TBS checks expect.
_VERIFIER_BASE_REQUIREMENTS: tuple[str, ...] = ("pytest==8.4.1", "pytest-json-ctrf==0.3.5")


def _template_text(name: str) -> str:
    """Read one vendored template asset that ships alongside this package."""
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


def safe_pip_install_block(requirements: list[str]) -> str:
    """Render a shell-continued pinned ``pip install`` RUN block.

    Returns ``""`` for an empty list, matching Omega: callers concatenate the
    result directly into a Dockerfile.
    """
    if not requirements:
        return ""
    slash = chr(92)
    continuation = f" {slash}\n    "
    return f"RUN pip install --no-cache-dir {slash}\n    " + continuation.join(requirements) + "\n\n"


def _generator_dockerfile(runtime: dict[str, Any], *, has_support: bool) -> str:
    """Omega ``static_fixes.generator_dockerfile`` with the support flag explicit."""
    version = str(runtime.get("python_version"))
    dependencies = runtime.get("dependencies", [])
    if version not in PYTHON_VERSIONS or not isinstance(dependencies, list):
        raise ValueError("invalid generator runtime")
    if any(
        not isinstance(dependency, str) or not _SAFE_GENERATOR_PIN_RE.fullmatch(dependency)
        for dependency in dependencies
    ):
        raise ValueError("generator dependencies require exact safe version pins")
    slash = chr(92)
    install = (
        ("RUN pip install --no-cache-dir " + slash + "\n    " + (" " + slash + "\n    ").join(dependencies) + "\n\n")
        if dependencies
        else ""
    )
    return (
        f"# {TBS_CANARY}\nFROM python:{version}-slim\n\n"
        "ENV PYTHONDONTWRITEBYTECODE=1\n"
        + install
        + "COPY generation_data.py /generator/generation_data.py\n"
        + ("COPY generation/support/ /generator/support/\n" if has_support else "")
        + 'ENTRYPOINT ["python3", "/generator/generation_data.py"]\n'
    )


def generator_dockerfile(runtime: dict[str, Any]) -> str:
    """Render the untrusted-generator image Dockerfile.

    Omega threads the ``generation/support/`` presence as a second argument; the
    frozen interface here carries it on ``runtime`` as
    ``has_generation_support`` (aliases ``has_support``), defaulting to ``False``.
    """
    has_support = bool(runtime.get("has_generation_support", runtime.get("has_support", False)))
    return _generator_dockerfile(runtime, has_support=has_support)


def _execution_md(
    name: str,
    runtime: dict[str, Any],
    artifact_paths: list[str],
) -> str:
    """Omega ``_render_scaffold`` execution.md, verbatim."""
    hidden_seed_arg = ",".join(str(seed) for seed in runtime["hidden_seeds"])
    return (
        f"# Execution: {name}\n\n"
        "This is an unexecuted two-stage draft. The model authored the task contract first, "
        "then the multi-file implementation. Omega authored only Docker infrastructure.\n\n"
        "## 1. Materialize native and test data\n\n"
        "`generation_data.py` is untrusted model-authored code. Run it only in the generated "
        "no-network, read-only, resource-limited container.\n\n"
        "```bash\n"
        "mkdir -p environment/data tests/data\n"
        "docker build -f generation/Dockerfile -t omega-task-generator .\n"
        "docker run --rm --network=none --read-only --tmpfs /tmp:rw,size=512m "
        f"--cpus={runtime['cpus']} --memory={runtime['memory_mb']}m "
        '-v "$PWD/environment/data:/outputs/public" '
        '-v "$PWD/tests/data:/outputs/test" omega-task-generator '
        "--public-output-dir /outputs/public --test-output-dir /outputs/test "
        f"--public-seed {runtime['public_seed']} --hidden-seeds {hidden_seed_arg}\n"
        "```\n\n"
        "Inspect both manifests and verify that hidden truth is absent from `environment/data/`.\n\n"
        "## 2. Build task images\n\n"
        "```bash\n"
        "docker build -t omega-task-agent environment\n"
        "docker build -t omega-task-tests tests\n"
        "```\n\n"
        "## 3. Reference and verifier\n\n"
        "Run `solution/solve.sh` with the solution directory mounted at `/solution`, collect only "
        "the declared artifacts, upload them into the separate verifier image, and run "
        "`tests/test.sh`.\n\nDeclared artifacts:\n\n"
        + "".join(f"- `{path}`\n" for path in artifact_paths)
        + "\n## 4. Publication gates\n\n"
        "- Generated data is scientifically valid; reproducibility is assessed under the TBS rubric.\n"
        "- Public and hidden identities are disjoint and public data leaks no truth.\n"
        "- The reference passes every metric with margin under the resource budgets.\n"
        "- All scientifically correct methods are accepted; malformed actual submissions fail.\n"
        "- The verifier is separate, offline, and reads only declared artifacts.\n"
        "- Docker builds and the complete Harbor verifier run are recorded.\n\n"
        "Run Omega's `science validate-paper-task` command to record deterministic-data, "
        "TBS static-check, repeated oracle, and nop evidence. Local validation is not maintainer approval. Do not set "
        "`terminal_bench_ready=true` before those checks succeed.\n"
    )


def render_scaffold(spec_draft: dict[str, Any], runtime: dict[str, Any], task_files: dict[str, str]) -> dict[str, str]:
    """Render Docker infrastructure around the model-authored package files.

    Returns the FULL merged file map: ``task_files`` plus the six deterministic
    scaffold files, with scaffold-owned paths winning on collision (matching
    Omega's ``{**task_files, **self._render_scaffold(...)}`` call site). Pure:
    no writes, and the two support modules come from the vendored assets.
    """
    name = str(spec_draft["package_name"])
    image = f"python:{runtime['python_version']}-slim"
    dependencies = list(runtime["dependencies"])
    public_data_dir = str(runtime["public_data_dir"])
    artifact_paths = [str(path) for path in runtime["artifact_paths"]]
    artifact_parents = sorted({str(PurePosixPath(path).parent) for path in artifact_paths})
    agent_install = safe_pip_install_block(dependencies)
    verifier_install = safe_pip_install_block([*_VERIFIER_BASE_REQUIREMENTS, *dependencies])
    program_artifact = str((spec_draft.get("evaluation_topology") or {}).get("program_artifact", ""))
    starter_root = "/root" if program_artifact.startswith("/root/") else "/app"
    starter_copy = (
        f"COPY starter/ {starter_root}/\n"
        if any(path.startswith("environment/starter/") for path in task_files)
        else ""
    )
    generation_dockerfile = _generator_dockerfile(
        runtime,
        has_support=any(path.startswith("generation/support/") for path in task_files),
    )
    environment_dockerfile = (
        f"# {TBS_CANARY}\n"
        f"FROM {image}\n\n"
        "ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1\n"
        f"{agent_install}"
        "WORKDIR /app\n"
        f"COPY data/ {public_data_dir}/\n"
        f"{starter_copy}"
    )
    verifier_dirs = " ".join(["/logs/verifier", *artifact_parents])
    tests_dockerfile = (
        f"# {TBS_CANARY}\n"
        f"FROM {image}\n\n"
        "ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1\n"
        f"{verifier_install}"
        "COPY . /tests/\n"
        f"RUN mkdir -p {verifier_dirs} && chmod -R go-rwx /tests\n"
        "WORKDIR /tests\n"
    )
    return {
        **task_files,
        "tests/support/omega_runtime.py": _template_text("runtime_support.py"),
        "tests/support/omega_reports.py": _template_text("report_publication.py"),
        "generation/Dockerfile": generation_dockerfile,
        "environment/Dockerfile": environment_dockerfile,
        "tests/Dockerfile": tests_dockerfile,
        "execution.md": _execution_md(name, runtime, artifact_paths),
    }
