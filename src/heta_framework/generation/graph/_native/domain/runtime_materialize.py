# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data materialization: build the generator image, run it twice, materialize the data.

This is the `validate` gate that needs a container, and it is deliberately *outside* the
Data Designer generation loop: a column cannot own a Docker build, a read-only networked-off
run, or the reproducibility comparison. The workflow hands a materialized ``package_dir``
to this module; every failure comes back as data, never as an exception.

Faithful to Omega ``validation.py:203-333``:

1. require ``generation/Dockerfile``, ``generation_data.py`` and ``generation_manifest.json``;
2. build ``omega-task-generator-<sha16>`` where the tag hashes the Dockerfile, the generator
   and ``generation/support/**/*.py``;
3. run the generator twice in a container with ``--network=none --read-only --cap-drop=ALL
   --security-opt=no-new-privileges --pids-limit=128``, a resource cap and ``/tmp`` tmpfs;
4. compare per-file SHA256 across the two runs (advisory, as in Omega) and require non-empty
   output with no symlinks;
5. check that ``evaluation_topology.hidden_case_glob`` actually matches generated test data;
6. copy run 1 into ``environment/data`` and ``tests/data``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from heta_framework.generation.graph._native.domain.process import run_cancellable
from heta_framework.generation.graph._native.io_utils import atomic_write_text, read_json, tree_hashes

HIDDEN_GLOB_PREFIX = "/tests/data/"
GENERATOR_IMAGE_PREFIX = "omega-task-generator-"
CONTAINER_PUBLIC_DIR = "/outputs/public"
CONTAINER_TEST_DIR = "/outputs/test"


class HardeningProfile(BaseModel):
    """Which optional container hardening flags this Docker host actually supports.

    ``--security-opt=no-new-privileges`` breaks ``exec`` inside the container on some
    nested/rootless Docker hosts (observed here: ``exec /bin/sh: operation not permitted``).
    Rather than silently dropping it or failing the whole gate, the profile records exactly
    what was applied so the validation report shows the deviation.
    """

    model_config = ConfigDict(extra="forbid")

    no_new_privileges: bool = True
    deviations: list[str] = Field(default_factory=list)


class MaterializeResult(BaseModel):
    """Outcome of the materialization gate; ``ok`` is the only field callers must branch on."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    built: bool = False
    deterministic: bool | None = None
    generation_runs: int = 0
    image: str = ""
    file_sha256: dict[str, dict[str, str]] = Field(default_factory=dict)
    materialized: list[str] = Field(default_factory=list)
    logs: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    hardening: HardeningProfile = Field(default_factory=HardeningProfile)
    docker_available: bool = True


def docker_available() -> bool:
    try:
        completed = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _host_user_args() -> list[str]:
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return ["--user", f"{os.getuid()}:{os.getgid()}"]
    return []


def _execute(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    cancelled: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Omega's ``_execute``: timeouts become rc 124, missing binaries rc 127.

    Delegates the wait to :mod:`omegadesigner.domain.process` so a Data Designer run
    cancellation can actually stop a `docker build`/`docker run` instead of leaving it
    running past the run that started it.
    """
    result = run_cancellable(command, cwd=cwd, timeout=timeout, cancelled=cancelled)
    stderr = result.output if result.returncode != 0 else ""
    if result.error:
        stderr = f"{stderr}\n{result.error}".strip()
    return subprocess.CompletedProcess(command, result.returncode, result.output, stderr)


_HARDENING_CACHE: dict[str, HardeningProfile] = {}


def probe_hardening(image: str, *, cancelled: Callable[[], bool] | None = None) -> HardeningProfile:
    """Probe optional flags against the freshly built generator image (cached per image)."""
    if image in _HARDENING_CACHE:
        return _HARDENING_CACHE[image]
    profile = HardeningProfile()
    base = ["docker", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL", "--entrypoint", "/bin/sh"]
    probe = _execute([*base, image, "-c", "exit 0"], cwd=Path.cwd(), timeout=120, cancelled=cancelled)
    if probe.returncode != 0:
        # Nothing works; report the deviation and let the run stage surface the real error.
        profile = HardeningProfile(no_new_privileges=False, deviations=["container-exec"])
        _HARDENING_CACHE[image] = profile
        return profile
    hardened = _execute(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "exit 0",
        ],
        cwd=Path.cwd(),
        timeout=120,
    )
    if hardened.returncode != 0:
        profile = HardeningProfile(no_new_privileges=False, deviations=["no-new-privileges"])
    _HARDENING_CACHE[image] = profile
    return profile


def _generator_image_tag(package_dir: Path) -> tuple[str, list[Path]]:
    dockerfile = package_dir / "generation" / "Dockerfile"
    generator = package_dir / "generation_data.py"
    sources = [dockerfile, generator, *sorted((package_dir / "generation" / "support").rglob("*.py"))]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(package_dir).as_posix().encode())
        digest.update(source.read_bytes())
    return f"{GENERATOR_IMAGE_PREFIX}{digest.hexdigest()[:16]}", sources


def materialize_package(
    package_dir: str | Path,
    *,
    validation_dir: str | Path | None = None,
    work_root: str | Path | None = None,
    timeout_floor_sec: int = 300,
    cancelled: Callable[[], bool] | None = None,
) -> MaterializeResult:
    """Build, run twice, verify reproducibility, and materialize the generated data.

    ``work_root`` is where the two generator runs write their outputs before they are
    copied into the package. It MUST be a path the Docker daemon can see: the outputs are
    bind-mounted into the container, and a daemon on a different filesystem view silently
    substitutes an empty directory instead of failing loudly (observed as a confusing
    ``PermissionError: /outputs/public``). It defaults to a directory next to the package,
    which is daemon-visible whenever the package is.
    """
    package = Path(package_dir).resolve()
    target = Path(validation_dir) if validation_dir else package / "authoring" / "validation"
    target.mkdir(parents=True, exist_ok=True)

    if not docker_available():
        return MaterializeResult(ok=False, docker_available=False, errors=["docker is not available"])

    dockerfile = package / "generation" / "Dockerfile"
    generator = package / "generation_data.py"
    manifest_path = package / "generation_manifest.json"
    missing = [str(path.relative_to(package)) for path in (dockerfile, generator, manifest_path) if not path.is_file()]
    if missing:
        return MaterializeResult(ok=False, errors=[f"missing for materialization: {', '.join(missing)}"])

    manifest = read_json(manifest_path)
    runtime = manifest.get("runtime_spec") or {}
    try:
        public_seed = int(runtime["public_seed"])
        hidden_seeds = [int(seed) for seed in runtime["hidden_seeds"]]
        cpus = int(runtime["cpus"])
        memory_mb = int(runtime["memory_mb"])
        build_timeout = int(runtime["build_timeout_sec"])
    except (KeyError, TypeError, ValueError) as error:
        return MaterializeResult(ok=False, errors=[f"invalid runtime_spec for materialization: {error}"])

    image, _sources = _generator_image_tag(package)
    logs: dict[str, str] = {}

    build = _execute(
        ["docker", "build", "-f", str(dockerfile), "-t", image, str(package)], cwd=package, timeout=build_timeout
    )
    logs["generator-build.log"] = (build.stdout or "") + (build.stderr or "")
    atomic_write_text(target / "generator-build.log", logs["generator-build.log"])
    if build.returncode != 0:
        return MaterializeResult(
            ok=False, built=False, image=image, logs=logs, errors=["generator Docker image build failed"]
        )

    hardening = probe_hardening(image, cancelled=cancelled)
    result = MaterializeResult(ok=True, built=True, image=image, logs=logs, hardening=hardening)
    scratch_root = Path(work_root) if work_root else package / "authoring" / "materialize-work"
    scratch_root.mkdir(parents=True, exist_ok=True)
    if hardening.deviations:
        result.warnings.append("container hardening deviations: " + ", ".join(hardening.deviations))
    generated: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(
        prefix="omegadesigner-data-", dir=scratch_root, ignore_cleanup_errors=True
    ) as temporary:
        base = Path(temporary)
        for repeat in (1, 2):
            public = base / f"run-{repeat}" / "public"
            private = base / f"run-{repeat}" / "test"
            public.mkdir(parents=True)
            private.mkdir(parents=True)
            public.parent.chmod(0o777)
            public.chmod(0o777)
            private.chmod(0o777)
            command = [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                *(["--security-opt=no-new-privileges"] if hardening.no_new_privileges else []),
                "--pids-limit=128",
                *_host_user_args(),
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-e",
                "MKL_NUM_THREADS=1",
                "--tmpfs",
                "/tmp:rw,size=512m",
                f"--cpus={cpus}",
                f"--memory={memory_mb}m",
                "-v",
                f"{public.parent}:/outputs",
                "--entrypoint",
                "/bin/sh",
                image,
                "-c",
                (
                    'python3 /generator/generation_data.py "$@"; status=$?; '
                    "chmod -R a+rwX /outputs/public /outputs/test || true; "
                    'exit "$status"'
                ),
                "omegadesigner-generator",
                "--public-output-dir",
                CONTAINER_PUBLIC_DIR,
                "--test-output-dir",
                CONTAINER_TEST_DIR,
                "--public-seed",
                str(public_seed),
                "--hidden-seeds",
                ",".join(map(str, hidden_seeds)),
            ]
            run = _execute(command, cwd=package, timeout=max(timeout_floor_sec, build_timeout))
            logs[f"generator-run-{repeat}.log"] = (run.stdout or "") + (run.stderr or "")
            atomic_write_text(target / f"generator-run-{repeat}.log", logs[f"generator-run-{repeat}.log"])
            if run.returncode != 0:
                if repeat == 1:
                    result.errors.append(f"generator run 1 failed (rc={run.returncode})")
                    result.ok = False
                    return result
                result.warnings.append("second generation failed; first dataset remains usable")
                break
            if not public.is_dir() or not private.is_dir() or public.is_symlink() or private.is_symlink():
                result.errors.append("generator did not create both output directories")
                result.ok = False
                return result
            if any(path.is_symlink() for directory in (public, private) for path in directory.rglob("*")):
                result.errors.append("generated data must not contain symlinks")
                result.ok = False
                return result
            generated.append((public, private))

        if not generated:
            result.errors.append("generator produced no usable dataset")
            result.ok = False
            return result

        first = {"public": tree_hashes(generated[0][0]), "test": tree_hashes(generated[0][1])}
        result.generation_runs = len(generated)
        result.file_sha256 = first
        if len(generated) == 2:
            second = {"public": tree_hashes(generated[1][0]), "test": tree_hashes(generated[1][1])}
            result.deterministic = first == second
            if not result.deterministic:
                result.warnings.append("same-seed byte hashes differ; scientific reproducibility needs review")
        if not first["public"] or not first["test"]:
            result.errors.append("generator produced empty public/test data")
            result.ok = False
            return result

        hidden_glob = str((manifest.get("evaluation_topology") or {}).get("hidden_case_glob") or "")
        if hidden_glob:
            if not hidden_glob.startswith(HIDDEN_GLOB_PREFIX) or ".." in Path(hidden_glob).parts:
                result.errors.append(f"hidden_case_glob must live under {HIDDEN_GLOB_PREFIX}: {hidden_glob}")
                result.ok = False
                return result
            relative_glob = hidden_glob[len(HIDDEN_GLOB_PREFIX) :]
            if not any(generated[0][1].glob(relative_glob)):
                result.errors.append(
                    f"hidden_case_glob matched no generated cases: {hidden_glob}; actual: {sorted(first['test'])[:12]}"
                )
                result.ok = False
                return result

        for source, destination in (
            (generated[0][0], package / "environment" / "data"),
            (generated[0][1], package / "tests" / "data"),
        ):
            if destination.exists() and any(destination.rglob("*")):
                if tree_hashes(destination) != tree_hashes(source):
                    result.errors.append(f"materialized data differs from existing {destination}")
                    result.ok = False
                    return result
            else:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination, dirs_exist_ok=True)
            result.materialized.append(str(destination))

    return result
