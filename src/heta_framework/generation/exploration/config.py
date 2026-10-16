"""Typed configuration for source-driven author/build/review production."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class HetaGenTaskProducerExplorationConfig:
    """Production policy and execution settings, validated at construction.

    Credentials remain outside this object. env_file points to the optional
    native credential file; exported environment variables take precedence.
    """

    mode: Literal["harbor", "mock"] = "harbor"
    tasks_per_package: int = 1
    max_concurrency: int = 1
    max_revisions: int = 1
    max_role_attempts: int = 2
    generation_guidance: str = ""
    difficulty_guidance: str = ""
    role_timeout_sec: int = 1800
    validation_timeout_sec: int = 1800
    environment_cpus: int = 2
    environment_memory_mb: int = 4096
    role_image: str = "sci-pipeline/role:codex-0.154.0"
    codex_version: str = "0.154.0"
    docker_builder: str = "default"
    job_overhead_timeout_sec: int = 1200
    worker_stop_grace_sec: int = 30
    mock_revise_once: bool = False
    env_file: Path = Path(".env")
    workspace: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"harbor", "mock"}:
            raise ValueError("mode must be harbor or mock")
        ranges = {
            "tasks_per_package": (1, 100),
            "max_concurrency": (1, None),
            "max_revisions": (0, 10),
            "max_role_attempts": (1, 3),
            "role_timeout_sec": (30, None),
            "validation_timeout_sec": (30, None),
            "environment_cpus": (1, None),
            "environment_memory_mb": (512, None),
            "job_overhead_timeout_sec": (60, None),
            "worker_stop_grace_sec": (1, 120),
        }
        for name, (minimum, maximum) in ranges.items():
            value = getattr(self, name)
            if (
                type(value) is not int
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                raise ValueError(f"invalid {name}: {value!r}")
        for name, pattern in (
            ("codex_version", r"\d+\.\d+\.\d+"),
            ("role_image", r"[A-Za-z0-9][A-Za-z0-9._/@:-]*"),
            ("docker_builder", r"[A-Za-z0-9][A-Za-z0-9_.-]*"),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ValueError(f"invalid {name}")
        for name in ("generation_guidance", "difficulty_guidance"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be text")
        if type(self.mock_revise_once) is not bool:
            raise TypeError("mock_revise_once must be bool")
        object.__setattr__(self, "env_file", Path(self.env_file))
        if self.workspace is not None:
            object.__setattr__(self, "workspace", Path(self.workspace))
