"""Explicit configuration for deriving Harbor tasks from existing tasks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit


@dataclass(frozen=True)
class HetaGenTaskProducerDerivationConfig:
    """Credentials are read from the environment or env_file, never serialized here."""

    mode: Literal["harbor", "mock"] = "harbor"
    tasks_per_seed: int = 1
    max_concurrency: int = 1
    model: str = "bailian/glm-5.3"
    api_base_url: str | None = None
    generation_guidance: str = ""
    temperature: float | None = 0.2
    reasoning_effort: str | None = None
    max_tokens: int = 32000
    request_timeout_sec: int = 300
    max_model_restarts: int = 1
    validation_timeout_sec: int = 1800
    environment_cpus: int = 2
    environment_memory_mb: int = 4096
    job_overhead_timeout_sec: int = 1200
    worker_stop_grace_sec: int = 30
    docker_builder: str = "default"
    environment_image: str = "python:3.12-slim"
    max_public_bytes: int = 128000
    env_file: Path = Path(".env")
    workspace: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"harbor", "mock"}:
            raise ValueError("mode must be harbor or mock")
        ranges = {
            "tasks_per_seed": (1, 100),
            "max_concurrency": (1, None),
            "max_tokens": (1, None),
            "request_timeout_sec": (1, None),
            "max_model_restarts": (0, 3),
            "validation_timeout_sec": (30, None),
            "environment_cpus": (1, None),
            "environment_memory_mb": (512, None),
            "job_overhead_timeout_sec": (60, None),
            "worker_stop_grace_sec": (1, 120),
            "max_public_bytes": (1, None),
        }
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if type(value) is not int or value < low or (high is not None and value > high):
                raise ValueError(f"invalid {name}: {value!r}")
        for name in ("model", "docker_builder"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if not isinstance(self.generation_guidance, str):
            raise TypeError("generation_guidance must be text")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.docker_builder):
            raise ValueError("docker_builder must be a plain builder name")
        if not isinstance(self.environment_image, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", self.environment_image
        ):
            raise ValueError("environment_image must be a plain Docker image reference")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2, or None")
        if self.reasoning_effort not in {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("invalid reasoning_effort")
        if self.api_base_url is not None:
            if not isinstance(self.api_base_url, str):
                raise ValueError("api_base_url must be an HTTP(S) endpoint")
            url = urlsplit(self.api_base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.netloc
                or (url.username or url.password or url.query or url.fragment)
            ):
                raise ValueError("api_base_url must be HTTP(S), without credentials or query")
        object.__setattr__(self, "env_file", Path(self.env_file))
        if self.workspace is not None:
            object.__setattr__(self, "workspace", Path(self.workspace))
