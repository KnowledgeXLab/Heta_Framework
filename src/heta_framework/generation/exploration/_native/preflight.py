"""Actionable local checks before any model budget is spent."""

import json
import shutil
import subprocess
from pathlib import Path

from .config import ApiSettings, NetworkSettings, RunConfig


def command(args: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot run {args[0]}: {type(exc).__name__}") from None
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}. Check Docker access/configuration; exit={result.returncode}"
        )
    return result.stdout.strip()


def check(
    config: RunConfig, api: ApiSettings, skills_root: Path, *, skill_names: list[str] | None = None
) -> dict:
    if not shutil.which("docker"):
        raise RuntimeError(
            "Docker CLI is missing. Install Docker Engine/Desktop with Compose and Buildx."
        )
    roles = ["author", "builder", "reviewer"]
    if config.generate_skills:
        roles += ["skill-extractor", "skill-reviewer"]
    for role in skill_names or [f"scientific-task-{name}" for name in roles]:
        if not (skills_root / role / "SKILL.md").is_file():
            raise RuntimeError(
                f"Missing {role} skill; set --skills-root to the repository's skills directory"
            )
    info = json.loads(command(["docker", "info", "--format", "{{json .}}"]))
    if info.get("OSType") != "linux":
        raise RuntimeError(
            "This pipeline requires Linux containers; switch Docker to Linux-container mode"
        )
    compose = command(["docker", "compose", "version", "--short"])
    command(["docker", "buildx", "inspect", config.docker_builder])
    try:
        image = json.loads(command(["docker", "image", "inspect", config.role_image]))[0]
    except RuntimeError:
        raise RuntimeError(
            "Role image unavailable. Run sci-pipeline --build-role-image --config <run.json>, or set role_image to an installed compatible image"
        ) from None
    codex = command(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "codex",
            config.role_image,
            "--version",
        ]
    )
    if codex.strip() != f"codex-cli {config.codex_version}":
        raise RuntimeError(
            "Role image Codex version differs from codex_version; rebuild the image or align the configuration"
        )
    return {
        "docker_server": info.get("ServerVersion"),
        "compose": compose,
        "builder": config.docker_builder,
        "docker_network_pool": api.docker_network_pool,
        "role_image_id": image["Id"],
        "role_image_architecture": image.get("Architecture"),
        "codex": codex,
        "model": api.model,
        "base_url": api.base_url or "https://api.openai.com/v1",
        "api_connectivity": "not_checked",
        "container_network": "not_checked",
    }


def build_role_image(config: RunConfig, network: NetworkSettings, context: Path) -> None:
    if not (context / "Dockerfile").is_file():
        raise RuntimeError("Role image Dockerfile missing; set --role-image-context")
    args = [
        "docker",
        "buildx",
        "build",
        "--builder",
        config.docker_builder,
        "--load",
        "--tag",
        config.role_image,
        "--build-arg",
        f"CODEX_VERSION={config.codex_version}",
    ]
    for key, value in network.container_environment().items():
        args.extend(["--build-arg", f"{key}={value}"])
    if network.container_proxy:
        args.extend(["--add-host", "host.docker.internal:host-gateway"])
    args.append(str(context))
    subprocess.run(args, check=True, timeout=config.job_overhead_timeout_sec)
