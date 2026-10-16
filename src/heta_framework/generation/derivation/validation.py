"""Run Oracle, empty and representative incorrect submissions with Harbor."""

from pathlib import Path

from harbor.models.task.config import TaskConfig

from heta_framework.generation.exploration._native.config import ApiSettings, RunConfig
from heta_framework.generation.exploration._native.harbor_runner import HarborRunner

from .config import HetaGenTaskProducerDerivationConfig


def _diagnostics(trial: dict) -> list[dict]:
    """Return bounded execution evidence for a rejected task."""
    if not trial.get("local_trial_dir"):
        return []
    root = Path(trial["local_trial_dir"]).resolve()
    diagnostics = []
    for name in ("verifier/test-stdout.txt", "agent/oracle.txt", "exception.txt"):
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 4096))
            text = stream.read(4096).decode("utf-8", errors="replace").strip()
        if text:
            diagnostics.append({"source": name, "tail": text})
    return diagnostics


def validate_task(
    task: Path, probe: Path, directory: Path, config: HetaGenTaskProducerDerivationConfig
) -> dict:
    task_config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    if task_config.steps or list((task / "environment").glob("*compose*")):
        raise ValueError("Derivation supports single-step Dockerfile tasks")
    for environment in (task_config.environment, task_config.verifier.environment):
        if environment is None:
            continue
        if getattr(environment, "gpus", 0) or getattr(environment, "tpu", None):
            raise ValueError("Derivation currently supports CPU tasks only")
        if (environment.cpus or 0) > config.environment_cpus or (
            environment.memory_mb or 0
        ) > config.environment_memory_mb:
            raise ValueError("Task or verifier exceeds configured resource budget")
    native = RunConfig(
        validation_timeout_sec=config.validation_timeout_sec,
        environment_cpus=config.environment_cpus,
        environment_memory_mb=config.environment_memory_mb,
        job_overhead_timeout_sec=config.job_overhead_timeout_sec,
        worker_stop_grace_sec=config.worker_stop_grace_sec,
        docker_builder=config.docker_builder,
    )
    # Verification runs receive no generation credentials.
    runner = HarborRunner(
        native,
        ApiSettings(api_key="", model="unused"),
        environment_import_path=(
            "heta_framework.generation.derivation.docker:DerivationDockerEnvironment"
        ),
    )
    checks = []
    for name, agent, expected in (
        ("oracle", {"name": "oracle"}, 1),
        ("empty_submission", {"name": "nop"}, 0),
        (
            "wrong_submission",
            {
                "import_path": (
                    "heta_framework.generation.exploration._native.probe:IncorrectSubmissionAgent"
                ),
                "kwargs": {"probe_path": str(probe)},
            },
            0,
        ),
    ):
        trial = runner.run(
            task,
            directory / name,
            {**agent, "override_timeout_sec": config.validation_timeout_sec},
            verify=True,
            timeout=config.validation_timeout_sec,
        )
        reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exception = trial.get("exception_info")
        passed = reward == expected and not exception
        failure_reason = (
            "reference_solution_rejected" if name == "oracle" else "invalid_submission_accepted"
        )
        if exception:
            failure_reason = "execution_error"
        checks.append(
            {
                "name": name,
                "expected_reward": expected,
                "reward": reward,
                "status": "passed" if passed else "failed",
                "exception_type": exception.get("exception_type") if exception else None,
                "failure_reason": None if passed else failure_reason,
                "diagnostics": [] if passed else _diagnostics(trial),
            }
        )
    return {
        "execution_mode": "harbor",
        "validated": all(check["status"] == "passed" for check in checks),
        "checks": checks,
        "limitations": ["One representative wrong submission; no difficulty calibration."],
    }
