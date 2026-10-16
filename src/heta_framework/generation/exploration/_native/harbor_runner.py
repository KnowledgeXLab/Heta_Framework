"""Native Harbor jobs, isolated subprocess credentials and artifact collection."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from .artifacts import copy_tree, read_json, write_json
from .config import ApiSettings, RunConfig
from .lifecycle import ExecutionSession, LifecycleError


class HarborExecutionError(RuntimeError):
    pass


class HarborRunner:
    def __init__(self, config: RunConfig, api: ApiSettings, *, environment_import_path=None):
        self.config = config
        self.api = api
        self.environment_import_path = environment_import_path

    def scrub(self, root: Path) -> bool:
        """Sanitize captured diagnostics before surfacing errors or model outputs."""
        secrets = {
            value.encode()
            for name, value in self.api.environment().items()
            if name.endswith("_KEY") and value
        }
        found = False
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                data = path.read_bytes()
                clean = data
                for secret in secrets:
                    clean = clean.replace(secret, b"[REDACTED_API_KEY]")
                if clean != data:
                    found = True
                    path.write_bytes(clean)
        return found

    def run(
        self,
        task: Path,
        directory: Path,
        agent: dict,
        *,
        verify: bool,
        timeout: int,
        artifacts: bool = False,
        credentials: bool = False,
    ) -> dict:
        directory.mkdir(parents=True, exist_ok=False)
        job_name = "sci-" + uuid4().hex[:12]
        task_config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
        requested = task_config.environment
        if requested.gpus or requested.tpu:
            raise HarborExecutionError(
                "This backend supports CPU tasks only; task requests GPU/TPU resources"
            )
        if (requested.cpus or 0) > self.config.environment_cpus or (
            requested.memory_mb or 0
        ) > self.config.environment_memory_mb:
            raise HarborExecutionError(
                "Task exceeds configured CPU/memory budget; increase environment_cpus/environment_memory_mb or revise task requirements"
            )
        container_env = self.api.container_environment()
        # Ownership labels allow cleanup even if the trial record was not fully written.
        compose_path = directory / "execution.compose.json"
        main = {"labels": {"sci.pipeline.execution": job_name}}
        if container_env:
            main.update(
                build={"args": container_env, "extra_hosts": ["host.docker.internal:host-gateway"]},
                extra_hosts=["host.docker.internal:host-gateway"],
            )
        write_json(
            compose_path,
            {
                "services": {"main": main},
                "networks": {"default": {"labels": {"sci.pipeline.execution": job_name}}},
            },
        )
        compose_paths = [str(compose_path)]
        config = JobConfig.model_validate(
            {
                "job_name": job_name,
                "jobs_dir": str(directory / "jobs"),
                "n_attempts": 1,
                "n_concurrent_trials": 1,
                "quiet": True,
                "retry": {"max_retries": 0},
                "tasks": [{"path": str(task.resolve())}],
                "agents": [agent],
                "environment": {
                    "type": "docker",
                    "import_path": self.environment_import_path,
                    "delete": True,
                    "override_cpus": requested.cpus or self.config.environment_cpus,
                    "override_memory_mb": requested.memory_mb or self.config.environment_memory_mb,
                    "override_gpus": 0,
                    "env": container_env,
                    "extra_docker_compose": compose_paths,
                },
                "verifier": {"disable": not verify, "override_timeout_sec": timeout},
                "artifacts": [{"source": "/workspace/output", "destination": "output"}]
                if artifacts
                else [],
            }
        )
        config_path = directory / "job.json"
        config_path.write_text(config.model_dump_json(indent=2))
        # Keep machine/tool access, but remove inherited model credentials and Codex overrides.
        allowed = {
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_CONFIG",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        }
        env = {key: value for key, value in os.environ.items() if key in allowed}
        # A project-scoped builder selection; never change the user's global builder.
        env["BUILDX_BUILDER"] = self.config.docker_builder
        if credentials:
            env.update(self.api.environment())
        log = directory / "worker.log"
        deadline = timeout + self.config.job_overhead_timeout_sec
        try:
            with ExecutionSession(
                directory, directory / "jobs" / job_name, grace=self.config.worker_stop_grace_sec
            ) as session:
                if self.api.docker_network_pool:
                    from .docker_network import create_execution_network

                    network = create_execution_network(self.api.docker_network_pool, job_name)
                    write_json(directory / "network.json", network)
                    write_json(
                        compose_path,
                        {
                            "services": {"main": main},
                            "networks": {
                                "default": {
                                    "external": True,
                                    "name": network["name"],
                                }
                            },
                        },
                    )
                with log.open("wb") as stream:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "heta_framework.generation.exploration._native.harbor_worker",
                            str(config_path),
                            str(deadline),
                        ],
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    session.attach(process)
                    code = session.wait(deadline + 120)
        except LifecycleError as exc:
            raise HarborExecutionError(str(exc)) from None
        finally:
            self.scrub(directory)
        trials = list((directory / "jobs" / job_name).glob("*/result.json"))
        if code or len(trials) != 1:
            raise HarborExecutionError(f"Harbor job did not produce one trial result; see {log}")
        result = read_json(trials[0])
        result["local_trial_dir"] = str(trials[0].parent)
        write_json(directory / "trial-summary.json", result)
        return result

    def codex(self) -> dict:
        return {
            "name": "codex",
            "model_name": self.api.model,
            "override_timeout_sec": self.config.role_timeout_sec,
            "kwargs": {"version": self.config.codex_version, "web_search": "disabled"},
            "env": {"OPENAI_API_KEY": "${OPENAI_API_KEY}", "OPENAI_BASE_URL": "${OPENAI_BASE_URL}"}
            if self.api.base_url
            else {"OPENAI_API_KEY": "${OPENAI_API_KEY}"},
        }

    def collect(self, result: dict, output: Path) -> None:
        if result.get("exception_info"):
            kind = result["exception_info"]["exception_type"]
            raise HarborExecutionError(f"Role failed: {kind}; see Harbor trial logs")
        source = Path(result["local_trial_dir"]) / "artifacts" / "output"
        if not source.is_dir():
            raise ValueError("Role did not return its output directory")
        if output.exists():
            if any(output.iterdir()):
                raise HarborExecutionError("Role output directory must be empty before collection")
            output.rmdir()
        copy_tree(source, output)
        if self.scrub(output):
            raise HarborExecutionError("Role output contained a credential; sanitized and rejected")
