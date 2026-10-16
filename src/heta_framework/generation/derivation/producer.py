"""Object-store boundary for seed-driven Harbor task derivation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key
from heta_framework.generation.types import (
    InputRequirement,
    ObjectInput,
    TaskRequest,
    TaskRequirements,
    TaskResult,
)

from .config import HetaGenTaskProducerDerivationConfig
from .inputs import stage_seeds


class HetaGenTaskProducerDerivation:
    """Derive independent tasks from existing Harbor packages using Data Designer."""

    name = "heta_gen_derivation"

    def __init__(self, config: HetaGenTaskProducerDerivationConfig | None = None) -> None:
        self._config = config or HetaGenTaskProducerDerivationConfig()

    @property
    def config(self) -> HetaGenTaskProducerDerivationConfig:
        return self._config

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            inputs=(
                InputRequirement(
                    name="seed_tasks",
                    description="Existing Harbor task roots, including public files.",
                ),
            )
        )

    async def produce(
        self, request: TaskRequest, *, object_store: ObjectStoreProtocol
    ) -> TaskResult:
        self.requirements.validate(request)
        if await object_store.list(request.output_prefix + "/"):
            raise FileExistsError(f"output prefix is already occupied: {request.output_prefix}")
        selection = request.inputs["seed_tasks"]
        assert isinstance(selection, ObjectInput)
        if self.config.workspace:
            self.config.workspace.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.config.workspace, prefix="heta-derivation-") as directory:
            root = Path(directory)
            seeds, inputs = await stage_seeds(
                selection,
                object_store,
                root / "seeds",
                request.output_prefix,
                self.config.max_public_bytes,
            )
            try:
                from data_designer.config import ModelProvider
                from dotenv import dotenv_values
                from harbor.models.task.config import TaskConfig

                from .designer import execute
            except ModuleNotFoundError as error:
                raise ImportError("Install heta-framework[generation] to use Derivation") from error
            for seed in seeds:
                TaskConfig.model_validate_toml((Path(seed["seed_path"]) / "task.toml").read_text())
            values = {**dotenv_values(self.config.env_file), **os.environ}
            secret = values.get("OPENAI_API_KEY") or ""
            provider = None
            if self.config.mode == "harbor":
                endpoint = self.config.api_base_url or values.get("OPENAI_BASE_URL")
                if not secret or not endpoint:
                    raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL/api_base_url are required")
                provider = ModelProvider(
                    name="derivation",
                    endpoint=endpoint,
                    provider_type="openai",
                    api_key=secret,
                )
            run = root / "run"
            run.mkdir()
            future = asyncio.create_task(
                asyncio.to_thread(
                    execute,
                    seeds,
                    run,
                    self.config,
                    provider,
                )
            )
            try:
                produced = await asyncio.shield(future)
            except asyncio.CancelledError:
                await asyncio.shield(future)
                raise
            # Never allow a generated credential to reach published artifacts.
            if secret:
                for path in run.rglob("*"):
                    if path.is_file():
                        data = path.read_bytes()
                        if secret.encode() in data:
                            path.write_bytes(data.replace(secret.encode(), b"[REDACTED_API_KEY]"))
                            for row in produced["results"]:
                                if row.get("task_package") and path.is_relative_to(
                                    Path(row["task_package"])
                                ):
                                    row["status"] = "credential_rejected"
            tasks, fingerprints = [], set()
            for row in produced["results"]:
                if self.config.mode != "harbor" or row["status"] != "accepted":
                    continue
                if not row.get("validation", {}).get("validated"):
                    raise ValueError("Accepted task lacks actual verification evidence")
                task = Path(row["task_package"]).resolve()
                if not task.is_relative_to(run.resolve()):
                    raise ValueError("task path escapes the run directory")
                TaskConfig.model_validate_toml((task / "task.toml").read_text())
                public_hash = hashlib.sha256()
                for path in sorted(task.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("task publication does not support symbolic links")
                    if path.is_file() and (
                        path.name == "instruction.md"
                        or path.relative_to(task).parts[0] == "environment"
                    ):
                        public_hash.update(path.relative_to(task).as_posix().encode())
                        public_hash.update(path.read_bytes())
                if public_hash.hexdigest() in fingerprints:
                    row["status"] = "duplicate_rejected"
                    continue
                fingerprints.add(public_hash.hexdigest())
                files = []
                for path in sorted(task.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(task).as_posix()
                    key = join_object_key(
                        request.output_prefix, f"tasks/{row['task_id']}/{relative}"
                    )
                    data = path.read_bytes()
                    await object_store.put(key, data)
                    files.append(
                        {
                            "path": relative,
                            "key": key,
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "mode": 0o755 if path.suffix == ".sh" else 0o644,
                        }
                    )
                evidence = join_object_key(
                    request.output_prefix, f"evidence/{row['task_id']}/validation.json"
                )
                await object_store.put(evidence, json.dumps(row["validation"]).encode())
                novelty_key = join_object_key(
                    request.output_prefix,
                    f"evidence/{row['task_id']}/novelty.json",
                )
                await object_store.put(novelty_key, json.dumps(row["novelty_review"]).encode())
                tasks.append(
                    {
                        "task_id": row["task_id"],
                        "seed_task_id": row["seed_task_id"],
                        "variant": row["variant"],
                        "files": files,
                        "validation_kind": "derivation_harbor_checks",
                        "evidence_keys": [evidence, novelty_key],
                    }
                )
            expected = len(seeds) * self.config.tasks_per_seed
            status = "succeeded" if len(tasks) == expected else "partial" if tasks else "failed"
            patterns_key = join_object_key(request.output_prefix, "patterns.json")
            pattern_json = json.dumps(produced["patterns"], ensure_ascii=False)
            if secret:
                pattern_json = pattern_json.replace(secret, "[REDACTED_API_KEY]")
            await object_store.put(patterns_key, pattern_json.encode())
            verdicts = [
                {k: v for k, v in row.items() if k != "task_package"} for row in produced["results"]
            ]
            for verdict in verdicts:
                if "candidate_check" in verdict:
                    verdict["candidate_check"] = {
                        k: v for k, v in verdict["candidate_check"].items() if k != "package_path"
                    }
            # Local staging paths and credentials do not belong in the published report.
            report = {
                "schema_version": 1,
                "producer": self.name,
                "status": status,
                "execution_mode": self.config.mode,
                "simulation_completed": self.config.mode == "mock",
                "inputs": inputs,
                "requested_task_count": expected,
                "delivered_task_count": len(tasks),
                "patterns_key": patterns_key,
                "stages": produced["stages"],
                "native_results": verdicts,
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(self.config).items()
                    if k not in {"env_file", "workspace", "api_base_url"}
                },
            }
            serialized = json.dumps(report, ensure_ascii=False).replace(str(root), "<workspace>")
            if secret:
                serialized = serialized.replace(secret, "[REDACTED_API_KEY]")
            report_key = join_object_key(request.output_prefix, "report.json")
            manifest_key = join_object_key(request.output_prefix, "tasks.json")
            await object_store.put(report_key, serialized.encode())
            await object_store.put(
                manifest_key,
                json.dumps(
                    {
                        "schema_version": 1,
                        "format": "harbor",
                        "producer": self.name,
                        "report_key": report_key,
                        "tasks": tasks,
                    },
                    ensure_ascii=False,
                ).encode(),
            )
        return TaskResult(status, manifest_key, report_key)
