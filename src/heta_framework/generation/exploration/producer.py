"""Object-store adapter for the migrated Sci-SKT source-package workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key
from heta_framework.generation.exploration.config import HetaGenTaskProducerExplorationConfig
from heta_framework.generation.types import (
    InputRequirement,
    ObjectInput,
    TaskRequest,
    TaskRequirements,
    TaskResult,
)


class HetaGenTaskProducerExploration:
    """Produce tasks from one native PACKAGE.md source package in-process."""

    name = "heta_gen_exploration"

    def __init__(self, config: HetaGenTaskProducerExplorationConfig | None = None) -> None:
        self._config = config or HetaGenTaskProducerExplorationConfig()

    @property
    def config(self) -> HetaGenTaskProducerExplorationConfig:
        """Read-only production configuration."""
        return self._config

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            inputs=(
                InputRequirement(
                    name="materials",
                    description="One source package containing PACKAGE.md and optional files.",
                ),
            )
        )

    async def produce(
        self,
        request: TaskRequest,
        *,
        object_store: ObjectStoreProtocol,
    ) -> TaskResult:
        self.requirements.validate(request)
        selection = request.inputs["materials"]
        assert isinstance(selection, ObjectInput)
        manifest_key = join_object_key(request.output_prefix, "tasks.json")
        # Reject occupied output trees, including interrupted publications.
        if await object_store.list(request.output_prefix + "/"):
            raise FileExistsError(f"output prefix is already occupied: {request.output_prefix}")
        keys = (
            selection.keys
            if selection.prefix is None
            else tuple(
                item.key
                for item in await object_store.list(selection.prefix)
                if not selection.prefix or item.key.startswith(selection.prefix.rstrip("/") + "/")
            )
        )
        if not keys:
            raise ValueError("materials selection contains no objects")
        roots = [key for key in keys if PurePosixPath(key).name == "PACKAGE.md"]
        if len(roots) != 1:
            raise ValueError("materials must contain exactly one PACKAGE.md")
        base = PurePosixPath(roots[0]).parent
        for key in keys:
            if not PurePosixPath(key).is_relative_to(base):
                raise ValueError("all materials must belong to the source package")
            if key == request.output_prefix or key.startswith(request.output_prefix + "/"):
                raise ValueError("output prefix overlaps selected inputs")
        try:
            from heta_framework.generation.exploration._native.config import RunConfig
            from heta_framework.generation.exploration._native.contracts import (
                TaskRequest as NativeRequest,
            )
            from heta_framework.generation.exploration._native.designer import (
                create_dataset,
            )
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Install heta-framework[generation] to use HetaGenTaskProducerExploration"
            ) from exc
        native_config = RunConfig.model_validate(
            {
                key: value
                for key, value in asdict(self.config).items()
                if key not in {"env_file", "workspace"}
            }
        )
        if native_config.generate_skills:
            raise ValueError("optional skill extraction is not part of this task producer")
        skills = Path(__file__).parent / "_native" / "skills"
        if self.config.workspace:
            self.config.workspace.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            dir=self.config.workspace, prefix="heta-material-exploration-"
        ) as directory:
            root = Path(directory)
            package = root / "source"
            package.mkdir()
            inputs = []
            for key in keys:
                relative = PurePosixPath(key).relative_to(base).as_posix()
                data = await object_store.get(key)
                path = package / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                inputs.append({"key": key, "sha256": hashlib.sha256(data).hexdigest()})
            run_dir = root / "run"
            run_dir.mkdir()
            native_requests = [
                NativeRequest(
                    task_id=f"p000-task-{variant:03d}",
                    package_uri=str(package),
                    variant=variant,
                )
                for variant in range(native_config.tasks_per_package)
            ]

            def execute() -> list[dict]:
                if native_config.mode == "harbor":
                    from ._native.config import ApiSettings
                    from ._native.preflight import check

                    check(native_config, ApiSettings.load(self.config.env_file.resolve()), skills)
                return create_dataset(
                    run_dir,
                    native_requests,
                    native_config,
                    self.config.env_file.resolve(),
                    skills,
                )

            execution = asyncio.create_task(asyncio.to_thread(execute))
            try:
                rows = await asyncio.shield(execution)
            except asyncio.CancelledError:
                # Never delete a workspace while native threads are still using it.
                await asyncio.shield(execution)
                raise
            tasks = []
            for row in rows:
                if not (
                    native_config.mode == "harbor"
                    and row["status"] == "accepted"
                    and row["execution_mode"] == "harbor"
                    and row["validated"]
                ):
                    continue
                task = Path(row["task_package_uri"]).resolve()
                if not task.is_relative_to(run_dir.resolve()):
                    raise ValueError("native task path escapes the run directory")
                from harbor.models.task.config import TaskConfig

                TaskConfig.model_validate_toml((task / "task.toml").read_text())
                for required in ("instruction.md", "solution/solve.sh", "tests/test.sh"):
                    if not (task / required).is_file():
                        raise ValueError(f"native task missing {required}")
                task_id = row["task_id"]
                files = []
                for path in sorted(task.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("task publication does not support symbolic links")
                    if not path.is_file():
                        continue
                    relative = path.relative_to(task).as_posix()
                    key = join_object_key(request.output_prefix, f"tasks/{task_id}/{relative}")
                    data = path.read_bytes()
                    await object_store.put(key, data)
                    files.append(
                        {
                            "path": relative,
                            "key": key,
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "mode": 0o755
                            if relative.endswith(".sh")
                            else path.stat().st_mode & 0o777,
                        }
                    )
                evidence_keys = []
                native_task_root = run_dir / "tasks" / task_id
                # Publish structured verification/review evidence, not credential-bearing job logs.
                for pattern in ("*/builder/validation.json", "*/reviewer/review.json"):
                    for path in sorted(native_task_root.glob(pattern)):
                        key = join_object_key(
                            request.output_prefix,
                            f"evidence/{task_id}/{path.relative_to(native_task_root).as_posix()}",
                        )
                        await object_store.put(key, path.read_bytes())
                        evidence_keys.append(key)
                tasks.append(
                    {
                        "task_id": task_id,
                        "files": files,
                        "revision": row["candidate_revision"],
                        "validation_kind": "author_builder_reviewer",
                        "evidence_keys": evidence_keys,
                    }
                )
            simulated = native_config.mode == "mock"
            status = (
                "succeeded"
                if len(tasks) == len(rows) and rows
                else ("partial" if tasks else "failed")
            )
            report_key = join_object_key(request.output_prefix, "report.json")
            # Keep portable native verdicts, stripping ephemeral local paths.
            verdicts = [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "task_package_uri",
                        "report_uri",
                        "skill_package_uri",
                        "skill_report_uri",
                    }
                }
                for row in rows
            ]
            report = {
                "schema_version": 1,
                "producer": self.name,
                "status": status,
                "execution_mode": native_config.mode,
                "simulation_completed": simulated,
                "inputs": inputs,
                "native_results": verdicts,
                "delivered_task_count": len(tasks),
            }
            await object_store.put(report_key, json.dumps(report, ensure_ascii=False).encode())
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
        return TaskResult(status=status, tasks_manifest_key=manifest_key, report_key=report_key)
