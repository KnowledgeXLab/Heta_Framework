"""Knowledge-grounded Harbor tasks, with ObjectStore as the durable handoff."""

from __future__ import annotations

import asyncio
import io
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key
from heta_framework.generation.types import (
    InputRequirement,
    TaskRequest,
    TaskRequirements,
    TaskResult,
)
from heta_framework.kb import KnowledgeBase

from .config import HetaGenTaskProducerGraphConfig
from .inputs import digest, encode, freeze, from_objects, stage_snapshot
from .knowledge import from_knowledge_base
from .types import GraphInputError, PreparedGraphInputs


class HetaGenTaskProducerGraph:
    """Produce evidence-grounded tasks through the native Omega column workflow."""

    name = "heta_gen_graph"

    def __init__(self, config: HetaGenTaskProducerGraphConfig | None = None) -> None:
        self._config = config or HetaGenTaskProducerGraphConfig()

    @property
    def config(self) -> HetaGenTaskProducerGraphConfig:
        return self._config

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(
            inputs=(
                (
                    InputRequirement(
                        "query", "Direction used to retrieve source documents", value_type="text"
                    ),
                )
                if self.config.input_mode == "knowledge_base"
                else (
                    InputRequirement("kg", "Omega graph JSON objects paired by basename"),
                    InputRequirement("text", "Full-text UTF-8 objects paired by basename"),
                )
            )
        )

    async def prepare_inputs(
        self,
        request: TaskRequest,
        *,
        object_store: ObjectStoreProtocol,
        knowledge_base: KnowledgeBase | None = None,
    ) -> PreparedGraphInputs:
        """Resolve and freeze inputs without calling the task-generation models."""
        self.requirements.validate(request)
        if await object_store.list(request.output_prefix + "/"):
            raise FileExistsError(f"output prefix is already occupied: {request.output_prefix}")
        if self.config.input_mode == "knowledge_base":
            if knowledge_base is None:
                raise GraphInputError("knowledge_base is required in knowledge_base mode")
            query = request.inputs["query"]
            assert isinstance(query, str)
            documents, provenance = await from_knowledge_base(knowledge_base, query, self.config)
        else:
            if knowledge_base is not None:
                raise GraphInputError("objects mode does not accept a knowledge_base")
            documents, provenance = await from_objects(request, object_store, self.config)
        return await freeze(documents, provenance, request, object_store)

    async def produce(
        self,
        request: TaskRequest,
        *,
        object_store: ObjectStoreProtocol,
        knowledge_base: KnowledgeBase | None = None,
    ) -> TaskResult:
        prepared = await self.prepare_inputs(
            request, object_store=object_store, knowledge_base=knowledge_base
        )
        tasks = []
        produced = {
            "complete": True,
            "stages": {},
            "rows": [],
            "rejections": [],
            "stage_row_counts": {},
            "reason": "no_query_matches",
        }
        if self.config.workspace:
            self.config.workspace.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.config.workspace, prefix="heta-graph-") as directory:
            root = Path(directory).resolve()
            secret = ""
            if prepared.document_count:
                try:
                    from data_designer.config import ModelProvider
                    from dotenv import dotenv_values

                    from .designer import accepted_package, execute
                except ModuleNotFoundError as error:
                    raise ImportError("Install heta-framework[generation] to use Graph") from error
                values = {**dotenv_values(self.config.env_file), **os.environ}
                secret = values.get("OPENAI_API_KEY") or ""
                endpoint = self.config.api_base_url or values.get("OPENAI_BASE_URL")
                if not secret or not endpoint:
                    raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL/api_base_url are required")
                provider = ModelProvider(
                    name="graph", endpoint=endpoint, provider_type="openai", api_key=secret
                )
                rows = await stage_snapshot(prepared, object_store, root / "inputs")
                future = asyncio.create_task(
                    asyncio.to_thread(execute, rows, root / "designer", self.config, provider)
                )
                try:
                    produced = await asyncio.shield(future)
                except asyncio.CancelledError:
                    await asyncio.shield(future)
                    raise
                # A partial workflow never supplies publishable completed task rows.
                if produced["complete"]:
                    for row in produced["rows"]:
                        from harbor.models.task.config import TaskConfig

                        try:
                            package = accepted_package(row, root, produced["oracle_repeats"])
                            if package is None:
                                continue
                            TaskConfig.model_validate_toml((package / "task.toml").read_text())
                            paths = sorted(package.rglob("*"))
                            if any(path.is_symlink() for path in paths):
                                raise ValueError("task publication does not support symbolic links")
                        except (ValueError, OSError) as error:
                            row["publication_error"] = f"{type(error).__name__}: {error}"
                            continue
                        if secret and any(
                            secret.encode() in path.read_bytes() for path in paths if path.is_file()
                        ):
                            row["publication_error"] = "credential_in_package"
                            continue
                        task_id = digest(str(package.relative_to(root)).encode())[:24]
                        files = []
                        # Preserve the complete reproducible native package, including generator.
                        # Local validation logs/cache stay outside the task file collection.
                        for path in paths:
                            if not path.is_file():
                                continue
                            relative = path.relative_to(package).as_posix()
                            if relative.split("/")[0] == "authoring":
                                continue
                            key = join_object_key(
                                request.output_prefix, f"tasks/{task_id}/{relative}"
                            )
                            raw = path.read_bytes()
                            await object_store.put(key, raw)
                            files.append(
                                {
                                    "path": relative,
                                    "key": key,
                                    "sha256": digest(raw),
                                    "mode": 0o755 if path.suffix == ".sh" else 0o644,
                                }
                            )
                        evidence_key = join_object_key(
                            request.output_prefix, f"evidence/{task_id}/validation.json"
                        )
                        evidence = Path(row["validation_report_path"]).read_bytes()
                        await object_store.put(
                            evidence_key, evidence.replace(str(root).encode(), b"<workspace>")
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "paper_id": row.get("paper_id"),
                                "tier": row.get("tier"),
                                "files": files,
                                "validation_kind": "omega_harbor_checks",
                                "evidence_keys": [evidence_key],
                            }
                        )
            failed_rows = bool(produced.get("failures")) or any(
                row.get("terminal_bench_ready") is not True or row.get("publication_error")
                for row in produced["rows"]
            )
            status = (
                "succeeded"
                if produced["complete"] and not failed_rows
                else "partial"
                if tasks
                else "failed"
            )
            report_key = join_object_key(request.output_prefix, "report.json")
            manifest_key = join_object_key(request.output_prefix, "tasks.json")
            archive_key = None
            if prepared.document_count:
                # Retain native checkpoint truth beyond this temporary workspace.
                archive_key = join_object_key(request.output_prefix, "evidence/designer.zip")
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted((root / "designer").rglob("*")):
                        if not path.is_file() or path.is_symlink():
                            continue
                        if path.suffix not in {".json", ".parquet"}:
                            continue
                        relative = path.relative_to(root / "designer")
                        # Package authoring/trials have their own published validation report.
                        if "task_packages" in relative.parts:
                            continue
                        raw = path.read_bytes()
                        if secret and path.suffix == ".parquet":
                            import pandas as pd

                            if secret in pd.read_parquet(path).to_json(orient="records"):
                                continue
                        elif secret:
                            raw = raw.replace(secret.encode(), b"[REDACTED_API_KEY]")
                        archive.writestr(relative.as_posix(), raw)
                await object_store.put(archive_key, buffer.getvalue())
            report = {
                "schema_version": 1,
                "producer": self.name,
                "status": status,
                "inputs_manifest_key": prepared.manifest_key,
                "input_document_count": prepared.document_count,
                "delivered_task_count": len(tasks),
                "designer_evidence_key": archive_key,
                "native_revision": "1fadf09c6b81acd3c33554864102c1ca5220d4f4",
                "config": {
                    "input_mode": self.config.input_mode,
                    "model": self.config.model,
                    "review_model": self.config.review_model,
                    "reasoning_effort": self.config.reasoning_effort,
                    "max_concurrency": self.config.max_concurrency,
                    "max_tokens": self.config.max_tokens,
                    "request_timeout_sec": self.config.request_timeout_sec,
                    "generation_guidance": self.config.generation_guidance,
                },
                "native": produced,
            }
            raw = encode(report).replace(str(root).encode(), b"<workspace>")
            if secret:
                raw = raw.replace(secret.encode(), b"[REDACTED_API_KEY]")
            await object_store.put(report_key, raw)
            await object_store.put(
                manifest_key,
                encode(
                    {
                        "schema_version": 1,
                        "format": "harbor",
                        "producer": self.name,
                        "report_key": report_key,
                        "tasks": tasks,
                    }
                ),
            )
        return TaskResult(status, manifest_key, report_key)
