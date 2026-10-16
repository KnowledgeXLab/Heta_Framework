"""Object-store handoff and migrated NeMo execution, without paid API calls."""

import asyncio
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.generation import ObjectInput, TaskProducerProtocol, TaskRequest  # noqa: E402
from heta_framework.generation.exploration import (  # noqa: E402
    HetaGenTaskProducerExploration,
    HetaGenTaskProducerExplorationConfig,
)


def test_config_is_typed_validated_and_exposed_read_only():
    config = HetaGenTaskProducerExplorationConfig(tasks_per_package=2)
    producer: TaskProducerProtocol[HetaGenTaskProducerExplorationConfig] = (
        HetaGenTaskProducerExploration(config)
    )
    assert producer.config is config
    assert producer.name == "heta_gen_exploration"
    with pytest.raises(AttributeError):
        producer.config = HetaGenTaskProducerExplorationConfig()
    with pytest.raises(FrozenInstanceError):
        config.tasks_per_package = 3
    with pytest.raises(ValueError, match="max_concurrency"):
        HetaGenTaskProducerExplorationConfig(max_concurrency=0)
    with pytest.raises(TypeError):
        HetaGenTaskProducerExplorationConfig(settings={"tasks_per_package": 2})


def test_source_contract_rejects_missing_package_before_loading_dependencies(tmp_path):
    async def run():
        objects = LocalObjectStore(tmp_path / "objects")
        await objects.put("raw/paper.txt", b"Scientific source material")
        producer = HetaGenTaskProducerExploration()
        assert isinstance(producer, TaskProducerProtocol)
        with pytest.raises(ValueError, match="PACKAGE.md"):
            await producer.produce(
                TaskRequest("out", {"materials": ObjectInput(prefix="raw")}),
                object_store=objects,
            )
        assert not await objects.list("out/")

    asyncio.run(run())


def test_real_designer_mock_does_not_publish_scientific_tasks(tmp_path):
    pytest.importorskip("data_designer.interface")

    async def run():
        objects = LocalObjectStore(tmp_path / "objects")
        await objects.put("raw/PACKAGE.md", b"# Source\nUse methods.txt.\n")
        await objects.put("raw/methods.txt", b"Scientific method notes")
        producer = HetaGenTaskProducerExploration(
            HetaGenTaskProducerExplorationConfig(
                mode="mock", tasks_per_package=2, max_concurrency=2, mock_revise_once=True
            )
        )
        result = await producer.produce(
            TaskRequest("out", {"materials": ObjectInput(prefix="raw")}),
            object_store=objects,
        )
        manifest = json.loads(await objects.get(result.tasks_manifest_key))
        report = json.loads(await objects.get(result.report_key))
        assert manifest["tasks"] == []
        assert report["simulation_completed"] is True
        assert len(report["native_results"]) == 2
        assert all(row["revision"] == 1 for row in report["native_results"])
        assert await objects.get("raw/methods.txt") == b"Scientific method notes"
        with pytest.raises(FileExistsError):
            await producer.produce(
                TaskRequest("out", {"materials": ObjectInput(prefix="raw")}),
                object_store=objects,
            )

    asyncio.run(run())


def test_accepted_native_tasks_are_published_with_evidence(tmp_path, monkeypatch):
    pytest.importorskip("data_designer.interface")
    pytest.importorskip("harbor")
    from heta_framework.generation.exploration._native import designer, preflight
    from heta_framework.generation.exploration._native.config import ApiSettings

    def fake_dataset(run_dir, requests, config, env_file, skills):
        task = run_dir / "tasks/p000-task-000/revision-000/author/tasks/p000-task-000/task"
        files = {
            "instruction.md": "Write the output.",
            "task.toml": 'schema_version="1.4"\n[environment]\ndocker_image="python:3.12-slim"\n',
            "solution/solve.sh": "#!/bin/sh\nexit 0\n",
            "tests/test.sh": "#!/bin/sh\nexit 0\n",
        }
        for relative, content in files.items():
            path = task / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        evidence = run_dir / "tasks/p000-task-000/revision-000/builder/validation.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text('{"checks":[{"name":"oracle","status":"passed"}]}')
        return [
            {
                "task_id": "p000-task-000",
                "status": "accepted",
                "execution_mode": "harbor",
                "validated": True,
                "task_package_uri": str(task),
                "report_uri": "local.json",
                "candidate_revision": "test-revision",
            }
        ]

    monkeypatch.setattr(designer, "create_dataset", fake_dataset)
    monkeypatch.setattr(preflight, "check", lambda *args: {})
    monkeypatch.setattr(ApiSettings, "load", lambda *args: object())

    async def run():
        objects = LocalObjectStore(tmp_path / "objects")
        await objects.put("raw/PACKAGE.md", b"# Source")
        # Injected native result tests publication, not actual Harbor execution.
        producer = HetaGenTaskProducerExploration(
            HetaGenTaskProducerExplorationConfig(mode="harbor")
        )
        result = await producer.produce(
            TaskRequest("out", {"materials": ObjectInput(prefix="raw")}),
            object_store=objects,
        )
        manifest = json.loads(await objects.get(result.tasks_manifest_key))
        task = manifest["tasks"][0]
        assert result.status == "succeeded"
        assert task["evidence_keys"]
        script = next(item for item in task["files"] if item["path"] == "solution/solve.sh")
        assert script["mode"] == 0o755
        assert await objects.get(script["key"]) == b"#!/bin/sh\nexit 0\n"
        assert await objects.get(task["evidence_keys"][0])

    asyncio.run(run())
