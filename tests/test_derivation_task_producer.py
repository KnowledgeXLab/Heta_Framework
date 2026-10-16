import asyncio
import hashlib
import json
import re
import subprocess
import threading
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from heta_framework.common.stores import LocalObjectStore
from heta_framework.generation import (
    HetaGenTaskProducerDerivation,
    HetaGenTaskProducerDerivationConfig,
    ObjectInput,
    TaskProducerProtocol,
    TaskRequest,
)
from heta_framework.generation.derivation.inputs import stage_seeds


@pytest.mark.parametrize("interpreter", ["python", "python3"])
@pytest.mark.parametrize("inverted", [False, True])
def test_docker_heredoc_normalization_preserves_interpreter(interpreter, inverted):
    from heta_framework.generation.derivation._tb.task_factory import _normalize_docker_heredocs

    command = f"<<'PY' {interpreter}" if inverted else f"{interpreter} - <<'PY'"
    files = {"environment/Dockerfile": f"FROM example\nRUN {command}\nprint(42)\nPY\n"}
    normalized = _normalize_docker_heredocs(files)["environment/Dockerfile"]
    assert f"&& {interpreter} /tmp/tb-science-factory-gen-0.py" in normalized
    assert (
        _normalize_docker_heredocs({"environment/Dockerfile": normalized})["environment/Dockerfile"]
        == normalized
    )


@pytest.mark.parametrize(
    ("written_reward", "exit_code", "expected"),
    [("0", 0, "0"), ("1", 0, "1"), ("1", 1, "0"), (None, 0, "0")],
)
def test_verifier_normalization_preserves_explicit_verdicts(
    tmp_path, written_reward, exit_code, expected
):
    from heta_framework.generation.derivation._tb.task_factory import _normalize_runtime_paths

    script = "#!/bin/sh\n"
    if written_reward is not None:
        script += f"printf '%s' '{written_reward}' > /logs/verifier/reward.txt\n"
    script += f"exit {exit_code}\n"
    files = {"tests/test.sh": script}
    # Production normalizes both before validation and during materialization.
    files = _normalize_runtime_paths(_normalize_runtime_paths(files))
    script = files["tests/test.sh"].replace("/logs/verifier", str(tmp_path / "verifier"))
    subprocess.run(["sh", "-c", script], check=False, capture_output=True)
    assert (tmp_path / "verifier/reward.txt").read_text().strip() == expected


def test_fixed_offline_verifier_enforces_network_none(tmp_path):
    """Exercise the real constructor, including capability checks before policy setup."""
    pytest.importorskip("harbor")
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths

    from heta_framework.generation.derivation.docker import DerivationDockerEnvironment

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    offline = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    instance = DerivationDockerEnvironment(
        environment_dir=environment,
        environment_name="main",
        session_id="offline-regression",
        trial_paths=TrialPaths(trial),
        task_env_config=EnvironmentConfig(),
        network_policy=offline,
        phase_network_policies=[offline],
    )
    assert instance.capabilities.disable_internet
    assert not instance.capabilities.dynamic_network_policy
    overlay = json.loads(instance._docker_compose_paths[-1].read_text())
    assert overlay["services"]["main"]["network_mode"] == "none"
    with pytest.raises(ValueError):
        asyncio.run(instance.set_network_policy(NetworkPolicy(network_mode=NetworkMode.PUBLIC)))


async def add_seed(store, name="sample"):
    await store.put(f"seeds/{name}/instruction.md", b"Analyze a public time series.")
    await store.put(f"seeds/{name}/task.toml", b'schema_version="1.4"\n')
    await store.put(f"seeds/{name}/environment/input.csv", b"time,value\n1,3\n2,5\n")
    await store.put(f"seeds/{name}/solution/solve.sh", b"PRIVATE_ORIGINAL_SOLUTION")
    await store.put(f"seeds/{name}/tests/check.py", b"PRIVATE_ORIGINAL_TESTS")


def test_config_and_protocol():
    config = HetaGenTaskProducerDerivationConfig(tasks_per_seed=4)
    producer = HetaGenTaskProducerDerivation(config)
    assert isinstance(producer, TaskProducerProtocol)
    assert producer.config is config
    with pytest.raises(FrozenInstanceError):
        config.tasks_per_seed = 2
    with pytest.raises(AttributeError):
        producer.config = config
    for kwargs in (
        {"tasks_per_seed": 0},
        {"max_concurrency": True},
        {"max_model_restarts": 4},
        {"temperature": float("nan")},
        {"reasoning_effort": "invalid"},
        {"environment_image": "python:3.12\nRUN false"},
        {"environment_image": "--build-arg=invalid"},
        {"environment_image": "${UNDECLARED_IMAGE}"},
    ):
        with pytest.raises(ValueError):
            HetaGenTaskProducerDerivationConfig(**kwargs)


def test_multiple_seed_staging_never_exposes_private_files(tmp_path):
    async def run():
        store = LocalObjectStore(tmp_path / "objects")
        await add_seed(store, "first")
        await add_seed(store, "second")
        rows, inputs = await stage_seeds(
            ObjectInput(prefix="seeds"), store, tmp_path / "stage", "out", 10000
        )
        assert len(rows) == 2
        assert len(inputs) == 10
        assert len({row["seed_task_id"] for row in rows}) == 2
        for row in rows:
            public = json.loads(row["public_task"])
            assert set(public) == {"instruction.md", "environment/input.csv"}
            assert "PRIVATE_ORIGINAL" not in row["public_task"]
        with pytest.raises(ValueError, match="max_public_bytes"):
            await stage_seeds(ObjectInput(prefix="seeds"), store, tmp_path / "small", "out", 10)

    asyncio.run(run())


def test_seed_input_rejects_unrelated_or_nested_objects(tmp_path):
    async def run():
        store = LocalObjectStore(tmp_path / "objects")
        await store.put("raw/paper.txt", b"not a task")
        with pytest.raises(ValueError, match="task.toml"):
            await HetaGenTaskProducerDerivation().produce(
                TaskRequest("out", {"seed_tasks": ObjectInput(prefix="raw")}),
                object_store=store,
            )
        await add_seed(store)
        await store.put("seeds/unrelated.txt", b"unrelated")
        with pytest.raises(ValueError, match="belong"):
            await stage_seeds(ObjectInput(prefix="seeds"), store, tmp_path / "stage", "out", 10000)

    asyncio.run(run())


def test_actual_designer_mock_two_phases(tmp_path):
    pytest.importorskip("data_designer.interface")

    async def run():
        store = LocalObjectStore(tmp_path / "objects")
        await add_seed(store)
        producer = HetaGenTaskProducerDerivation(
            HetaGenTaskProducerDerivationConfig(
                mode="mock",
                tasks_per_seed=3,
                max_concurrency=3,
            )
        )
        request = TaskRequest("out", {"seed_tasks": ObjectInput(prefix="seeds")})
        result = await producer.produce(request, object_store=store)
        report = json.loads(await store.get(result.report_key))
        patterns = json.loads(await store.get(report["patterns_key"]))
        assert len(patterns) == 1
        assert [stage["returned_rows"] for stage in report["stages"]] == [1, 3]
        assert {row["status"] for row in report["native_results"]} == {"simulated"}
        assert len({row["task_id"] for row in report["native_results"]}) == 3
        assert json.loads(await store.get(result.tasks_manifest_key))["tasks"] == []
        assert result.status == "failed"
        with pytest.raises(FileExistsError):
            await producer.produce(request, object_store=store)

    asyncio.run(run())


def test_native_llm_columns_with_local_api_and_partial_validation(tmp_path, monkeypatch):
    """Exercise real DD model columns without paid requests or Docker in this test."""
    pytest.importorskip("data_designer.interface")
    from heta_framework.generation.derivation import designer, validation

    requests = []
    pattern = {
        "workflow": ["Read inputs", "Compute", "Write output"],
        "capabilities": ["Numerical analysis"],
        "scientific_outputs": ["JSON"],
        "hardness_knobs": ["Boundary values"],
        "derivation_rules": ["New inputs"],
        "source_evidence": [
            {"file_path": "instruction.md", "finding": "Public requirements"},
            {"file_path": "environment/input.csv", "finding": "Public input"},
        ],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers.get("Authorization") == "Bearer local-test-key"
            if "max_tokens" in body:
                assert body.get("reasoning_effort") == "low"
            requests.append(body)
            prompt = json.dumps(body["messages"])
            if "Abstract the public Harbor task" in prompt:
                content = pattern
            elif "Create exactly 1" in prompt:
                task_id = re.search(r"derived-[a-f0-9]{16}", prompt).group()
                variant = int(re.search(r"Independent variant: (\d+)", prompt).group(1))
                content = designer.mock_generate({"task_id": task_id, "variant": variant}, None)[
                    "candidate"
                ]
                content["public_files"] = [
                    {"path": "observations.csv", "content": "time,value\n0,2\n1,3\n"}
                ]
            elif "Assess independent task derivation" in prompt:
                content = {"independent_task": True, "preserves_capabilities": True, "findings": []}
            else:
                content = "ready"
            payload = {
                "id": "local-completion",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "```json\n" + json.dumps(content) + "\n```",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    verdicts = []
    verdict_lock = threading.Lock()

    def verify(task, probe, directory, config):
        assert task.is_dir() and probe.is_file()
        public_bytes = (task / "environment/inputs/observations.csv").read_bytes()
        assert public_bytes == b"time,value\n0,2\n1,3\n"
        assert (task / "tests/inputs/observations.csv").read_bytes() == public_bytes
        with verdict_lock:
            passed = not verdicts
            verdicts.append(passed)
        return {
            "execution_mode": "harbor",
            "validated": passed,
            "checks": [
                {
                    "name": name,
                    "status": "passed" if passed else "failed",
                    "reward": reward,
                    "exception_type": None,
                }
                for name, reward in (
                    ("oracle", 1),
                    ("empty_submission", 0),
                    ("wrong_submission", 0),
                )
            ],
        }

    monkeypatch.setattr(validation, "validate_task", verify)

    async def run():
        store = LocalObjectStore(tmp_path / "objects")
        await add_seed(store)
        config = HetaGenTaskProducerDerivationConfig(
            tasks_per_seed=2,
            max_concurrency=2,
            model="local-test",
            api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            request_timeout_sec=10,
            reasoning_effort="low",
            max_model_restarts=0,
        )
        result = await HetaGenTaskProducerDerivation(config).produce(
            TaskRequest("out", {"seed_tasks": ObjectInput(prefix="seeds")}),
            object_store=store,
        )
        report = json.loads(await store.get(result.report_key))
        assert [stage["returned_rows"] for stage in report["stages"]] == [1, 2], report
        assert result.status == "partial"
        manifest = json.loads(await store.get(result.tasks_manifest_key))
        assert len(manifest["tasks"]) == 1
        assert {row["status"] for row in report["native_results"]} == {
            "accepted",
            "validation_failed",
        }
        for file in manifest["tasks"][0]["files"]:
            assert hashlib.sha256(await store.get(file["key"])).hexdigest() == file["sha256"]
        assert "PRIVATE_ORIGINAL" not in json.dumps(requests)
        assert (
            sum(
                "Abstract the public Harbor task" in json.dumps(body["messages"])
                for body in requests
            )
            == 1
        )

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_incorrect_submission_exception_never_counts_as_rejection(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from heta_framework.generation.derivation import validation
    from heta_framework.generation.derivation._tb.task_factory import (
        materialize_candidate,
        normalize_candidate,
    )
    from heta_framework.generation.derivation.designer import mock_candidate

    task = materialize_candidate(
        normalize_candidate(mock_candidate("derived-example", 0)), tmp_path
    )
    probe = tmp_path / "incorrect.sh"
    probe.write_text("exit 1")

    def trial(self, task, directory, agent, **kwargs):
        wrong = "import_path" in agent
        return {
            "verifier_result": {"rewards": {"reward": 1 if agent.get("name") == "oracle" else 0}},
            "exception_info": {"exception_type": "RuntimeError"} if wrong else None,
        }

    monkeypatch.setattr(validation.HarborRunner, "run", trial)
    result = validation.validate_task(
        task, probe, tmp_path / "checks", HetaGenTaskProducerDerivationConfig()
    )
    assert not result["validated"]
    assert result["checks"][-1]["status"] == "failed"
    assert result["checks"][-1]["failure_reason"] == "execution_error"


@pytest.mark.parametrize("failure", ["field_escape", "duplicate_file", "novelty"])
def test_candidate_rejections_do_not_materialize(tmp_path, failure):
    pytest.importorskip("data_designer.interface")
    from heta_framework.generation.derivation.designer import (
        WorkerParams,
        mock_generate,
        prepare_candidate,
    )

    params = WorkerParams(root=str(tmp_path), config={"mode": "mock"})
    row = {
        "task_id": "derived-example",
        "variant": 0,
        "task_pattern": {},
        "public_task": json.dumps({"instruction.md": "Public seed"}),
        "novelty": {"independent_task": True, "preserves_capabilities": True, "findings": []},
    }
    row = mock_generate(row, params)
    if failure == "field_escape":
        row["candidate"]["field"] = "../../escaped"
    elif failure == "duplicate_file":
        row["candidate"]["files"].append(row["candidate"]["files"][0])
    else:
        row["novelty"]["independent_task"] = False
    result = prepare_candidate(row, params)
    assert not result["candidate_check"]["passed"]
    assert result["candidate_check"]["package_path"] is None
    assert not list(tmp_path.glob("**/instruction.md"))


def test_missing_designer_rows_are_reported(tmp_path, monkeypatch):
    pytest.importorskip("data_designer.interface")
    from heta_framework.generation.derivation import designer

    seed = {
        "seed_task_id": "seed-example",
        "seed_sha256": "test-hash",
        "public_task": json.dumps({"instruction.md": "Public seed"}),
    }
    pattern = designer.mock_pattern(seed, None)["task_pattern"]
    calls = []

    def phase(rows, name, root, config, provider):
        calls.append((name, len(rows)))
        if name == "patterns":
            return [
                {**seed, "task_pattern": pattern, "pattern_result": {"passed": True, "errors": []}}
            ], {}
        return [], {"error_type": "ProviderError"}

    monkeypatch.setattr(designer, "run_phase", phase)
    result = designer.execute(
        [seed], tmp_path, HetaGenTaskProducerDerivationConfig(tasks_per_seed=3)
    )
    assert calls == [("patterns", 1), ("derivatives", 3)]
    assert len(result["results"]) == 3
    assert {row["status"] for row in result["results"]} == {"generation_failed"}


def test_controller_emits_valid_harbor_metadata(tmp_path):
    pytest.importorskip("data_designer.interface")
    import tomllib
    from harbor.models.task.config import TaskConfig

    from heta_framework.generation.derivation._tb.task_factory import validate_candidate
    from heta_framework.generation.derivation.designer import (
        WorkerParams,
        mock_generate,
        prepare_candidate,
    )

    params = WorkerParams(
        root=str(tmp_path),
        config={
            "mode": "mock",
            "environment_cpus": 1,
            "environment_memory_mb": 512,
            "validation_timeout_sec": 120,
        },
    )
    row = {
        "task_id": "derived-stable",
        "variant": 0,
        "task_pattern": {},
        "public_task": json.dumps({"instruction.md": "Public seed"}),
        "novelty": {"independent_task": True, "preserves_capabilities": True, "findings": []},
    }
    row = mock_generate(row, params)
    # Metadata is controller-owned, even if a provider includes obsolete fields.
    row["candidate"]["task_toml"] = "invalid legacy TOML"
    row["candidate"]["slug"] = "invalid-over-long-provider-generated-slug"
    result = prepare_candidate(row, params)
    assert result["candidate_check"]["passed"]
    task = Path(result["candidate_check"]["package_path"])
    raw = tomllib.loads((task / "task.toml").read_text())
    assert raw["artifacts"] == ["/workspace/output/result.json"]
    assert raw["task"]["name"] == "terminal-bench-science/derived-stable"
    config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    assert config.environment.cpus == config.verifier.environment.cpus == 1
    assert config.environment.memory_mb == config.verifier.environment.memory_mb == 512
    native = json.loads((tmp_path / "tasks/derived-stable/candidate.json").read_text())
    assert validate_candidate(native) == []


@pytest.mark.parametrize(
    "artifacts",
    [
        ["/solution/solve.sh"],
        ["/workspace/../solution/solve.sh"],
        ["/workspace/out.json", "/workspace/out.json"],
    ],
)
def test_generated_artifact_paths_are_rejected(artifacts):
    from heta_framework.generation.derivation.format import task_toml

    with pytest.raises(ValueError):
        task_toml(
            {"artifacts": artifacts}, "derived-example", HetaGenTaskProducerDerivationConfig()
        )


def test_compiled_package_shares_raw_inputs_and_preserves_code(tmp_path):
    pytest.importorskip("data_designer.interface")
    from heta_framework.generation.derivation._tb.task_factory import materialize_candidate
    from heta_framework.generation.derivation.designer import mock_generate
    from heta_framework.generation.derivation.package import compile_candidate

    candidate = mock_generate({"task_id": "derived-shared", "variant": 0}, None)["candidate"]
    data = "time,value\r\n0,2\r\n1,3\r\n"
    candidate["public_files"] = [{"path": "series/observations.csv", "content": data}]
    candidate["files"].append({"path": "tests/expected.json", "content": '{"value": 2}\n'})
    original = {item["path"]: item["content"] for item in candidate["files"]}
    compiled = compile_candidate(candidate, "derived-shared", HetaGenTaskProducerDerivationConfig())
    task = materialize_candidate(compiled, tmp_path, normalize_files=False)
    for context in ("environment", "tests"):
        assert (task / context / "inputs/series/observations.csv").read_bytes() == data.encode()
        assert "COPY inputs/ /workspace/inputs/" in (task / context / "Dockerfile").read_text()
    for name, content in original.items():
        assert (task / name).read_bytes() == content.encode()
    assert not (task / "environment/solution").exists()
    assert not (task / "environment/check.py").exists()
    assert (task / "tests/test.sh").stat().st_mode & 0o111


@pytest.mark.parametrize(
    ("slot", "name"),
    [
        ("public_files", "../outside.csv"),
        ("public_files", "/absolute.csv"),
        ("public_files", "."),
        ("public_files", "nested//input.csv"),
        ("public_files", ".heta-inputs/child"),
        ("files", "environment/Dockerfile"),
        ("files", "tests/Dockerfile"),
        ("files", "tests/.dockerignore"),
        ("files", "tests/inputs/reference.csv"),
    ],
)
def test_compiled_package_rejects_unsafe_and_controller_owned_paths(slot, name):
    pytest.importorskip("data_designer.interface")
    from heta_framework.generation.derivation.designer import mock_generate
    from heta_framework.generation.derivation.package import compile_candidate

    candidate = mock_generate({"task_id": "derived-paths", "variant": 0}, None)["candidate"]
    candidate[slot].append({"path": name, "content": "untrusted"})
    with pytest.raises(ValueError):
        compile_candidate(candidate, "derived-paths", HetaGenTaskProducerDerivationConfig())


@pytest.mark.parametrize("artifact", ["/workspace/inputs", "/workspace/inputs/observations.csv"])
def test_submitted_artifacts_cannot_replace_trusted_inputs(artifact):
    from heta_framework.generation.derivation.format import task_toml

    with pytest.raises(ValueError):
        task_toml(
            {"artifacts": [artifact]}, "derived-example", HetaGenTaskProducerDerivationConfig()
        )


def test_verification_diagnostics_are_bounded_and_do_not_follow_symlinks(tmp_path):
    from heta_framework.generation.derivation.validation import _diagnostics

    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/test-stdout.txt").write_text("x" * 6000 + "\nwrong initial state")
    (tmp_path / "exception.txt").symlink_to("/etc/hosts")
    logs = _diagnostics({"local_trial_dir": str(tmp_path)})
    assert len(logs) == 1
    assert logs[0]["tail"].endswith("wrong initial state")
    assert len(logs[0]["tail"]) <= 4096
