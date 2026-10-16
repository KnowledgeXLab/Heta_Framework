"""Regression coverage for heterogeneous static results and scheduler row loss."""

import json

import pytest


@pytest.mark.parametrize("already_encoded", [False, True])
def test_static_runtime_seed_preserves_mixed_verdicts(tmp_path, already_encoded):
    pytest.importorskip("data_designer")
    pd = pytest.importorskip("pandas")
    duckdb = pytest.importorskip("duckdb")
    from heta_framework.generation.graph._native.dd.workflow import make_runtime_seed_callback

    stage = tmp_path / "static"
    inputs = stage / "parquet-files"
    inputs.mkdir(parents=True)
    expected = {}
    for index in range(8):
        failed = ["check-reward", "check-canary"] if index % 3 == 1 else []
        summary = {
            "passed": not failed,
            "attempted": 24,
            "failed": failed,
            "warnings": {"check": ["first", "second"] if index % 2 else []},
        }
        identity = f"paper-{index}"
        expected[identity] = summary
        pd.DataFrame(
            [
                {
                    "paper_id": identity,
                    "tier": "medium",
                    "package_dir": f"packages/{identity}",
                    "static_passed": not failed,
                    "static_summary": json.dumps(summary) if already_encoded else summary,
                }
            ]
        ).to_parquet(inputs / f"batch_{index:05}.parquet", index=False)

    callback = make_runtime_seed_callback(version="regression")
    output = callback(stage)
    with duckdb.connect() as connection:
        table = connection.read_parquet(str(output / "*.parquet")).to_arrow_table()
    records = table.to_pylist()
    assert len(records) == 8
    assert {r["paper_id"] for r in records} == set(expected)
    for row in records:
        assert isinstance(row["static_summary"], str)
        assert json.loads(row["static_summary"]) == expected[row["paper_id"]]
        assert row["static_passed"] == expected[row["paper_id"]]["passed"]
    # Original stage evidence is never rewritten by the boundary callback.
    original = pd.read_parquet(inputs / "batch_00001.parquet").iloc[0]["static_summary"]
    assert isinstance(original, str) == already_encoded


def test_runtime_seed_callback_is_versioned_in_workflow(tmp_path):
    pytest.importorskip("data_designer")
    pd = pytest.importorskip("pandas")
    from heta_framework.generation.graph._native.config import Settings
    from heta_framework.generation.graph._native.dd.workflow import build_workflow

    class Workflow:
        def __init__(self):
            self.stages = {}

        def add_stage(self, name, builder, **kwargs):
            self.stages[name] = kwargs

    workflow = Workflow()
    pd.DataFrame([{"paper_id": "paper"}]).to_parquet(tmp_path / "seed.parquet", index=False)

    class Designer:
        def compose_workflow(self, **kwargs):
            return workflow

    build_workflow(
        Designer(),
        Settings(),
        seed_path=tmp_path / "seed.parquet",
        model_configs=[],
        workflow_name="test",
        packages_root=tmp_path / "tasks",
    )
    stage = workflow.stages["validate_static"]
    assert stage["on_success_version"] == "static-runtime-seed-v1"
    assert callable(stage["on_success"])


def test_record_loss_reports_identity_and_tier_but_not_gate_rejection(tmp_path):
    pd = pytest.importorskip("pandas")
    from heta_framework.generation.graph.designer import stage_record_losses

    run = tmp_path / "graph"
    inputs = run / "static-seed"
    inputs.mkdir(parents=True)
    rows = [
        {"paper_id": "paper", "tier": tier, "static_passed": tier != "simple"}
        for tier in ["simple", "medium", "complex"]
    ]
    pd.DataFrame(rows).to_parquet(inputs / "data.parquet", index=False)
    metadata = {
        "stages": [
            {"name": "validate_static", "status": "completed", "num_records_requested": 3},
            {
                "name": "validate_runtime",
                "status": "completed",
                "num_records_requested": 3,
                "seed_path": "static-seed",
            },
            {"name": "unneeded", "status": "skipped_empty_upstream", "num_records_requested": 3},
        ]
    }
    # The failed static verdict is present, so it is not a lost row.
    frames = {"validate_static": rows, "validate_runtime": rows[:2]}
    losses = stage_record_losses(frames, metadata, run)
    assert losses == [
        {
            "stage": "validate_runtime",
            "requested": 3,
            "actual": 2,
            "missing_count": 1,
            "missing_rows": [{"paper_id": "paper", "tier": "complex"}],
        }
    ]
    frames["validate_runtime"] = rows
    assert stage_record_losses(frames, metadata, run) == []


def test_execution_report_retains_healthy_rows_and_records_scheduler_loss(tmp_path, monkeypatch):
    pytest.importorskip("data_designer")
    pd = pytest.importorskip("pandas")
    from data_designer import interface

    from heta_framework.generation import HetaGenTaskProducerGraphConfig
    from heta_framework.generation.graph import designer
    from heta_framework.generation.graph._native.dd import workflow as native

    rows = [{"paper_id": f"paper-{i}", "tier": "medium"} for i in range(3)]

    class Designer:
        def __init__(self, **kwargs):
            pass

        def set_run_config(self, config):
            pass

    class Workflow:
        def run(self):
            stages = []
            run = tmp_path / "graph"
            for index, name in enumerate(native.STAGE_NAMES):
                directory = run / f"stage-{index}-{name}" / "parquet-files"
                directory.mkdir(parents=True)
                pd.DataFrame(rows[:2] if name == "validate_runtime" else rows).to_parquet(
                    directory / "data.parquet", index=False
                )
                stages.append(
                    {
                        "name": name,
                        "status": "completed",
                        "num_records_requested": 3,
                        "seed_path": "stage-6-validate_static/parquet-files"
                        if name == "validate_runtime"
                        else None,
                    }
                )
            (run / "workflow-metadata.json").write_text(json.dumps({"stages": stages}))

    monkeypatch.setattr(interface, "DataDesigner", Designer)
    monkeypatch.setattr(native, "build_workflow", lambda *args, **kwargs: Workflow())
    result = designer.execute(rows, tmp_path, HetaGenTaskProducerGraphConfig(), None)
    assert result["complete"] is True
    assert result["records_complete"] is False
    assert len(result["rows"]) == 2
    assert result["failures"] == [
        {
            "stage": "validate_runtime",
            "paper_id": None,
            "column": "stage_records",
            "detail": result["stage_record_losses"][0],
        }
    ]
    assert result["stage_record_losses"][0]["missing_rows"] == [
        {"paper_id": "paper-2", "tier": "medium"}
    ]


def test_failed_static_rows_survive_the_real_runtime_scheduler(tmp_path, monkeypatch):
    pytest.importorskip("data_designer")
    pd = pytest.importorskip("pandas")
    import data_designer.config as dd
    from data_designer.config.run_config import RunConfig
    from data_designer.interface import DataDesigner

    from heta_framework.generation.graph._native.config import Settings
    from heta_framework.generation.graph._native.dd.workflow import (
        make_runtime_seed_callback,
        read_stage_frame,
        stage_validate_runtime,
    )
    from heta_framework.generation.graph._native.domain import validation_runtime

    stage = tmp_path / "static"
    (stage / "parquet-files").mkdir(parents=True)
    for index in range(8):
        failed = index % 2 == 0
        pd.DataFrame(
            [
                {
                    "paper_id": f"paper-{index}",
                    "package_dir": str(tmp_path / "packages" / f"paper-{index}"),
                    "static_passed": not failed,
                    "static_summary": {
                        "passed": not failed,
                        "attempted": 24,
                        "failed": ["check-canary"] if failed else [],
                    },
                }
            ]
        ).to_parquet(stage / "parquet-files" / f"batch_{index}.parquet", index=False)
    seed = make_runtime_seed_callback(version="test")(stage)
    monkeypatch.setattr(
        validation_runtime,
        "materialization_phase",
        lambda *args, **kwargs: {
            "ok": False,
            "errors": ["fixture materialization failure"],
            "deterministic": None,
        },
    )
    monkeypatch.setattr(
        validation_runtime,
        "harbor_phase",
        lambda *args, **kwargs: pytest.fail("Harbor must not run when materialization failed"),
    )
    builder = stage_validate_runtime(Settings(), [])
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed / "data.parquet")))
    designer = DataDesigner(artifact_path=tmp_path / "designer")
    designer.set_run_config(
        RunConfig(
            buffer_size=1,
            max_concurrent_row_groups=8,
            non_inference_max_parallel_workers=8,
            display_tui=False,
            otel_metrics_port=None,
        )
    )
    workflow = designer.compose_workflow(name="negative_paths")
    workflow.add_stage("validate_runtime", builder, num_records=8, allow_empty=True)
    workflow.run()
    output = read_stage_frame(tmp_path / "designer/negative_paths/stage-0-validate_runtime")
    rows = json.loads(output.to_json(orient="records"))
    assert len(rows) == 8
    assert {row["paper_id"] for row in rows} == {f"paper-{i}" for i in range(8)}
    assert all(row["terminal_bench_ready"] is False for row in rows)
    assert all(row["validation_report_path"] for row in rows)


@pytest.mark.parametrize("column", ["runtime_materialize", "harbor_validation"])
def test_runtime_early_returns_always_supply_the_primary_column(column):
    pytest.importorskip("data_designer")
    from heta_framework.generation.graph._native.config import Settings
    from heta_framework.generation.graph._native.dd import validation

    row = getattr(validation, column)({"package_dir": ""}, Settings())
    assert row[column] == "failed"
