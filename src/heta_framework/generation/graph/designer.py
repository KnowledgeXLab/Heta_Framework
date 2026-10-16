"""Explicit model configuration around the pinned Omega Data Designer workflow."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .inputs import encode


def stage_record_losses(frames: dict, metadata: dict, run: Path) -> list[dict]:
    """Report scheduler row loss separately from intentional boundary filtering."""
    import pandas as pd

    losses = []
    for stage in metadata.get("stages", []):
        if stage.get("status") not in {"completed", "completed_empty"}:
            continue
        requested = stage.get("num_records_requested")
        actual = frames.get(stage["name"], [])
        if type(requested) is not int or len(actual) >= requested:
            continue
        loss = {
            "stage": stage["name"],
            "requested": requested,
            "actual": len(actual),
            "missing_count": requested - len(actual),
            "missing_rows": [],
        }
        seed = run / stage["seed_path"] if stage.get("seed_path") else run.parent / "seed.parquet"
        try:
            paths = [seed] if seed.is_file() else sorted(seed.glob("*.parquet"))
            expected = []
            for path in paths:
                expected.extend(json.loads(pd.read_parquet(path).to_json(orient="records")))

            def identity(row):
                return (row.get("paper_id"), row.get("tier"))

            missing = Counter(map(identity, expected)) - Counter(map(identity, actual))
            loss["missing_rows"] = [
                {"paper_id": paper, "tier": tier}
                for (paper, tier), count in missing.items()
                for _ in range(count)
            ]
        except (OSError, ValueError, KeyError) as error:
            loss["input_read_error"] = f"{type(error).__name__}: {error}"
        losses.append(loss)
    return losses


def execute(rows, root: Path, config, provider):
    import data_designer.config as dd
    import pandas as pd
    from data_designer.config.run_config import RunConfig
    from data_designer.interface import DataDesigner

    from ._native.config import Settings
    from ._native.dd.workflow import STAGE_NAMES, build_workflow, read_stage_frame

    settings = Settings.load(config.settings_file).model_copy(
        update={"generation_guidance": config.generation_guidance}
    )
    root.mkdir(parents=True, exist_ok=True)
    seed = root / "seed.parquet"
    pd.DataFrame(rows).to_parquet(seed, index=False)
    models = [
        dd.ModelConfig(
            alias=alias,
            model=(config.review_model or config.model) if alias == "cheap" else config.model,
            provider="graph",
            skip_health_check=True,
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=config.max_tokens,
                timeout=config.request_timeout_sec,
                max_parallel_requests=config.max_concurrency,
                extra_body={"reasoning_effort": config.reasoning_effort}
                if config.reasoning_effort
                else None,
            ),
        )
        for alias in (
            "cheap",
            "strong",
            "strong_creative",
            "strong_deterministic",
            "strong_no_retry",
        )
    ]
    designer = DataDesigner(artifact_path=root, model_providers=[provider])
    designer.set_run_config(
        RunConfig(
            buffer_size=1,
            max_concurrent_row_groups=config.max_concurrency,
            non_inference_max_parallel_workers=config.max_concurrency,
            display_tui=False,
            otel_metrics_port=None,
        )
    )
    workflow = build_workflow(
        designer,
        settings,
        seed_path=seed,
        model_configs=models,
        workflow_name="graph",
        packages_root=root / "graph" / "task_packages",
        num_records=len(rows),
        paper_namespaced=True,
    )
    error = None
    try:
        workflow.run()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    run = root / "graph"
    metadata_path = run / "workflow-metadata.json"
    metadata = json.loads(metadata_path.read_bytes()) if metadata_path.exists() else {}
    stages = {stage["name"]: stage["status"] for stage in metadata.get("stages", [])}
    completed = {"completed", "completed_empty", "skipped_empty_upstream"}
    complete = error is None and all(stages.get(name) in completed for name in STAGE_NAMES)
    frames = {}
    for name in STAGE_NAMES:
        if stages.get(name) not in {"completed", "completed_empty"}:
            continue
        paths = sorted(run.glob(f"stage-*-{name}"))
        if len(paths) == 1:
            frame = read_stage_frame(paths[0])
            frames[name] = json.loads(frame.to_json(orient="records", double_precision=15))
    ledgers = []
    for path in sorted(run.glob("stage-*/rejected_*.json")):
        ledgers.append(
            {"stage": path.parent.name, "kind": path.stem, "detail": json.loads(path.read_bytes())}
        )
    losses = stage_record_losses(frames, metadata, run)
    result = {
        "complete": complete,
        "records_complete": not losses,
        "stage_record_losses": losses,
        "error": error,
        "stages": stages,
        "rows": frames.get("validate_runtime", []),
        "rejections": ledgers,
        "stage_row_counts": {key: len(value) for key, value in frames.items()},
        "oracle_repeats": settings.validation.oracle_repeats,
        "settings": settings.dump(),
        "failures": [
            {"stage": name, "paper_id": row.get("paper_id"), "column": column, "detail": value}
            for name, records in frames.items()
            for row in records
            for column, value in row.items()
            if column.endswith("_failure") and value
        ]
        + [
            {"stage": loss["stage"], "paper_id": None, "column": "stage_records", "detail": loss}
            for loss in losses
        ],
    }
    (root / "summary.json").write_bytes(encode(result))
    return result


def accepted_package(row, root: Path, repeats: int):
    """Require completed-row gates and independently persisted runtime evidence to agree."""
    if row.get("terminal_bench_ready") is not True:
        return None
    package = Path(row.get("package_dir", "")).resolve()
    if not package.is_relative_to(root.resolve()) or package == root.resolve():
        raise ValueError("native package path escapes workspace")
    report_path = Path(row.get("validation_report_path", ""))
    if report_path.resolve() != package / "authoring" / "validation" / "validation_report.json":
        raise ValueError("native validation report is not paired with package")
    report = json.loads(report_path.read_bytes())
    if report.get("errors") or report.get("materialize_errors"):
        raise ValueError("validation report contains runtime errors")
    gates = ("static_passed", "materialize_ok", "harbor_passed", "deterministic")
    if any(row.get(gate) is not True for gate in gates):
        raise ValueError("ready row lacks complete successful gates")
    if any(
        report.get(gate) is not True
        for gate in ("terminal_bench_ready", "static_passed", "data_materialized", "deterministic")
    ):
        raise ValueError("row and validation report disagree")
    if row.get("oracle_rewards") != [1.0] * repeats or row.get("nop_reward") != 0.0:
        raise ValueError("ready row has invalid Oracle/Nop rewards")
    for agent, expected, count in (("oracle", 1.0, repeats), ("nop", 0.0, 1)):
        detail = (report.get("harbor") or {}).get(agent, {})
        runs = detail.get("runs", [])
        if (
            detail.get("passed") is not True
            or len(runs) != count
            or any(
                run.get("returncode") != 0
                or run.get("completed") is not True
                or type(run.get("reward")) not in (int, float)
                or run["reward"] != expected
                or run.get("error")
                for run in runs
            )
        ):
            raise ValueError(f"validation report lacks successful {agent} trial evidence")
    return package
