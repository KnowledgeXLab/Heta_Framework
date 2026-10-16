"""Data Designer integration: one expanded seed row produces one task result."""

import json
from pathlib import Path

import data_designer.config as dd
from data_designer.config.run_config import RunConfig as DesignerRunConfig
from data_designer.interface import DataDesigner, ResumeMode
from pydantic import BaseModel

from .config import ApiSettings, RunConfig
from .contracts import TaskRequest, TaskResult
from .mock import MockBackend
from .pipeline import produce_task


class GeneratorParams(BaseModel):
    run_dir: str
    config: RunConfig
    env_file: str
    skills_root: str


@dd.custom_column_generator(required_columns=["task_id", "package_uri", "variant"])
def generate_task(row: dict, generator_params: GeneratorParams) -> dict:
    config = generator_params.config
    if config.mode == "mock":
        backend = MockBackend(config.mock_revise_once)
    else:
        from .harbor_backend import HarborBackend

        backend = HarborBackend(
            config,
            ApiSettings.load(Path(generator_params.env_file)),
            Path(generator_params.skills_root),
        )
    request = TaskRequest.model_validate({key: row[key] for key in TaskRequest.model_fields})
    result = produce_task(request, Path(generator_params.run_dir), config, backend)
    return {**row, "task_result": result.model_dump_json()}


def create_dataset(
    run_dir: Path,
    requests: list[TaskRequest],
    config: RunConfig,
    env_file: Path,
    skills_root: Path,
    *,
    submission_dir: Path | None = None,
) -> list[dict]:
    # No scheduling is necessary once every task has a durable terminal result.
    completed = [run_dir / "tasks" / request.task_id / "result.json" for request in requests]
    if completed and all(
        path.is_file()
        and (
            not config.generate_skills
            or json.loads(path.read_text()).get("skill_status") not in {None, "disabled", "pending"}
        )
        for path in completed
    ):
        return [TaskResult.model_validate_json(path.read_text()).model_dump() for path in completed]
    submission_dir = submission_dir or run_dir
    seeds = submission_dir / "expanded-seeds.jsonl"
    seeds.write_text("".join(request.model_dump_json() + "\n" for request in requests))
    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seeds)))
    builder.add_column(
        dd.CustomColumnConfig(
            name="task_result",
            generator_function=generate_task,
            generator_params=GeneratorParams(
                run_dir=str(run_dir),
                config=config,
                env_file=str(env_file),
                skills_root=str(skills_root),
            ),
        )
    )
    designer = DataDesigner()
    designer.set_run_config(
        DesignerRunConfig(
            buffer_size=1,
            max_concurrent_row_groups=config.max_concurrency,
            display_tui=False,
            otel_metrics_port=None,
            non_inference_max_parallel_workers=config.max_concurrency,
            max_in_flight_tasks=config.max_concurrency,
        )
    )
    generated = designer.create(
        builder,
        num_records=len(requests),
        dataset_name="dataset",
        artifact_path=submission_dir / "designer",
        resume=(
            ResumeMode.ALWAYS
            if (submission_dir / "designer" / "dataset").exists()
            else ResumeMode.NEVER
        ),
    )
    rows = generated.load_dataset().to_dict(orient="records")
    expected = {request.task_id for request in requests}
    if len(rows) != len(requests) or {row["task_id"] for row in rows} != expected:
        raise RuntimeError("Data Designer did not return exactly one result per expanded request")
    return sorted((json.loads(row["task_result"]) for row in rows), key=lambda row: row["task_id"])
