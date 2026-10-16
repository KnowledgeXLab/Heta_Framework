# HetaGen Task Producer Derivation

`HetaGenTaskProducerDerivation` derives independent Harbor tasks from existing task packages. It implements `TaskProducerProtocol[HetaGenTaskProducerDerivationConfig]` with seed tasks and configuration as inputs.

## Install

Use Python 3.12 and the same environment:

```bash
pip install "heta-framework[generation]"
```

Production needs Docker/Compose/Buildx and an OpenAI-compatible Chat Completions service supporting Data Designer's structured-output prompts. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env` or the environment. Exported variables take precedence. Configure `model` explicitly; use inference parameters supported by your provider. The model generates file contents directly.

## Usage

Replace `your-model` with a model identifier provided by your endpoint, and use only inference parameters supported by that provider.

Prepare `seed-tasks/`, with one Harbor task per subdirectory, containing `instruction.md`, `task.toml`, `environment/`, `solution/`, and `tests/`.

```python
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from heta_framework.common.stores import LocalObjectStore
from heta_framework.generation import (
    HetaGenTaskProducerDerivation,
    HetaGenTaskProducerDerivationConfig,
    ObjectInput,
    TaskRequest,
)

async def main():
    objects = LocalObjectStore("heta-workspace/objects")
    try:
        source = Path("seed-tasks")
        for file in sorted(source.rglob("*")):
            if file.is_file():
                await objects.put(
                    f"sources/harbor-tasks/{file.relative_to(source).as_posix()}",
                    file.read_bytes(),
                )

        producer = HetaGenTaskProducerDerivation(
            HetaGenTaskProducerDerivationConfig(
                model="your-model",
                reasoning_effort="low",
                tasks_per_seed=4,
                max_concurrency=4,
                generation_guidance="Preserve capabilities; change the scenario and synthetic data",
            )
        )
        result = await producer.produce(
            TaskRequest(
                inputs={"seed_tasks": ObjectInput(prefix="sources/harbor-tasks/")},
                output_prefix=f"tasks/derivation/{uuid4().hex}/",
            ),
            object_store=objects,
        )
        manifest = json.loads(await objects.get(result.tasks_manifest_key))
        report = json.loads(await objects.get(result.report_key))
        print(result.status, len(manifest["tasks"]))
        print(report["native_results"])
        print(report["patterns_key"])
    finally:
        await objects.aclose()

asyncio.run(main())
```

The example calls the model and Docker. Use `mode="mock"` to exercise both real Data Designer workflows without model or Docker calls. Mock results never publish formal scientific tasks: the manifest stays empty and the report sets `simulation_completed` to true.

If seeds already exist in the Object Store, skip uploading. Requested counts are per seed; inspect the manifest for actual delivered tasks. Each file entry includes its object key, SHA-256 and mode; restore executable modes when downloading.

## Inputs

The required `seed_tasks` slot selects one or more task directories. Each root is identified by `task.toml` and must include `instruction.md`. Nested roots and unrelated selected files are rejected. The output prefix must be empty.

Models see public instructions and `environment/` files only. Binary materials are represented by names, sizes and hashes. Oversized public text is rejected rather than silently truncated. Seed `task.toml`, solutions and hidden tests are excluded from model prompts. Users remain responsible for checking whether public environment files already leak answers.

## Data Designer Workflow

There are two workflows with different row granularity:

1. One seed per row: structured `task_pattern` extraction, then `pattern_result` checks public evidence paths. Each source-evidence entry contains an exact public `file_path` and a separate explanatory `finding`.
2. One derivative per row: structured `candidate` generation and `novelty` assessment, then custom `candidate_check`, `task_package` and `task_result` columns.

Passing patterns expand into variants once. Request identifiers and variants enter generation prompts; each request has its own workspace. Data Designer manages model calls, column dependencies and batch concurrency. A deterministic compiler prepares the package before TB Factory validation and materialization.

The model supplies task content, public data, the reference solution, tests and output paths. Internal candidates declare raw public data as `{path, content}` entries in `public_files`; `files` contains only `solution/` and `tests/` content. The compiler owns identifiers, `task.toml` and both Dockerfiles. Users still provide only seed tasks and configuration.

Identical public bytes are materialized under `environment/inputs/` and `tests/inputs/`, available at `/workspace/inputs/` in both containers. The separate verifier reads trusted inputs from its own image; contestant edits cannot change its reference inputs. Public inputs cannot be declared as submitted artifacts. CSV/JSON remains valid data without canary comments. Reference solutions and hidden tests stay outside the contestant image.

Outputs must be under `/workspace/` or `/app/`, excluding `/workspace/inputs/`. Entrypoints are `solution/solve.sh` and `tests/test.sh`. Declared code and data are preserved as generated. Actual Harbor checks still determine acceptance.

Novelty assessment checks retained capabilities and independence. Direct reuse of substantive public files and identical published public task content within the batch are rejected. This does not guarantee semantic deduplication, calibrated difficulty or scientific quality.

## Configuration

The frozen configuration class is separate from the request; `producer.config` is read-only.

| Setting | Default | Purpose |
| --- | --- | --- |
| `tasks_per_seed` | 1 | Requested derivatives per seed, 1–100 |
| `max_concurrency` | 1 | Model request and custom task concurrency |
| `model` | `bailian/glm-5.3` | Chat Completions model |
| `generation_guidance` | empty | Generation preferences |
| `temperature`, `reasoning_effort` | 0.2, None | Optional inference parameters; None omits them |
| `max_tokens` | 32000 | Per-request output budget |
| `request_timeout_sec` | 300 | Model request timeout |
| `max_model_restarts` | 1 | Data Designer conversation restart budget, 0–3 |
| `validation_timeout_sec` | 1800 | Per-check Harbor timeout |
| `environment_cpus`, `environment_memory_mb` | 2, 4096 | Main and separate verifier budgets |
| `environment_image` | `python:3.12-slim` | Common base image with required dependencies preinstalled |
| `max_public_bytes` | 128000 | Public text byte limit per seed |
| `env_file`, `workspace` | `.env`, None | Credentials file and temporary workspace parent |

`api_base_url` can override the endpoint. `docker_builder`, `job_overhead_timeout_sec` and `worker_stop_grace_sec` control local job execution and cleanup. Temporary staging is removed when the call finishes; cross-call resume is not exposed .

## Results and Validation

Only tasks passing actual Harbor checks are published: Oracle reward 1, empty-submission reward 0, representative incorrect-submission reward 0, and no trial execution exceptions. A crashing negative script does not count as successful rejection. Verification containers receive no generation credentials.

The separate verifier stays offline throughout execution. The Docker adapter enforces `network_mode=none` without host firewall support; dynamic network policy switching is not supported.

`patterns.json` retains extracted patterns. `report.json` records stages, requested and delivered counts, configuration, candidate checks, novelty assessments and per-task outcomes, including missing generation rows. Validation and novelty evidence are stored separately from task files.

Rejected checks include `failure_reason` and bounded log excerpts. `reference_solution_rejected` means Oracle failed, `invalid_submission_accepted` means a negative submission was accepted, and `execution_error` means execution failed. Diagnostic logs retain the concrete missing file or numerical disagreement.

The result status is `succeeded`, `partial` or `failed`. Failed candidates are never included in the formal manifest. The component does not run content revision loops, independent Solver attempts or difficulty calibration.

[Exploration](exploration-task-producer.en.md) discovers tasks from mixed raw materials; Derivation extracts capabilities from existing tasks and creates independent derivatives. Both share the same producer protocol and Object Store handoff.
