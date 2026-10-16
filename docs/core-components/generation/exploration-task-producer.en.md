# HetaGen Task Producer Exploration

`HetaGenTaskProducerExploration` explores verifiable problems in a mixed package of papers, code, data, or task descriptions and produces Harbor tasks. Inputs need not include a pre-extracted KG or workflow pattern. An author explores and writes candidates; a builder validates them and a reviewer independently evaluates them.

The implementation originates from Sci-SKT and implements `TaskProducerProtocol[HetaGenTaskProducerExplorationConfig]`. It retains bounded revisions. Web search, multiple source packages, and solver rollouts are outside this entry point.

## Install

Use Python 3.12 and install `heta-framework[generation]`. Data Designer 0.9.2 and Harbor 0.23.0 execute in the same Python environment as Heta. Production also needs Docker/Compose/Buildx, a compatible role image, and model credentials.

## Usage

Prepare a local `source-package/` directory containing `PACKAGE.md` and any papers, code, or data. The Markdown file describes sources and the intended tasks. Configure `OPENAI_API_KEY` and `SCI_PIPELINE_MODEL` in `.env`, and prepare Docker and a compatible role image before production.

```python
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from heta_framework.common.stores import LocalObjectStore
from heta_framework.generation import (
    HetaGenTaskProducerExploration,
    HetaGenTaskProducerExplorationConfig,
    ObjectInput,
    TaskRequest,
)

async def main():
    objects = LocalObjectStore("heta-workspace/objects")
    try:
        # 1. Upload source materials.
        source = Path("source-package")
        for file in sorted(source.rglob("*")):
            if file.is_file():
                key = f"sources/package-001/{file.relative_to(source).as_posix()}"
                await objects.put(key, file.read_bytes())

        # 2. Configure and run production with a fresh output prefix.
        producer = HetaGenTaskProducerExploration(
            HetaGenTaskProducerExplorationConfig(tasks_per_package=2)
        )
        result = await producer.produce(
            TaskRequest(
                inputs={"materials": ObjectInput(prefix="sources/package-001/")},
                output_prefix=f"tasks/{uuid4().hex}/",
            ),
            object_store=objects,
        )

        # 3. Read delivered tasks and the production report.
        tasks = json.loads(await objects.get(result.tasks_manifest_key))
        report = json.loads(await objects.get(result.report_key))
        print(result.status, len(tasks["tasks"]))
        print(result.tasks_manifest_key, result.report_key)
        print(report["native_results"])
    finally:
        await objects.aclose()

asyncio.run(main())
```

This defaults to real model calls. For a handoff-only test, use `HetaGenTaskProducerExplorationConfig(mode="mock", tasks_per_package=2)`; no API or Docker is needed, and no scientific tasks are delivered. Skip uploading if materials already exist in the store. The manifest, rather than the requested count, determines actual delivery; each task's `files` records downloadable object keys.

## Inputs And Configuration

Select exactly one `PACKAGE.md` and its package files. The destination must be unoccupied. Configuration is a frozen dataclass in `generation.exploration.config`, validated at construction and exposed through a read-only `config` property.

Fields cover task counts, concurrency, revision/role-attempt limits, textual generation/difficulty guidance, execution timeouts/resources, and credential/workspace paths. Production defaults to `harbor` mode. Provide `OPENAI_API_KEY`, `SCI_PIPELINE_MODEL`, and optionally `OPENAI_BASE_URL` through the credential file or environment.

## Results And Verification

Results reference a task manifest and a report. Publication requires native accepted/harbor/validated verdicts plus Harbor configuration and required-file checks. File records include hashes and permissions; consumers must restore executable script permissions. Verification and review evidence are archived separately.

Setting `mode` to `mock` tests actual NeMo scheduling and bounded revisions without API or Docker calls. Mock runs publish no scientific tasks and return a failed delivery status with simulation diagnostics.

This entry point does not expose cross-run resume. Temporary workspaces are prepared automatically; Harbor role workers use the current interpreter rather than an alternate environment.

## Role Image

`role_image` must name a locally installed image. Its default tag, `sci-pipeline/role:codex-0.154.0`, is not a public image download address. Installing the Python extra does not create this image.

The image must provide the `codex` command and a `node` user, and support role execution in `/workspace`. `codex --version` must match `codex_version`; the default expects `codex-cli 0.154.0`. Set both fields when using your own image.
