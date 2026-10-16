# HetaGen Task Producer Graph

`HetaGenTaskProducerGraph` retrieves papers from an existing Heta knowledge base, reuses their full text and evidence-backed graphs, and produces Harbor tasks through Omega's native Data Designer workflow. It also accepts prepared graph/text objects without a knowledge base.

## Install

Use Python 3.12 in the same environment as Heta:

```bash
pip install "heta-framework[generation]"
```

The extra installs Data Designer and Harbor. Knowledge-base mode also requires the dependencies for your KB procedure and storage backends.

Production requires Docker, Compose, Buildx and an OpenAI-compatible Chat Completions endpoint. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env`; exported environment variables take precedence. `api_base_url` overrides the endpoint.

## Knowledge Base Usage

Replace `your-model` with a model identifier provided by your endpoint, and use only inference parameters supported by that provider.

Given an existing Heta `KnowledgeBase` and destination `ObjectStoreProtocol`:

```python
import json
from uuid import uuid4

from heta_framework.generation import (
    HetaGenTaskProducerGraph,
    HetaGenTaskProducerGraphConfig,
    TaskRequest,
)

async def generate_tasks(kb, objects):
    producer = HetaGenTaskProducerGraph(
        HetaGenTaskProducerGraphConfig(
            query_mode="heta_graph_search",
            top_k=20,
            max_documents=2,
            model="your-model",
            reasoning_effort="low",
            max_concurrency=2,
            generation_guidance="Assess numerical implementation and hidden-case generalization",
        )
    )
    result = await producer.produce(
        TaskRequest(
            inputs={"query": "Numerical methods for heat transfer"},
            output_prefix=f"tasks/graph/{uuid4().hex}/",
        ),
        knowledge_base=kb,
        object_store=objects,
    )
    manifest = json.loads(await objects.get(result.tasks_manifest_key))
    report = json.loads(await objects.get(result.report_key))
    print(result.status, len(manifest["tasks"]))
    print(report["inputs_manifest_key"], report["native"]["stages"])
    return result
```

Call `await generate_tasks(kb, objects)` in your async application. This runs models and Docker. `query` selects materials; `generation_guidance` describes assessment goals. `max_documents` limits selected papers, not guaranteed task count. The destination may differ from the KB's source stores.

Inspect and freeze inputs separately with `await producer.prepare_inputs(request, knowledge_base=kb, object_store=objects)`. It returns `PreparedGraphInputs(manifest_key, document_count)` without calling task-generation models. The selected query engine may itself use models; preparation requests answer generation be disabled. Use a fresh output prefix for each preparation or production run.

## Prepared Object Usage

```python
from heta_framework.generation import ObjectInput

producer = HetaGenTaskProducerGraph(
    HetaGenTaskProducerGraphConfig(
        input_mode="objects",
        model="your-model",
        reasoning_effort="low",
        max_documents=2,
    )
)
result = await producer.produce(
    TaskRequest(
        inputs={
            "kg": ObjectInput(prefix="prepared/run-001/inputs/kg/"),
            "text": ObjectInput(prefix="prepared/run-001/inputs/text/"),
        },
        output_prefix="tasks/graph/run-001/",
    ),
    object_store=objects,
)
```

Explicit `ObjectInput(keys=(...,))` selections also work. JSON and UTF-8 text files pair by identical basename: `kg/paper-01.json` with `text/paper-01.txt`. Missing partners, duplicate basenames and selections exceeding `max_documents` are errors. No fuzzy title matching is used.

## Input Contract

KB mode resolves canonical query provenance (`document_id`, `document_ids`, or source object keys) to existing build artifacts:

| Config field | Default artifact | Typed content |
| --- | --- | --- |
| `parsed_documents_artifact` | `parsed_document_keys` | `ParsedDocument` |
| `entities_artifact` | `entity_keys` | `ExtractedEntity` |
| `relations_artifact` | `relation_keys` | `ExtractedRelation` |
| `chunks_artifact` | `chunk_keys` | `ParsedChunk` |

These match `ParseDocuments` and the default `HetaGraphProcedure` outputs. Override artifact names for other procedures. Source object stores are resolved from each producing step's requirements; initial artifacts use the KB default store. Ambiguous bindings are errors. The adapter scans these artifact object lists before filtering documents, so preparation cost can scale with the KB's artifact count rather than `max_documents`.

Default semantic correspondences are `TASK`, `METHOD`, `DATASET`, `METRIC`, `BASELINE`, `IMPLEMENTATION`, `THEOREM` and `DOMAIN_TERM` to their native Omega groups. Unknown types require an explicit mapping, for example `entity_type_mapping={"NumericalMethod": "Methods"}`; the adapter does not guess scientific categories.

Relations must resolve to entities in the same document. Evidence chunks must belong to that document and occur uniquely in the saved full text. Exact quoted lines and original IDs are retained; repeated mentions keep their individual descriptions and evidence. Conflicting semantic types sharing a native normalized name are rejected. Different documents remain separate.

Object mode requires native `A_Meta_Factual_Entities.Paper.title`, B entity groups and nonempty `E_Relations`. Every entity and relation needs a nonempty `evidence` list containing one-based inclusive `source_lines` and an exact `quote` from the paired text. Edges require `head`, `tail`, `relation`, with both endpoints present in B groups. The object input contract does not accept endpoints represented only as inferred C concepts.

Paired text/KG format example (field illustration only; production inputs must still meet screening thresholds):

`paper-01.txt`, including its trailing newline:

```text
Method A uses Method B.
```

`paper-01.json`:

```json
{
  "A_Meta_Factual_Entities": {
    "Paper": {
      "title": "Method comparison"
    }
  },
  "B_Textually_Mentioned_Entities": {
    "Methods": [
      {
        "name": "Method A",
        "evidence": [
          {
            "source_lines": [
              1,
              1
            ],
            "quote": "Method A uses Method B.\n"
          }
        ]
      },
      {
        "name": "Method B",
        "evidence": [
          {
            "source_lines": [
              1,
              1
            ],
            "quote": "Method A uses Method B.\n"
          }
        ]
      }
    ]
  },
  "E_Relations": [
    {
      "head": "Method A",
      "tail": "Method B",
      "relation": "USES",
      "evidence": [
        {
          "source_lines": [
            1,
            1
          ],
          "quote": "Method A uses Method B.\n"
        }
      ]
    }
  ]
}
```

Preparation checks integrity; native screening decides suitability. Defaults require a largest component with at least 20 nodes, 20 edges, 3 relation types, node fraction 0.35, and a B anchor.

## Configuration and Workflow

| Setting | Default | Purpose |
| --- | --- | --- |
| `input_mode` | `knowledge_base` | `knowledge_base` or `objects` |
| `query_mode` | `heta_graph_search` | Query mode supported by the existing KB indexes |
| `top_k` | 20 | Query result limit |
| `max_documents` | 3 | Maximum selected documents per call |
| `generation_guidance` | empty | Task-generation instructions |
| `model` | `bailian/glm-5.3` | Generation model identifier |
| `review_model` | `None` | Review model; defaults to `model` |
| `api_base_url` | `None` | Endpoint override; otherwise loaded from environment settings |
| `reasoning_effort` | `None` | Optional inference parameter supported by the provider |
| `max_tokens` | 32000 | Per-request output budget |
| `request_timeout_sec` | 600 | Per-request timeout in seconds |
| `max_concurrency` | 1 | Data Designer concurrency limit |
| `settings_file` | `None` | External production-policy JSON |
| `env_file` | `.env` | Model credentials file |
| `workspace` | `None` | Temporary workspace parent |

External native settings use `settings_file`, a JSON file validated by upstream `Settings`. For example:

```json
{
  "difficulty": {"tiers": ["medium", "complex"]},
  "repair": {"validation_attempts": 3},
  "validation": {"oracle_repeats": 2, "harbor_env": "docker"}
}
```

Omitted settings retain upstream defaults. `generation_guidance` is carried through candidate generation, specification, implementation and implementation review, separately from scientific evidence. `review_model` optionally selects a separate semantic review model; all other roles use `model`. Defaults are `max_tokens=32000` and `request_timeout_sec=600` per call. `max_concurrency` controls native Data Designer row groups, non-inference workers and model admission.

The eight stages remain `screen → review → candidates → candidate_review → spec → implementation → validate_static → validate_runtime`. Stages perform their configured internal revisions. Package directories are namespaced by document identity so equal generated slugs cannot overwrite another paper. Local workspaces are temporary; native checkpoint evidence is archived to Object Store.

## Results and Validation

`TaskResult` returns keys for `tasks.json` and `report.json`. Reports include the input manifest, native stage statuses/counts, rejection ledgers, step failures and `designer_evidence_key`. Task entries contain file paths, object keys, SHA-256 hashes, modes and validation evidence keys. Restore relative paths and executable permissions when downloading.

`report["native"]["complete"]` indicates stage execution completion; `records_complete` indicates whether completed stages retained all requested rows. `stage_record_losses` records the affected stage, requested and actual counts, and available document/tier identities. Missing rows make the run failed or partial, while independently validated tasks remain eligible for publication.

Publication requires a completed workflow, successful runtime rows, and an agreeing independent `validation_report.json`: static checks, data materialization, determinism and Harbor verification must pass. Oracle must reward 1 on every repeat (two by default); Nop must reward 0 without trial errors. An existing directory or a single ready flag is insufficient.

No hits and normal structural/semantic rejection can yield `succeeded` with zero tasks. Model-step and runtime failures yield `partial` if valid tasks were delivered, otherwise `failed`. Input errors raise `GraphInputError` before task-manifest publication. Validators do not inherit model credentials.

Snapshots record the KB build run ID, query response and hashes of bytes actually read. Staging rechecks saved hashes. This is not a cross-store transaction or a concurrent-writer lock: each run owns a fresh prefix.

## Usage Boundaries

Production checks cover format, deterministic data generation and reference/no-op behavior. They do not establish scientific correctness or task difficulty. Review instructions, data, grading and starter files before using tasks for evaluation or training.

[Task input and output example](hopf-task-example.md).
