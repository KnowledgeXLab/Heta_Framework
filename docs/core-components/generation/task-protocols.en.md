# Task Production Protocols

A task producer declares its inputs, executes a complete production workflow, and publishes Harbor tasks and a production report to an Object Store.

```text
TaskRequirements   declares required inputs
TaskRequest        binds inputs and an output location
TaskProducer       executes the existing workflow
TaskResult         identifies the task manifest and report
```

`heta_framework.generation` provides the protocol, data types and three producers: [Exploration](exploration-task-producer.md), [Derivation](derivation-task-producer.md) and [Graph](graph-task-producer.md).

## Design Principles

Each existing workflow is integrated as a complete component. Its implementation owns model calls, NeMo configuration, role handoffs, and revision logic.

`ObjectStoreProtocol` is the only required shared component. Adapters read inputs, prepare local workspaces when needed, and publish outputs through the store. Graph additionally accepts `knowledge_base` in KB mode.

Input declarations describe delivery forms and native material requirements. Each producer validates its required file formats.

## Producer Contract

```python
from typing import Protocol, TypeVar, runtime_checkable

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.generation import TaskRequest, TaskRequirements, TaskResult

ConfigT_co = TypeVar("ConfigT_co", covariant=True)

@runtime_checkable
class TaskProducerProtocol(Protocol[ConfigT_co]):
    @property
    def name(self) -> str: ...

    @property
    def config(self) -> ConfigT_co: ...

    @property
    def requirements(self) -> TaskRequirements: ...

    async def produce(
        self,
        request: TaskRequest,
        *,
        object_store: ObjectStoreProtocol,
    ) -> TaskResult: ...
```

| Member | Responsibility |
| --- | --- |
| `name` | Stable producer identity for reports and execution records. |
| `config` | Read-only typed configuration owned by the concrete producer. |
| `requirements` | Input names, delivery types, and required flags for the configured mode. |
| `produce` | Validate the request, read materials, execute production, and publish outputs. |

`TaskProducerProtocol[ConfigT_co]` preserves each implementation's configuration type. Covariance supports static type checking; configuration classes own validation.

Requirements express the input contract; `TaskResult` expresses the output contract. All producers deliver Harbor task collections.

`@runtime_checkable` permits structural `isinstance` checks. It does not check method signatures, input contents, or task quality.

## Requirements

```python
from heta_framework.generation import InputRequirement, TaskRequirements

requirements = TaskRequirements(
    inputs=(
        InputRequirement(
            name="materials",
            description="A native source package containing PACKAGE.md and files.",
        ),
        InputRequirement(
            name="instruction",
            required=False,
            value_type="text",
        ),
    ),
)
```

| Field | Meaning |
| --- | --- |
| `name` | Unique slot name matching a request input key. |
| `description` | Native content, format, or layout requirements; defaults to an empty string. |
| `required` | Whether the input must be supplied; defaults to `True`. |
| `value_type` | `objects` or `text`; defaults to `objects`. |

Objects may contain papers, code, seed tasks, or graph files. The delivery type does not classify their scientific content. Configurations select the input mode before requirements are declared. Graph, for example, accepts query text in KB mode or paired files in object mode.

## Object Inputs

```python
from heta_framework.generation import ObjectInput

ObjectInput(keys=("inputs/paper.pdf", "inputs/data.csv"))
ObjectInput(prefix="inputs/source-package/")
```

Provide either nonempty, unique `keys` or a `prefix`, exclusively. Keys are relative POSIX object keys, not local absolute paths or S3 URLs. The supplied store determines the backend.

Users need not prepare input manifests. Producers resolve selections and record the actual objects consumed. Prefixes do not provide atomic snapshots or imply automatic paper parsing or native input preparation.

## Request And Configuration

```python
from heta_framework.generation import ObjectInput, TaskRequest

request = TaskRequest(
    inputs={
        "materials": ObjectInput(prefix="inputs/source-package/"),
        "instruction": "Generate verifiable scientific computing tasks.",
    },
    output_prefix="tasks/run-001/",
)
```

| Field | Meaning |
| --- | --- |
| `inputs` | Mapping of declared input names to `ObjectInput` or text. |
| `output_prefix` | Nonempty destination prefix for this run. |

Model selection, counts, concurrency, and revision budgets belong in the concrete producer's configuration. Requests bind only per-run inputs and the destination.

Raw materials must still satisfy the chosen workflow's native requirements. For example, Graph object mode needs paired text and KG files; a raw PDF is not a substitute for these inputs.

## Validation

```python
requirements.validate(request)
```

This rejects unknown names, missing required inputs, delivery-type mismatches, and blank supplied text. It performs no I/O. Producers check object existence, nonempty selections, and native formats during preparation.

Implementations must call validation even when invoked directly. The protocol does not wrap methods automatically.

## Result

```python
from heta_framework.generation import TaskResult

result = TaskResult(
    status="succeeded",
    tasks_manifest_key="tasks/run-001/tasks.json",
    report_key="tasks/run-001/report.json",
)

manifest_bytes = await objects.get(result.tasks_manifest_key)
report_bytes = await objects.get(result.report_key)
```

The example assumes `objects` is the execution store and both objects have already been published by the producer.

| Field | Meaning |
| --- | --- |
| `status` | `succeeded`, `partial`, or `failed`. |
| `tasks_manifest_key` | Object key for the Harbor task collection manifest. |
| `report_key` | Object key for production results, validation evidence, and failures. |

Consult the manifest for task counts. Normal screening can yield zero tasks; partial or failed runs may retain delivered tasks. Preserve native exit codes in the report rather than interpreting them as delivery counts.

Returned keys must be readable. Errors that prevent publication propagate to the caller. The current result type validates fields and keys; it does not verify existence or implement the full manifest/report JSON schema.

Record Harbor format checks separately from scientific validation. Associate tasks with fixed versions and evidence. Preserve review, Oracle, and Nop outcomes individually, including checks that were not run.

## Built-in Producers

| Producer | Inputs | Method |
| --- | --- | --- |
| [Exploration](exploration-task-producer.md) | `materials`: source package containing `PACKAGE.md` | Explore, author, build and review |
| [Derivation](derivation-task-producer.md) | `seed_tasks`: existing Harbor tasks | Extract capability patterns and derive tasks |
| [Graph](graph-task-producer.md) | `query` and `knowledge_base`, or `kg` and `text` objects | Generate from full text and graph evidence |

Each configuration class uses the corresponding producer name plus `Config` and is exposed through the read-only `config` property. Solving tasks and collecting trajectories are downstream operations outside this protocol.

## Implementation Example

This skeleton requires native preparation, execution, and publication to be filled in:

```python
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.generation import (
    InputRequirement,
    TaskRequest,
    TaskRequirements,
    TaskResult,
)

class NativeProducer:
    name = "native_task_producer"

    def __init__(self, config: object) -> None:
        self._config = config

    @property
    def config(self) -> object:
        return self._config

    @property
    def requirements(self) -> TaskRequirements:
        return TaskRequirements(inputs=(InputRequirement(name="materials"),))

    async def produce(
        self,
        request: TaskRequest,
        *,
        object_store: ObjectStoreProtocol,
    ) -> TaskResult:
        self.requirements.validate(request)
        # Stage inputs, execute the workflow, and publish tasks and evidence.
        raise NotImplementedError("Implement native execution and publication")
```

See [Extending Task Production](../../development/task-production-contract.md) for publication and validation requirements.
