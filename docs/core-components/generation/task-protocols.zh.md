# 任务生产协议

Task producer 是 Heta 的任务生产组件。每个 producer 声明需要的输入，执行一条完整的生产管线，将符合约定 Harbor 格式的任务和生产报告写入 Object Store。

协议解决三个问题：

- 这个 producer 需要哪些输入
- 本次输入在哪里，产物写到哪里
- 执行后得到哪些任务和报告

```text
TaskRequirements   声明需要什么
TaskRequest        绑定本次输入和输出位置
TaskProducer       执行现有生产管线
TaskResult         返回任务清单和报告的位置
```

`heta_framework.generation` 提供统一协议，以及 [Exploration](exploration-task-producer.md)、[Derivation](derivation-task-producer.md) 和 [Graph](graph-task-producer.md) 三种任务生产组件。

## 组件边界

每条现有生产管线作为完整组件接入。模型调用、NeMo 配置、角色交接和内部修订由具体实现管理。

公共协议只要求 `ObjectStoreProtocol`。输入文件通过 store 读取，任务与报告通过 store 保存。已有管线需要本地目录时，由适配器准备工作区并回收产物。

输入声明描述所需材料和交付形式。具体文件格式在 `description` 中说明，并由实现检查。

Graph 在知识库模式下还接受 `knowledge_base` 参数，详见其[使用说明](graph-task-producer.md)。

## 组件协议

每个 producer 实现 `TaskProducerProtocol`：

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

| 成员 | 说明 |
| --- | --- |
| `name` | producer 的稳定标识，用于报告和运行记录。 |
| `config` | 当前 producer 的只读、类型化配置；具体类型由实现定义。 |
| `requirements` | 当前配置下需要的输入名称、交付形式和必填条件。 |
| `produce` | 校验请求、读取资料、运行管线并保存任务和报告。 |

`TaskProducerProtocol[ConfigT_co]` 保留具体配置类型，例如 `TaskProducerProtocol[HetaGenTaskProducerExplorationConfig]`。协变类型参数用于类型检查，不加载或校验配置；配置类负责参数校验。

`requirements` 是运行前的输入合同，`TaskResult` 是运行后的交付合同。所有 producer 的交付目标均为 Harbor 任务集合。

`@runtime_checkable` 允许用 `isinstance(producer, TaskProducerProtocol)` 检查结构是否具备协议成员；它不验证方法签名、输入内容或任务质量。

## 输入声明

`TaskRequirements` 包含一组 `InputRequirement`。以下为自定义组件的声明示例；内置组件的输入以各自的 `requirements` 为准：

```python
from heta_framework.generation import InputRequirement, TaskRequirements

requirements = TaskRequirements(
    inputs=(
        InputRequirement(
            name="materials",
            description="原生资料包目录，包含 PACKAGE.md 和相关文件。",
        ),
        InputRequirement(
            name="instruction",
            description="本次任务生成要求。",
            required=False,
            value_type="text",
        ),
    ),
)
```

| 字段 | 说明 |
| --- | --- |
| `name` | 输入槽位名称，与 `TaskRequest.inputs` 的 key 对应；同一声明中不能重复。 |
| `description` | 说明材料要求、原生格式或目录布局，默认空字符串。 |
| `required` | 是否必须提供，默认 `True`。 |
| `value_type` | `objects` 表示 Object Store 文件选择，`text` 表示文本，默认 `objects`。 |

`objects` 与 `text` 只描述交付形式。论文、代码、种子任务或图谱文件都可以通过 `objects` 交付。

如果一个实现支持多个入口，由它的配置选择模式，再返回对应的 `requirements`。例如 Graph 的知识库模式要求查询文本，对象模式要求正文与 KG 文件。

## 对象输入

`ObjectInput` 从本次传入的 Object Store 选择文件：

```python
from heta_framework.generation import ObjectInput

ObjectInput(keys=("inputs/paper.pdf", "inputs/data.csv"))
ObjectInput(prefix="inputs/source-package/")
```

| 字段 | 说明 |
| --- | --- |
| `keys` | 明确的对象 key 集合，不能重复。 |
| `prefix` | 按前缀选择文件；不设置时为 `None`。 |

非空 `keys` 与 `prefix` 必须且只能提供一个。key 使用相对 POSIX 路径，不填写本地绝对路径或 S3 URL。数据具体位于本地还是 S3，由传入的 Object Store 决定。

用户无需预先编写输入 manifest。producer 在准备输入时解析选择、检查文件并记录本次实际使用的对象。prefix 选择不是原子快照，也不表示自动完成论文解析、知识抽取或原生输入准备。

## 请求与配置

`TaskRequest` 绑定本次输入和产物位置：

```python
from heta_framework.generation import ObjectInput, TaskRequest

request = TaskRequest(
    inputs={
        "materials": ObjectInput(prefix="inputs/source-package/"),
        "instruction": "根据这些资料生成可验证的科学计算任务。",
    },
    output_prefix="tasks/run-001/",
)
```

| 字段 | 说明 |
| --- | --- |
| `inputs` | 输入名称到 `ObjectInput` 或文本的映射，名称必须与声明匹配。 |
| `output_prefix` | 本次任务、报告和清单的保存前缀，不能为空。 |

任务数、模型、并发、修订预算等保留在具体 producer 的配置中。`TaskRequest` 只表达本次输入与输出位置。

输入材料须满足所选组件的要求。例如，Graph 的对象模式需要配对的正文和 KG，原始 PDF 不能直接替代这两项输入。

## 输入校验

producer 在调用原生管线之前执行：

```python
requirements.validate(request)
```

校验拒绝以下输入：

- 未声明的输入名称
- 缺失的必填输入
- 对象与文本类型不匹配
- 已提供但内容为空白的文本

这一步不访问 Object Store。文件是否存在、prefix 是否选中实际文件，以及资料是否符合原生格式，由 producer 在准备阶段检查。

直接调用 `produce()` 时也应执行校验；协议本身不会自动包装实现。

## 返回结果

`TaskResult` 返回已保存产物的对象 key：

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

以上示例假设 `objects` 是执行时使用的 Object Store，两个对象已由 producer 保存。

| 字段 | 说明 |
| --- | --- |
| `status` | `succeeded`、`partial` 或 `failed`，描述本次生产执行。 |
| `tasks_manifest_key` | Harbor 任务集合清单的对象 key。 |
| `report_key` | 生产、验证和失败原因报告的对象 key。 |

任务数量以清单为准。正常筛选结束可以产生零任务；部分完成或失败的运行也可能已经交付部分任务。原生退出码应记录在报告中，不单独用它推断实际交付数量。

正常返回时清单和报告应已可读取。存储不可写等导致产物无法发布的错误向调用者传播。`TaskResult` 校验字段与对象 key；任务清单和报告的内容由具体组件定义。

Harbor 格式检查与科学验证分别记录。任务清单应关联固定任务版本及验证证据；审核、Oracle、Nop 等实际执行的检查分别保留结果，未运行的检查明确标记。格式兼容不等于已经完成运行验证。

## 内置组件

| 组件 | 输入 | 生产方式 |
| --- | --- | --- |
| [Exploration](exploration-task-producer.md) | `materials`：包含 `PACKAGE.md` 的资料包 | 探索选题，构建任务并独立审核 |
| [Derivation](derivation-task-producer.md) | `seed_tasks`：已有 Harbor 任务 | 提炼能力模式，派生新任务 |
| [Graph](graph-task-producer.md) | `query` 与 `knowledge_base`，或 `kg`、`text` 对象 | 从正文和知识图谱证据生成任务 |

每个组件的配置类型为对应类名加 `Config`，并通过只读 `config` 属性获取。求解任务和采集轨迹属于下游执行，不在任务生产协议范围内。

## 自定义组件

自定义实现可以按下面的结构接入已有管线。`NativeProducer` 是示例名称；准备、执行和发布部分需要实现者填写。

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
        return TaskRequirements(
            inputs=(InputRequirement(name="materials"),),
        )

    async def produce(
        self,
        request: TaskRequest,
        *,
        object_store: ObjectStoreProtocol,
    ) -> TaskResult:
        self.requirements.validate(request)
        # 读取实际输入，准备本地工作区。
        # 调用原生管线，收集任务与验证证据。
        # 上传任务和报告，最后发布任务清单。
        raise NotImplementedError("Implement native execution and publication")
```

实现者使用统一协议交接输入和输出，原有管线仍然负责具体的任务生产策略。

扩展组件的发布与验证约定见[扩展任务生产组件](../../development/task-production-contract.md)。
