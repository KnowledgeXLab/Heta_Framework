# KnowledgeBaseBuilder

`KnowledgeBaseBuilder` 是 recipe 的执行器。它接收一个 `KnowledgeRecipe`，展开其中的 procedures，按顺序执行 steps，并返回一次构建的运行结果。

```python
from heta_framework.kb import KnowledgeBaseBuilder

result = await KnowledgeBaseBuilder().build(recipe)
```

Builder 不定义知识库结构。知识库结构来自 `KnowledgeRecipe`，具体业务动作来自每个 step。

## Responsibilities

Builder 负责把静态 recipe 变成一次真实 build run：

| 能力 | 说明 |
| --- | --- |
| Validate recipe | 运行前调用 `recipe.validate()`。 |
| Create context | 为 steps 创建 `StepExecutionContext`。 |
| Run steps | 按 recipe 中声明的顺序执行 steps。 |
| Pass artifacts | 在 steps 之间传递 artifact map。 |
| Collect capabilities | 汇总本次构建解锁的 query modes 和 search assets。 |
| Collect issues | 汇总 steps 上报的 non-fatal issues。 |
| Record execution | 记录 run record 和每个 step 的状态。 |

运行时检查由 Builder 和 Step 共同完成。例如 object key 是否存在、模型调用是否成功、数据库是否可用，都属于运行时问题，不属于 recipe 静态校验。

## Build Result

`build()` 返回 `RecipeRunResult`：

```python
result = await KnowledgeBaseBuilder().build(recipe)

print(result.record.status)
print(result.capabilities.queries)
print(result.artifacts.keys())
```

结果包含：

| 字段 | 说明 |
| --- | --- |
| `record` | 本次构建的运行记录。 |
| `artifacts` | 构建结束后的 artifact map。 |
| `capabilities` | 本次构建解锁的能力。 |
| `issues` | steps 上报的非致命问题。 |

## Run Records

`RecipeRunRecord` 是一次构建结束后的记录。

它包含：

```text
run_id
status
started_at / finished_at
step_records
artifacts
capabilities
issues
```

每个 `StepRunRecord` 记录一个 step：

```text
index
step_name
step_type
status
started_at / finished_at
input_artifacts
output_artifacts
issues
error
```

这些记录用于 build report、debug、query capability 恢复和断点续跑。

## Run State

`RecipeRunState` 是构建过程中的可更新状态。它和 `RecipeRunRecord` 的关系是：

| 类型 | 用途 |
| --- | --- |
| `RecipeRunState` | 构建过程中持续更新，可落盘到 ObjectStore。 |
| `RecipeRunRecord` | 构建结束后的不可变快照。 |

当 `KnowledgeBase.create()` 为 Builder 传入 `run_state` 时，Builder 会在这些时机更新 state：

```text
step started
step succeeded
step failed
run finished
```

因此即使 Python 进程被 kill、机器重启或外部 API 中断，ObjectStore 中仍然可以保留最近一次 run 的状态。

## Resume

Builder 可以跳过之前已经成功的 steps：

```python
from heta_framework.kb import KnowledgeBaseBuilder, KnowledgeBaseBuilderConfig

builder = KnowledgeBaseBuilder(
    KnowledgeBaseBuilderConfig(skip_succeeded_steps=True)
)

result = await builder.build(
    recipe,
    previous_record=previous_record,
)
```

恢复时，Builder 会复用 `previous_record.artifacts`，并跳过匹配且状态为 `succeeded` 的 step。

在 `KnowledgeBase.create()` 中，这个过程会自动发生：

```text
1. 从 _heta/knowledge_bases/{name}/latest_run.json 找到最近 run。
2. 加载 runs/{run_id}/state.json。
3. 转换为 previous_record。
4. 配置 skip_succeeded_steps=True。
5. 从第一个未成功 step 继续执行。
```

Step 是否能避免重复外部调用，还取决于 step 自身的 artifact 幂等设计。例如 `EmbedChunks` 会复用已存在的 embedding artifact，`ExtractEntities` 和 `ExtractRelations` 会复用已存在的 chunk 级抽取结果。

## Failure Behavior

默认情况下，某个 step 抛出异常后，Builder 会停止后续执行：

```python
builder = KnowledgeBaseBuilder()
```

如果需要记录失败但继续执行，可以配置：

```python
builder = KnowledgeBaseBuilder(
    KnowledgeBaseBuilderConfig(stop_on_error=False)
)
```

继续执行不代表后续 steps 一定能成功。如果必需 artifact 没有产生，后续 step 仍然可能因为 requirements 不满足而失败。
