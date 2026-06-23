# KnowledgeBaseBuilder

`KnowledgeBaseBuilder` 是 recipe 的执行器。
它接收一个 `KnowledgeRecipe`，按顺序执行展开后的 steps，并返回一次构建的运行结果。

```python
from heta_framework.kb import KnowledgeBaseBuilder

result = await KnowledgeBaseBuilder().build(recipe)
```

Builder 不定义知识库结构。
结构来自 `KnowledgeRecipe`，真正的业务动作来自每个 step。

## Responsibilities

`KnowledgeBaseBuilder` 负责：

- 调用 `recipe.validate()`。
- 创建 `StepExecutionContext`。
- 按顺序执行 steps。
- 在 steps 之间传递 artifacts。
- 收集 capabilities。
- 收集 non-fatal issues。
- 记录每个 step 的运行状态。
- 返回 `RecipeRunResult`。

运行时检查由 Builder 和 Step 共同完成，例如 object key 是否存在、模型调用是否成功、数据库是否可用。

## Build Result

`build()` 返回 `RecipeRunResult`：

```python
result = await KnowledgeBaseBuilder().build(recipe)

print(result.record.status)
print(result.capabilities.queries)
print(result.artifacts.keys())
```

结果包含：

- `record`：本次构建的运行记录。
- `artifacts`：最终 artifact map。
- `capabilities`：本次构建解锁的能力。
- `issues`：steps 上报的非致命问题。

## Run Records

`RecipeRunRecord` 是一次构建的运行记录。

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

`StepRunRecord` 记录单个 step：

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

这些记录是 build report、debug 和断点恢复的基础。

## Resume

Builder 支持跳过之前已经成功的 steps：

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

是否继续执行取决于后续 steps 的 artifact requirements。
如果必需 artifact 没有产生，后续 step 仍然可能失败。
