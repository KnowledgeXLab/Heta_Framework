# Knowledge Base Core Design

本页是 Heta Framework 顶层知识库核心层的开发设计备忘。
它记录 `KnowledgeRecipe`、`KnowledgeBaseBuilder`、`KnowledgeBase`、run record 和 manifest 的边界。

## Principle

```text
Recipe describes.
Builder builds.
Step executes.
Record remembers.
KnowledgeBase owns.
Manifest persists.
```

另一条边界：

```text
Recipe validates static logic.
Builder validates runtime reality.
```

## Modules

```text
kb/
  components.py
  validation.py
  state.py
  manifests.py
  recipe.py
  builder.py
  knowledge_base.py
```

## Components

`KnowledgeModels`、`KnowledgeStores`、`KnowledgeParsers` 保存 runtime components，并根据 `ComponentRef` 做 lookup。

它们不做连接检查，不序列化 runtime 对象。

组件 key：

```text
models.language
models.embedding
models.language.strong
stores.objects
stores.vector
stores.sql
parsers.documents
```

## Validation

`KnowledgeRecipe.validate()` 做静态逻辑检查：

```text
procedure 能展开
component refs 在 recipe 中存在
artifact requirements 被 initial_artifacts 或前序 steps 提供
query requirements 被 initial_queries 或前序 steps 提供
重复 artifact 输出给 warning
```

它不检查：

```text
ObjectStore key 是否存在
数据库是否连接
模型 API 是否可用
LLM 输出是否正确
```

检查方式是 ordered dataflow validation。Recipe 不拓扑排序，不自动重排 steps。

## State

`StepRunRecord` 记录单个 step：

```text
index
step_name
step_type
status
started_at / finished_at
requirements
capabilities
input_artifacts
output_artifacts
issues
error
```

`RecipeRunRecord` 记录整次构建：

```text
run_id
status
started_at / finished_at
step_records
artifacts
capabilities
issues
```

它是 resume、manifest 和 build report 的基础。

## Manifest

`StepManifest`、`KnowledgeRecipeManifest`、`KnowledgeBaseManifest` 是可持久化 metadata。

Manifest 用于审计、展示、恢复 KB metadata 和断点基础。
Manifest 不序列化 runtime model/store/parser，不试图自动恢复 client 连接。

## KnowledgeRecipe

`KnowledgeRecipe` 是静态构建方案。

字段：

```text
models
stores
parsers
steps
metadata
```

方法：

```text
expanded_steps()
get_component(ref)
has_component(ref)
validate(...)
require_valid(...)
manifest()
```

它不执行、不记录进度、不访问外部资源。

## KnowledgeBaseBuilder

`KnowledgeBaseBuilder` 按 recipe 构建 knowledge base。

职责：

```text
调用 recipe.validate()
创建 StepExecutionContext
按顺序执行 steps
运行前后 diff artifacts
记录 StepRunRecord
收集 issues
生成 RecipeRunResult
```

支持：

```text
previous_record
skip_succeeded_steps
```

断点逻辑：

```text
previous_record.artifacts 作为初始 artifacts
skip_succeeded_steps=True 时跳过 succeeded step
failed / pending 后续 step 继续执行
```

## KnowledgeBase

`KnowledgeBase` 是用户入口和构建结果对象。

字段：

```text
name
description
recipe
run_record
created_at
updated_at
metadata
```

方法：

```text
create(...)
restore(...)
resume(...)
manifest()
```

`create()` 调用 `KnowledgeBaseBuilder.build()`。
`restore()` 由 manifest + runtime recipe 恢复 KB metadata。
`resume()` 用 previous record 继续构建，并返回新的 immutable `KnowledgeBase`。
