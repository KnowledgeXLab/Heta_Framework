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
  cleanup.py
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

## Runtime State

Manifest 是最终 metadata，不承担构建过程中的强中断恢复。
强中断恢复由 `RecipeRunState` 负责。

当 recipe 配置了 `stores.objects` 时，`KnowledgeBase.create()` 会在 ObjectStore 中使用保留前缀：

```text
_heta/
  knowledge_bases/
    {knowledge_base_name}/
      manifest.json
      latest_run.json
      runs/
        {run_id}/
          state.json
          record.json
```

职责划分：

```text
latest_run.json
    指向最近一次 run。

state.json
    构建过程中的可更新状态。
    step started / succeeded / failed 都会写入。

record.json
    run 完成后的不可变记录。

manifest.json
    KB metadata、recipe manifest 和最终 run record。
```

`RecipeRunState` 记录：

```text
run_id
status
started_at / finished_at
current_step
step_records
artifacts
issues
```

`RecipeRunRecord` 仍然是最终不可变快照。
State 用于恢复，Record 用于报告、manifest 和 query capability。

## Cleanup

`cleanup.py` 定义 KB 生命周期里的删除协议。

核心类型：

```text
CleanupTarget
    一个可删除的持久化资源。

StepCleanupPlan
    单个 step 声明自己产生了哪些资源。

KnowledgeBaseDeletePlan
    KnowledgeBase 聚合所有 steps 后得到的完整删除计划。

KnowledgeBaseDeleteResult
    删除执行结果和非致命 issue。
```

第一版支持四类目标：

```text
object_key
    ObjectStore 中的单个派生产物。

runtime_prefix
    KnowledgeBase runtime metadata 前缀。

sql_table
    SQLStore 中由 step 创建的表。

vector_collection
    VectorStore 中由 step 创建的 collection。
```

边界原则：

```text
Step 声明 cleanup target。
KnowledgeBase 执行 cleanup。
ObjectStore raw/ 原始输入不属于 cleanup 范围。
```

这样新 step 在加入框架时必须同时说明它创建了什么、如何被清理。
删除逻辑仍然收束在 `KnowledgeBase.delete()`，不会散落在各个 step 内部。

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
`delete_plan()` 聚合 steps 的 cleanup plan。
`delete()` 删除 KB 派生产物、持久层索引和 runtime metadata，但不删除 raw 输入。

## Create Resume Semantics

`KnowledgeBase.create()` 是普通用户的统一入口。

当 ObjectStore 中已有同名 KB runtime metadata 时：

```text
latest run succeeded
    create() 拒绝再次构建，避免误覆盖已完成 KB。

latest run failed / running
    create() 加载 state.json，跳过已 succeeded 的 steps，继续未完成部分。

no latest run
    create() 创建新的 run_id 和 state.json。
```

这种设计避免引入额外的 `resume_existing()` API。
同名 KB 的失败恢复仍然通过同一个 `create()` 完成。

Step 级重复调用控制不完全依赖 Builder。
长成本 step 应该尽量让 ObjectStore artifact 天然幂等：

```text
ParseDocuments
    parsed/{document_id}.json 已存在时复用。

EmbedChunks
    embeddings/{chunk_id}.json 已存在时复用。

ExtractEntities
    entities/{chunk_id}/*.json 已存在时复用。

ExtractRelations
    relations/{chunk_id}/*.json 已存在时复用。
```

Builder 负责 run/step 级恢复，Step 负责 item/artifact 级复用。
