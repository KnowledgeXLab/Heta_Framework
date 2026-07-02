# KnowledgeBase

`KnowledgeBase` 是用户最终拿到的知识库对象。它保存构建 recipe、最近一次 run record、知识库 metadata，并提供 query、load、resume 和 delete 等生命周期能力。

```python
from heta_framework.kb import KnowledgeBase

kb = await KnowledgeBase.create(
    recipe=recipe,
    name="papers",
    description="论文知识库",
)
```

`KnowledgeBase.create()` 会使用 `KnowledgeBaseBuilder` 执行 recipe。构建完成后，`KnowledgeBase` 会记录本次构建状态和已解锁的 query capabilities。

## Fields

`KnowledgeBase` 持有：

```text
name
description
recipe
run_record
created_at
updated_at
metadata
```

其中：

| 字段 | 说明 |
| --- | --- |
| `recipe` | 当前 KB 的构建说明和 runtime components。 |
| `run_record` | 最近一次构建记录，包含 artifacts 和 capabilities。 |
| `metadata` | 用户可写的轻量 metadata。 |

## Create

创建知识库：

```python
kb = await KnowledgeBase.create(
    recipe=recipe,
    name="quickstart",
    description="A first Heta knowledge base.",
)
```

如果需要传入初始 artifacts：

```python
kb = await KnowledgeBase.create(
    recipe=recipe,
    name="papers",
    initial_artifacts={"source_keys": ["raw/paper.pdf"]},
)
```

是否需要 `initial_artifacts` 由 recipe 中的 steps 决定。例如 `ParseDocuments` 默认从 `ObjectStore` 的 `raw/` 前缀读取对象，不需要额外传入 source list。

## Runtime Metadata

如果 recipe 配置了 `stores.objects`，`KnowledgeBase.create()` 会把运行时 metadata 写入 ObjectStore 的保留前缀：

```text
_heta/knowledge_bases/{knowledge_base_name}/
```

目录结构如下：

```text
_heta/
  knowledge_bases/
    papers/
      manifest.json
      latest_run.json
      runs/
        run_xxx/
          state.json
          record.json
```

这些文件属于框架 runtime metadata，不属于用户原始数据：

| 文件 | 作用 |
| --- | --- |
| `manifest.json` | 保存 KB metadata、recipe manifest 和最终 run record。 |
| `latest_run.json` | 指向最近一次 run。 |
| `state.json` | 构建过程中的可更新状态。 |
| `record.json` | 一次 run 完成后的不可变记录。 |

它们用于审计、调试、load 和断点续跑。

## Existing Runs

当 `stores.objects` 可用时，同名 `KnowledgeBase.create()` 会检查：

```text
_heta/knowledge_bases/{knowledge_base_name}/latest_run.json
```

行为如下：

| 最近一次 run | create 行为 |
| --- | --- |
| `succeeded` | 失败并拒绝覆盖，避免误改一个已经完成的 KB。 |
| `failed` / `running` | 加载上次 state，跳过已成功 steps，从未完成位置继续。 |
| 不存在 | 创建新的 run。 |

因此失败恢复仍然使用同一个入口：

```python
kb = await KnowledgeBase.create(
    recipe=recipe,
    name="papers",
)
```

用户不需要单独调用 `resume_existing()`。

## Load

`KnowledgeBase.load()` 用于重新打开一个已经成功构建完成的知识库：

```python
kb = await KnowledgeBase.load(
    recipe=recipe,
    name="papers",
)
```

它会从当前 recipe 的 `stores.objects` 中读取：

```text
_heta/knowledge_bases/{knowledge_base_name}/manifest.json
```

`load()` 不重新执行 steps，也不重新构建知识库。它只恢复已经存在的 KB metadata、最近一次成功 run 的 artifacts 和 query capabilities。

进程退出后的推荐流程是：

```python
recipe = build_runtime_recipe_again(...)
kb = await KnowledgeBase.load(recipe=recipe, name="papers")
response = await kb.query("...", mode="vector_search")
```

注意：`load()` 使用当前传入的 runtime recipe 提供模型、向量库、SQL、对象存储等运行时组件。因此你需要传入能够连接到原持久化后端的 recipe。

如果 KB 不存在，`load()` 会抛出 `KnowledgeBaseNotFoundError`。如果最近一次 run 还没有成功完成，`load()` 会抛出 `KnowledgeBaseNotReadyError`。失败恢复仍然应该继续使用同名 `KnowledgeBase.create()`。

## Delete

`KnowledgeBase.delete()` 删除这个知识库构建过程中产生的派生产物：

```python
result = await kb.delete()
```

删除范围由 recipe 中每个 step 的 `cleanup_plan()` 声明，`KnowledgeBase` 统一执行。默认删除：

```text
ObjectStore 中的 parsed/chunks/embeddings/entities/relations 等派生产物
SQLStore 中由 steps 创建的表
VectorStore 中由 steps 创建的 collection
_heta/knowledge_bases/{knowledge_base_name}/ 下的 runtime metadata
```

`delete()` 不删除 `raw/` 下的用户原始文件。原始文件通常由用户上传、同步或外部系统管理，不属于 KB 派生产物。

可以先查看删除计划：

```python
plan = kb.delete_plan()
print(plan.object_keys)
print(plan.sql_tables)
print(plan.vector_collections)
```

也可以 dry run：

```python
result = await kb.delete(dry_run=True)
```

删除过程中某个目标失败不会中断其他目标删除。失败项会记录在 `result.issues` 中，调用方可以据此重试或人工处理。

## Manifest

`KnowledgeBase.manifest()` 导出可持久化 metadata：

```python
manifest = kb.manifest()
```

`KnowledgeBaseManifest` 记录：

```text
name
description
created_at
updated_at
recipe
run_record
metadata
```

Manifest 适合用于审计、展示、恢复 KB metadata 和断点基础。它不会保存模型 client、数据库连接、Milvus client 等 runtime objects。

## Restore

`KnowledgeBase.restore()` 适合已经手动拿到 `KnowledgeBaseManifest` 的高级场景：

```python
restored = KnowledgeBase.restore(
    manifest=manifest,
    recipe=recipe,
)
```

恢复时必须重新提供 runtime recipe。这样可以避免把 API key、数据库连接、HTTP client 等运行时对象写入 manifest。

普通用户更推荐使用 `KnowledgeBase.load()`，让框架自己从 ObjectStore 读取 manifest。

## Resume

`KnowledgeBase.resume()` 用当前内存对象的 `run_record` 继续构建：

```python
resumed = await restored.resume()
```

默认行为是：

- 使用上一轮 `run_record.artifacts` 作为初始 artifacts。
- 跳过之前已经 `succeeded` 的 steps。
- 生成新的 run record。
- 更新 `updated_at`。

对于普通用户，更推荐使用同名 `KnowledgeBase.create()`。当 ObjectStore 中存在失败 run 时，`create()` 会自动加载 runtime state 并续跑。`resume()` 更适合已经手动持有或恢复了一个 `KnowledgeBase` 对象的高级场景。

## Query APIs

`KnowledgeBase.query()` 负责调用当前 KB 已解锁的 query engine：

```python
response = await kb.query(
    "What does this knowledge base contain?",
    mode="vector_search",
    top_k=5,
)
```

具体查询能力由 steps 解锁，例如：

```text
IndexVectors -> vector_search
IndexFullText -> full_text_search
BuildGraph / MergeGraphIntoStore -> heta_graph_search
```

底层 query engine 会使用当前 runtime recipe 中的模型和存储组件。因此 `load()` 后也需要提供能够连接到原持久化后端的 recipe。
