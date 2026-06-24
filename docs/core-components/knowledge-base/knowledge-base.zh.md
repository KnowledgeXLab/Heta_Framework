# KnowledgeBase

`KnowledgeBase` 是用户最终拿到的知识库对象。
它持有构建 recipe、最近一次 run record，以及知识库自身的 metadata。

```python
from heta_framework.kb import KnowledgeBase

kb = await KnowledgeBase.create(
    recipe=recipe,
    name="papers",
    description="论文知识库",
)
```

`KnowledgeBase.create()` 会使用 `KnowledgeBaseBuilder` 执行 recipe。
构建完成后，`KnowledgeBase` 记录这次构建的状态和能力。

如果 recipe 配置了 `stores.objects`，`KnowledgeBase.create()` 还会把运行时 metadata
写入 ObjectStore 的保留前缀：

```text
_heta/knowledge_bases/{knowledge_base_name}/
```

这让同一个知识库在进程中断后可以通过同名 `create()` 继续构建。

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

- `recipe` 是构建说明。
- `run_record` 是最近一次构建记录。
- `metadata` 是用户可写的轻量 metadata。

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

是否需要 `initial_artifacts` 由 recipe 中的 steps 决定。
例如 `ParseDocuments` 默认从 `ObjectStore` 的 `raw/` 前缀读取对象，不需要额外传入 source list。

### Existing Runs

当 `stores.objects` 可用时，`KnowledgeBase.create()` 会检查这个 KB 是否已有运行记录：

```text
_heta/knowledge_bases/{knowledge_base_name}/latest_run.json
```

行为如下：

- 如果最近一次 run 已经 `succeeded`，同名 `create()` 会失败，避免误覆盖一个已经完成的 KB。
- 如果最近一次 run 是 `failed` 或 `running`，同名 `create()` 会加载上次 state，并跳过已经成功的 steps。
- 如果没有运行记录，会创建新的 run。

这意味着失败恢复的推荐方式仍然是同一个入口：

```python
kb = await KnowledgeBase.create(
    recipe=recipe,
    name="papers",
)
```

用户不需要单独调用 `resume_existing()`。

## Runtime Metadata

运行时 metadata 默认写在当前 KB 的 ObjectStore 中：

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

其中：

- `latest_run.json` 指向最近一次 run。
- `state.json` 是构建过程中的可更新状态，step 开始、成功、失败时都会更新。
- `record.json` 是一次 run 完成后的不可变记录。
- `manifest.json` 保存 KB metadata、recipe manifest 和最终 run record。

这些文件属于框架 runtime metadata，不属于用户原始数据。
它们用于审计、调试和断点续跑。

## Delete

`KnowledgeBase.delete()` 删除这个知识库构建过程中产生的派生产物：

```python
result = await kb.delete()
```

删除范围由 recipe 中每个 step 的 `cleanup_plan()` 声明，`KnowledgeBase` 统一执行。
默认删除：

```text
ObjectStore 中的 parsed/chunks/embeddings/entities/relations 等派生产物
SQLStore 中由 steps 创建的表
VectorStore 中由 steps 创建的 collection
_heta/knowledge_bases/{knowledge_base_name}/ 下的 runtime metadata
```

`delete()` 不删除 `raw/` 下的用户原始文件。
原始文件通常由用户上传、同步或外部系统管理，不属于 KB 派生产物。

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

删除过程中某个目标失败不会中断其他目标删除。
失败项会记录在 `result.issues` 中，调用方可以据此重试或人工处理。

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

Manifest 适合用于审计、展示、恢复 KB metadata 和断点基础。
它不会保存模型 client、数据库连接、Milvus client 等 runtime 对象。

## Restore

恢复时必须重新提供 runtime recipe：

```python
restored = KnowledgeBase.restore(
    manifest=manifest,
    recipe=recipe,
)
```

这样做可以避免把 API key、数据库连接、HTTP client 等运行时对象写入 manifest。

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

对于普通用户，更推荐使用同名 `KnowledgeBase.create()`。
当 ObjectStore 中存在失败 run 时，`create()` 会自动加载 runtime state 并续跑。
`resume()` 更适合已经手动持有或恢复了一个 `KnowledgeBase` 对象的高级场景。

## Query APIs

当前 `KnowledgeBase` 主要负责构建结果和生命周期 metadata。
具体查询能力由 steps 解锁，例如：

```text
IndexVectors -> vector_search
BuildGraph / MergeGraphIntoStore -> heta_graph_search
```

后续面向用户的 query API 可以挂在 `KnowledgeBase` 上，但底层能力仍然来自 recipe 中实际执行过的 steps。
