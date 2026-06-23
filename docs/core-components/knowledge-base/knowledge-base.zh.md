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

`KnowledgeBase.resume()` 用当前 `run_record` 继续构建：

```python
resumed = await restored.resume()
```

默认行为是：

- 使用上一轮 `run_record.artifacts` 作为初始 artifacts。
- 跳过之前已经 `succeeded` 的 steps。
- 生成新的 run record。
- 更新 `updated_at`。

## Query APIs

当前 `KnowledgeBase` 主要负责构建结果和生命周期 metadata。
具体查询能力由 steps 解锁，例如：

```text
IndexVectors -> vector_search
BuildGraph / MergeGraphIntoStore -> heta_graph_search
```

后续面向用户的 query API 可以挂在 `KnowledgeBase` 上，但底层能力仍然来自 recipe 中实际执行过的 steps。
