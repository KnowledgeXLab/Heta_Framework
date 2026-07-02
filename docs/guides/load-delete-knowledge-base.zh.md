# Load And Delete A KnowledgeBase

Heta 的 KnowledgeBase 生命周期分成三件事：

```text
create  -> build or resume
load    -> reopen a completed KB
delete  -> remove derived resources
```

这让框架可以覆盖本地脚本、Web 后台和离线评测三种常见使用方式。

## Create

`KnowledgeBase.create()` 用 recipe 构建知识库。

```python
kb = await KnowledgeBase.create(recipe=recipe, name="papers")
```

当 recipe 配置了 `ObjectStore` 时，Heta 会在保留前缀下写入运行 metadata：

```text
_heta/knowledge_bases/{name}/
  manifest.json
  latest_run.json
  runs/
    {run_id}/
      state.json
      record.json
```

如果进程中断，同名 `create()` 可以读取已有 state，并从未完成的 step 继续。

## Load

`KnowledgeBase.load()` 用于重新打开已经成功构建完成的 KB：

```python
kb = await KnowledgeBase.load(recipe=recipe, name="papers")
```

`load()` 不重新执行 steps，也不重新写入索引。它只恢复 metadata、run record 和 query capabilities，然后继续使用 recipe 中配置的 runtime components。

适合：

- Web 服务重启后重新挂载已有 KB。
- 评测完成后重新查询结果。
- 离线构建完成后在另一个进程中使用。

如果 KB 还没有成功构建，应该继续用同名 `create()` 恢复构建，而不是 `load()`。

## Delete

`KnowledgeBase.delete()` 删除派生产物，但保留用户原始输入：

```python
result = await kb.delete()
```

默认不删除 `raw/` 下的原始文件。它会根据各 step 的 cleanup plan 删除：

- parsed documents。
- chunks。
- embeddings。
- extracted entities and relations。
- SQL tables。
- vector collections。
- text indexes。
- `_heta/knowledge_bases/{name}/` runtime metadata。

如果只想看会删除什么，可以先 dry run：

```python
plan = kb.delete_plan()
dry_run = await kb.delete(dry_run=True)
```

## Naming

KB name 会用于运行 metadata 的路径。建议使用稳定、可读、唯一的名称：

```text
papers
faa_handbook
marine_biology_vector_v1
```

不要把临时随机字符串作为生产 KB 名称。稳定名称更适合失败恢复、load 和 delete。

## Next

- 想看完整 API 细节，看 [KnowledgeBase](../core-components/knowledge-base/knowledge-base.zh.md)。
- 想看 builder 和 state 记录，看 [KnowledgeBaseBuilder](../core-components/knowledge-base/knowledge-base-builder.zh.md)。
- 想看 cleanup 协议，看 [Step Protocols](../core-components/steps/step-protocols.zh.md)。
