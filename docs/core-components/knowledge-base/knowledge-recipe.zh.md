# Knowledge Recipe

`KnowledgeRecipe` 是知识库的静态构建方案。
它描述要使用哪些 components、按什么顺序执行哪些 steps，但不执行构建。

```python
from heta_framework.kb import KnowledgeModels, KnowledgeRecipe, KnowledgeStores

recipe = KnowledgeRecipe(
    models=KnowledgeModels(
        language=llm,
        embedding=embedding,
    ),
    stores=KnowledgeStores(
        objects=object_store,
        vector=vector_store,
        sql=sql_store,
    ),
    steps=(
        ParseDocuments(...),
        SplitDocuments(...),
        EmbedChunks(...),
        IndexVectors(...),
        *HetaGraphProcedure.build().steps(),
    ),
)
```

`KnowledgeRecipe` 负责：

- 保存构建方案。
- 展开 procedures。
- 根据 `ComponentRef` 查找 runtime components。
- 做静态逻辑校验。
- 导出 recipe manifest。

它不负责：

- 执行 steps。
- 记录运行状态。
- 访问 `ObjectStore` / `SQLStore` / `VectorStore`。
- 调用模型。

## Components

Recipe 持有三类主要 components：

```python
recipe = KnowledgeRecipe(
    parsers=KnowledgeParsers(documents=parser_registry),
    models=KnowledgeModels(language=llm, embedding=embedding),
    stores=KnowledgeStores(objects=object_store, vector=vector_store, sql=sql_store),
    steps=(...),
)
```

Steps 通过 `ComponentRef` 声明依赖：

```text
models.language
models.embedding
stores.objects
stores.vector
stores.sql
parsers.documents
```

Recipe 负责把这些引用解析到真实 runtime 对象。

## Steps And Procedures

`steps` 是 recipe 的执行顺序。
每个 step 声明自己的 requirements 和 capabilities。

```python
steps=(
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    IndexVectors(),
    *HetaGraphProcedure.build().steps(),
)
```

`Procedure` 是可复用的 step composition。
Recipe 校验和执行时都会先展开 procedure，最终仍然以真实 steps 为准。

## Static Validation

Recipe 层只检查静态逻辑：

```python
result = recipe.validate()
recipe.require_valid()
```

检查内容：

- procedure 能否展开。
- component refs 是否已在 recipe 中声明。
- artifact requirements 是否由前序 steps 或 `initial_artifacts` 提供。
- query requirements 是否由前序 steps 或 `initial_queries` 提供。
- 重复 artifact 输出是否可疑。

不检查：

- ObjectStore key 是否存在。
- 数据库是否可连接。
- 模型 API 是否可用。
- LLM 输出是否正确。

校验方式是 ordered dataflow validation。
Recipe 不做拓扑排序，也不会自动重排 steps。

## Manifest

`KnowledgeRecipe.manifest()` 导出静态 recipe 结构：

```text
steps
component_refs
artifacts_required
capabilities_provided
metadata
```

Recipe manifest 适合审计、展示和恢复构建说明。
它不会尝试完整序列化模型 client、SQL engine、Milvus client 等 runtime 对象。
