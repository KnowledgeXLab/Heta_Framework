# Knowledge Recipe

`KnowledgeRecipe` 是 Heta 的静态构建方案。它声明一个知识库要使用哪些 components，以及这些 components 会按什么 steps 顺序工作。

Recipe 本身不执行构建，也不访问数据库、对象存储或模型 API。它的职责是把构建方案表达清楚，让 `KnowledgeBaseBuilder` 可以按这个方案运行。

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

## Responsibilities

`KnowledgeRecipe` 负责保存构建方案，并在构建前做静态检查。

它会处理：

| 能力 | 说明 |
| --- | --- |
| Components | 保存 parser、model、store 等 runtime components。 |
| Steps | 保存有序 build steps。 |
| Procedures | 展开可复用的 step composition。 |
| Component refs | 根据 `ComponentRef` 找到 recipe 中声明的 runtime component。 |
| Static validation | 检查 step 顺序、依赖和产出是否自洽。 |
| Manifest | 导出 recipe 的静态描述。 |

它不会处理：

| 不负责的内容 | 原因 |
| --- | --- |
| 执行 steps | 执行由 `KnowledgeBaseBuilder` 负责。 |
| 记录运行状态 | 运行状态属于 builder / KB runtime metadata。 |
| 连接存储后端 | Recipe 只持有 runtime object，不主动访问后端。 |
| 调用模型 | 模型调用发生在具体 step 或 query engine 中。 |

## Components

Recipe 持有 Heta 构建过程中会用到的 components：

```python
recipe = KnowledgeRecipe(
    parsers=KnowledgeParsers(documents=parser_registry),
    models=KnowledgeModels(language=llm, embedding=embedding),
    stores=KnowledgeStores(objects=object_store, vector=vector_store, sql=sql_store),
    steps=(...),
)
```

Steps 不直接按变量名查找对象，而是通过 `ComponentRef` 声明依赖：

```text
models.language
models.embedding
stores.objects
stores.vector
stores.sql
parsers.documents
```

这样 step 只关心自己需要什么能力，recipe 负责把引用解析到真实 runtime object。

## Steps And Procedures

`steps` 是构建顺序。Heta 不会自动重排 steps，因此顺序本身就是 recipe 的一部分。

```python
steps=(
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    IndexVectors(),
    *HetaGraphProcedure.build().steps(),
)
```

每个 step 会声明：

| 声明 | 用途 |
| --- | --- |
| `requirements` | 这个 step 运行前需要哪些 artifacts、queries 或 components。 |
| `capabilities` | 这个 step 运行后会产出哪些 artifacts、search assets 或 query modes。 |

`Procedure` 是一组可复用 steps。Recipe 在校验和执行前都会展开 procedure，最终仍然按真实 step 列表运行。

## Static Validation

Recipe 层只做静态逻辑检查：

```python
result = recipe.validate()
recipe.require_valid()
```

当前检查包括：

- procedure 是否能展开。
- step 需要的 component refs 是否已经在 recipe 中声明。
- step 需要的 artifacts 是否由前序 steps 或 `initial_artifacts` 提供。
- step 需要的 query modes 是否由前序 steps 或 `initial_queries` 提供。
- 多个 steps 输出同名 artifact 时是否可疑。

Recipe 不检查运行时状态：

- `ObjectStore` key 是否真实存在。
- SQL / Vector / Text index 后端是否可连接。
- 模型 API key 是否有效。
- LLM 输出是否符合预期。

这类问题只能在运行时由 builder、step 或 query engine 处理。

## Manifest

`KnowledgeRecipe.manifest()` 会导出 recipe 的静态结构：

```text
steps
component_refs
artifacts_required
capabilities_provided
metadata
```

Manifest 适合用于审计、展示和恢复构建说明。它不会完整序列化模型 client、SQL engine、Milvus client 等 runtime objects，也不会保存 API key 或数据库连接。
