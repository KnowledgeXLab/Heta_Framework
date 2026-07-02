# Quick Start

这个页面用一个本地文本文件跑通第一个 Heta `KnowledgeBase`。

第一遍只构建最小向量知识库：

```text
raw text
  -> ParseDocuments
  -> SplitDocuments
  -> EmbedChunks
  -> IndexVectors
  -> vector_search
```

这条路径需要的组件最少，适合先确认安装、模型调用、parser、chunk 和向量检索都能正常工作。后续再按需要加入 full-text search、Heta graph search 或 benchmark。

## Install

Heta 已发布到 PyPI。安装时使用包名 `heta`，代码中使用导入名 `heta_framework`。

最小向量示例只需要核心包：

```bash
python -m pip install heta
```

如果你的项目需要生产存储或全文索引，可以按需安装 extra：

```bash
python -m pip install "heta[sql]"          # SQLStore and SQLite/PostgreSQL-style flows
python -m pip install "heta[postgres]"     # PostgreSQL driver
python -m pip install "heta[mysql]"        # MySQL driver
python -m pip install "heta[milvus]"       # Milvus VectorStore
python -m pip install "heta[s3]"           # S3-compatible ObjectStore
python -m pip install "heta[text-index]"   # Elasticsearch full-text index
```

设置模型 API key：

```bash
export OPENAI_API_KEY="sk-..."
```

Heta 的模型层由 LiteLLM 驱动，`model_name` 使用 LiteLLM 的模型命名方式，例如 `openai/gpt-4o-mini`、`openai/text-embedding-3-small`。

## Build Your First KnowledgeBase

创建 `quickstart.py`：

```python
--8<-- "docs/examples/home_vector_case.py"
```

运行：

```bash
python quickstart.py
```

你会看到类似输出：

```text
Heta builds a knowledge base by creating KnowledgeBase objects from Recipe definitions [1].
Heta builds KnowledgeBase objects from Recipe definitions. Vector search retrieves chunks by semantic similarity.
```

第一行是 query engine 用检索结果生成的 answer。第二行是命中的原始 chunk evidence。

这个示例已经完成了三件事：

1. 把 `raw/heta.txt` 写入 `LocalObjectStore`。
2. 用 `TextParser`、`SplitDocuments` 和 `EmbedChunks` 生成 chunk 与 embedding。
3. 用 `IndexVectors` 建立向量索引，并通过 `vector_search` 查询。

## What The Recipe Does

示例中的 recipe 是 Heta 的核心构建单元：

```text
KnowledgeRecipe
  parsers -> TextParser
  models  -> LanguageModel + EmbeddingModel
  stores  -> LocalObjectStore + InMemoryVectorStore
  steps   -> ParseDocuments -> SplitDocuments -> EmbedChunks -> IndexVectors
```

`KnowledgeBase.create()` 会执行这份 recipe。构建完成后，`KnowledgeBase` 只开放当前 recipe 真正构建出来的 query mode。

在这个最小示例里：

```text
available queries: vector_search
```

如果继续添加其他 steps，新的 query mode 会随之开放。

## Generated Files

示例会生成一个本地 workspace：

```text
heta-demo-vector/
  objects/
    raw/
      heta.txt
    parsed/
      ...
    chunks/
      ...
    embeddings/
      ...
    _heta/
      knowledge_bases/
        home-vector/
          manifest.json
          latest_run.json
          runs/
            ...
```

其中：

- `raw/` 保存输入文件。
- `parsed/` 保存统一的 `ParsedDocument`。
- `chunks/` 保存切分后的 `ParsedChunk`。
- `embeddings/` 保存 chunk embedding 产物。
- `_heta/knowledge_bases/...` 保存 KB 的运行记录，供 `load()`、失败恢复和 `delete()` 使用。

这个 quickstart 使用 `InMemoryVectorStore`，所以向量索引只存在于当前进程中。生产环境可以替换为 `MilvusVectorStore`。

## Add More Capabilities

Heta 的构建方式是逐步组合，不需要一开始就选择完整方案。

| 你想要 | 添加什么 | 会得到 |
| --- | --- | --- |
| 语义检索 | `EmbedChunks` + `IndexVectors` | `vector_search` |
| BM25-style 关键词检索 | `IndexFullText` + `TextIndexStore` | `full_text_search` |
| SQL 文本持久化 | `PersistChunks` + `SQLStore` | `sql_text_search` |
| Heta 式图谱检索 | `HetaGraphProcedure` + SQL/vector stores | `heta_graph_search` |
| 混合检索 / rerank / rewrite / multi-hop | vector、full-text、graph 资产组合 | Heta query modes |
| 评估 recipe | `BenchmarkRunner` + benchmark adapter | `EvaluationReport` |

下一步建议：

- 想知道 Recipe 是什么，看 [What Is A Recipe](guides/what-is-recipe.zh.md)。
- 想选择构建路径，看 [Choose A Build Path](guides/choose-build-path.zh.md)。
- 想理解查询方式，看 [Query A KnowledgeBase](guides/query-knowledge-base.zh.md)。
- 想评估不同构建方案，看 [Evaluate A Recipe](guides/evaluate-recipe.zh.md)。

## Add Heta Graph Search

如果你想继续体验 Heta 式建图，可以在最小向量知识库之后加入 `HetaGraphProcedure`。

这条路径会从 chunk 中抽取 entities 和 relations，并把图谱 facts 写入 SQL 与 vector stores，最后开放 `heta_graph_search`。

它需要 SQL 支持：

```bash
python -m pip install "heta[sql]"
```

创建 `graph_quickstart.py`：

```python
--8<-- "docs/examples/home_graph_case.py"
```

运行：

```bash
python graph_quickstart.py
```

你会看到类似输出：

```text
Heta creates a KnowledgeBase by building it from recipes. Specifically, the process involves running the steps outlined in the recipes to construct the KnowledgeBase [1][2][3].
relation Relation: Heta -> KnowledgeBase
Name: builds
Type: creates
Description: Heta builds knowledge bases from recipes.
```

这个例子使用：

- `LocalObjectStore` 保存原始文本和中间产物。
- `SQLStore` 保存实体、关系和 evidence。
- `InMemoryVectorStore` 保存图谱 facts 的向量索引。
- `HetaGraphProcedure.build(deduplicate=False)` 展开 Heta 式建图 steps。

生产环境可以把 SQLite 换成 PostgreSQL，把内存向量库换成 Milvus。Recipe 的整体结构不需要改变。

## Replace Local Components

Recipe 不绑定具体存储实现。生产环境通常只替换 components，steps 可以保持不变：

```python
object_store = S3ObjectStore(...)
vector_store = MilvusVectorStore(...)
sql_store = SQLStore("postgresql+psycopg://postgres:postgres@host:5432/postgres")
```

同一份 recipe 仍然通过：

```python
kb = await KnowledgeBase.create(recipe=recipe, name="production-kb")
```

完成构建。
