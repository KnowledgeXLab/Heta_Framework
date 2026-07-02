# What Is A Recipe

`KnowledgeRecipe` 是 Heta 的核心入口。它不是一个固定 RAG pipeline，而是一份可以执行、复用和评估的知识库构建方案。

一个 recipe 会说明：

- 用哪些 models，例如 LLM、embedding model 或 reranker。
- 用哪些 stores，例如 ObjectStore、VectorStore、SQLStore 或 TextIndexStore。
- 用哪些 parsers，把原始文件解析成统一的 `ParsedDocument`。
- 按什么顺序执行 steps，例如 parse、split、embed、index 或 build graph。

## Why Recipe

很多 RAG 项目一开始只是几行脚本，后面会慢慢加入 PDF 解析、向量库、关键词检索、图谱、rerank、评测和多环境部署。问题通常不在某一个组件，而在这些组件被写死在业务代码里，导致后续很难替换、复用和比较。

Heta 把这些选择收束到 recipe：

```text
Recipe
  -> build KnowledgeBase
  -> unlock query modes
  -> run benchmarks
```

同一个 recipe 可以在本地使用内存 store，也可以在生产环境换成 S3、Milvus、PostgreSQL 或 Elasticsearch。只要组件满足同一协议，steps 不需要重写。

## What Recipe Controls

Recipe 控制“怎么建库”，不直接代表已经建好的知识库。

| 内容 | 由 Recipe 声明 | 运行时由谁执行 |
| --- | --- | --- |
| parser 选择 | 是 | `ParseDocuments` |
| model 和 store 选择 | 是 | 各个 step / query engine |
| build step 顺序 | 是 | `KnowledgeBase.create()` |
| query mode 是否可用 | 间接声明 | 完成对应 step 后开放 |
| benchmark 如何评估 | 被 `BenchmarkRunner` 使用 | `BenchmarkRunner` |

## Minimal Shape

一个最小向量知识库通常包含：

```python
recipe = KnowledgeRecipe(
    parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
    models=KnowledgeModels(embedding=embedding),
    stores=KnowledgeStores(objects=objects, vector=vectors),
    steps=(
        ParseDocuments(),
        SplitDocuments(),
        EmbedChunks(),
        IndexVectors(),
    ),
)
```

构建完成后，这个 KnowledgeBase 会开放 `vector_search`。如果继续加入 `IndexFullText`，会开放 `full_text_search`；如果加入 Heta graph steps，会开放 Heta graph 相关查询。

## Next

- 想先跑起来，看 [Quick Start](../quick-start.zh.md)。
- 想选择构建路径，看 [Choose A Build Path](choose-build-path.zh.md)。
- 想理解完整类职责，看 [Knowledge Recipe](../core-components/knowledge-base/knowledge-recipe.zh.md)。
