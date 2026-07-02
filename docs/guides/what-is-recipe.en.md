# What Is A Recipe

`KnowledgeRecipe` is the main entry point in Heta. It is not a fixed RAG pipeline; it is an executable, reusable, and evaluable build plan for a knowledge base.

A recipe declares:

- Which models to use, such as an LLM, embedding model, or reranker.
- Which stores to use, such as `ObjectStore`, `VectorStore`, `SQLStore`, or `TextIndexStore`.
- Which parsers turn raw files into normalized `ParsedDocument` objects.
- Which steps run in order, such as parse, split, embed, index, or build graph.

## Why Recipe

Many RAG projects start as a few lines of script. Over time they add PDF parsing, vector databases, keyword retrieval, graphs, reranking, evaluation, and multiple deployment environments. The hard part is usually not one component; it is that the component choices become hard-coded in business code and are difficult to replace, reuse, and compare.

Heta moves those choices into a recipe:

```text
Recipe
  -> build KnowledgeBase
  -> unlock query modes
  -> run benchmarks
```

The same recipe can use in-memory stores locally and S3, Milvus, PostgreSQL, or Elasticsearch in production. As long as components satisfy the same protocols, the steps do not need to be rewritten.

## What Recipe Controls

A recipe controls how to build a knowledge base. It is not the already-built knowledge base.

| Item | Declared by Recipe | Executed by |
| --- | --- | --- |
| Parser selection | Yes | `ParseDocuments` |
| Model and store selection | Yes | Steps / query engines |
| Build step order | Yes | `KnowledgeBase.create()` |
| Query mode availability | Indirectly | Opened after corresponding steps finish |
| Benchmark evaluation | Used by `BenchmarkRunner` | `BenchmarkRunner` |

## Minimal Shape

A minimal vector knowledge base usually looks like this:

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

After the build, this KnowledgeBase opens `vector_search`. If you add `IndexFullText`, it opens `full_text_search`; if you add Heta graph steps, it opens Heta graph query modes.

## Next

- To run your first KB, read [Quick Start](../quick-start.en.md).
- To choose a build path, read [Choose A Build Path](choose-build-path.en.md).
- To see the full class contract, read [Knowledge Recipe](../core-components/knowledge-base/knowledge-recipe.en.md).
