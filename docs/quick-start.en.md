# Quick Start

This page builds your first Heta `KnowledgeBase` from a local text file.

The first run builds the smallest vector knowledge base:

```text
raw text
  -> ParseDocuments
  -> SplitDocuments
  -> EmbedChunks
  -> IndexVectors
  -> vector_search
```

This path uses the fewest components. It is the fastest way to confirm that installation, model calls, parsing, chunking, and vector search are working. After that, add full-text search, Heta graph search, or benchmarks as needed.

## Install

Heta is published on PyPI. Install it with the package name `heta`; import it in Python with `heta_framework`.

The minimal vector example only needs the core package:

```bash
python -m pip install heta
```

Install extras only when you need production stores or full-text indexing:

```bash
python -m pip install "heta[sql]"          # SQLStore and SQLite/PostgreSQL-style flows
python -m pip install "heta[postgres]"     # PostgreSQL driver
python -m pip install "heta[mysql]"        # MySQL driver
python -m pip install "heta[milvus]"       # Milvus VectorStore
python -m pip install "heta[s3]"           # S3-compatible ObjectStore
python -m pip install "heta[text-index]"   # Elasticsearch full-text index
```

Set your model API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Heta's model layer is powered by LiteLLM. `model_name` follows LiteLLM naming, for example `openai/gpt-4o-mini` or `openai/text-embedding-3-small`.

## Build Your First KnowledgeBase

Create `quickstart.py`:

```python
--8<-- "docs/examples/home_vector_case.py"
```

Run it:

```bash
python quickstart.py
```

You should see output similar to:

```text
Heta builds a knowledge base by creating KnowledgeBase objects from Recipe definitions [1].
Heta builds KnowledgeBase objects from Recipe definitions. Vector search retrieves chunks by semantic similarity.
```

The first line is the answer generated from retrieved evidence. The second line is the source chunk that was retrieved.

This example does three things:

1. Writes `raw/heta.txt` into a `LocalObjectStore`.
2. Uses `TextParser`, `SplitDocuments`, and `EmbedChunks` to create chunks and embeddings.
3. Uses `IndexVectors` to build a vector index and queries it with `vector_search`.

## What The Recipe Does

The recipe is Heta's main build unit:

```text
KnowledgeRecipe
  parsers -> TextParser
  models  -> LanguageModel + EmbeddingModel
  stores  -> LocalObjectStore + InMemoryVectorStore
  steps   -> ParseDocuments -> SplitDocuments -> EmbedChunks -> IndexVectors
```

`KnowledgeBase.create()` executes this recipe. After the build, the `KnowledgeBase` only exposes the query modes that the recipe actually created.

In this minimal example:

```text
available queries: vector_search
```

If you add more steps, new query modes become available.

## Generated Files

The example creates a local workspace:

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

Where:

- `raw/` stores input files.
- `parsed/` stores normalized `ParsedDocument` files.
- `chunks/` stores `ParsedChunk` files.
- `embeddings/` stores chunk embedding artifacts.
- `_heta/knowledge_bases/...` stores KB runtime metadata for `load()`, recovery, and `delete()`.

This quickstart uses `InMemoryVectorStore`, so the vector index only lives in the current process. In production, replace it with `MilvusVectorStore`.

## Add More Capabilities

Heta is designed for gradual composition. You do not need to choose the full architecture up front.

| You want | Add | You get |
| --- | --- | --- |
| Semantic retrieval | `EmbedChunks` + `IndexVectors` | `vector_search` |
| BM25-style keyword retrieval | `IndexFullText` + `TextIndexStore` | `full_text_search` |
| SQL text persistence | `PersistChunks` + `SQLStore` | `sql_text_search` |
| Heta-style graph retrieval | `HetaGraphProcedure` + SQL/vector stores | `heta_graph_search` |
| Hybrid / rerank / rewrite / multi-hop | Vector, full-text, and graph assets | Heta query modes |
| Recipe evaluation | `BenchmarkRunner` + benchmark adapter | `EvaluationReport` |

Recommended next steps:

- To understand Recipe, read [What Is A Recipe](guides/what-is-recipe.en.md).
- To choose a build path, read [Choose A Build Path](guides/choose-build-path.en.md).
- To query a KB, read [Query A KnowledgeBase](guides/query-knowledge-base.en.md).
- To evaluate build strategies, read [Evaluate A Recipe](guides/evaluate-recipe.en.md).

## Add Heta Graph Search

To try Heta-style graph building, add `HetaGraphProcedure` after the minimal vector KB path.

This path extracts entities and relations from chunks, writes graph facts into SQL and vector stores, and then opens `heta_graph_search`.

It requires SQL support:

```bash
python -m pip install "heta[sql]"
```

Create `graph_quickstart.py`:

```python
--8<-- "docs/examples/home_graph_case.py"
```

Run it:

```bash
python graph_quickstart.py
```

You should see output similar to:

```text
Heta creates a KnowledgeBase by building it from recipes. Specifically, the process involves running the steps outlined in the recipes to construct the KnowledgeBase [1][2][3].
relation Relation: Heta -> KnowledgeBase
Name: builds
Type: creates
Description: Heta builds knowledge bases from recipes.
```

This example uses:

- `LocalObjectStore` for raw text and intermediate artifacts.
- `SQLStore` for entities, relations, and evidence.
- `InMemoryVectorStore` for graph fact vectors.
- `HetaGraphProcedure.build(deduplicate=False)` to expand Heta-style graph steps.

In production, replace SQLite with PostgreSQL and the in-memory vector store with Milvus. The recipe structure does not need to change.

## Replace Local Components

Recipes are not tied to a storage implementation. Production deployment usually replaces components, not steps:

```python
object_store = S3ObjectStore(...)
vector_store = MilvusVectorStore(...)
sql_store = SQLStore("postgresql+psycopg://postgres:postgres@host:5432/postgres")
```

The same recipe still builds with:

```python
kb = await KnowledgeBase.create(recipe=recipe, name="production-kb")
```
