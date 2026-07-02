# Heta

<p align="center">
  <img src="docs/images/banner.png" alt="Heta Framework banner" width="100%">
</p>

<p align="center">
  <a href="https://knowledgexlab.github.io/">KnowledgeX Lab</a>
</p>

Heta is a Python framework for building, querying, and evaluating knowledge
bases from composable recipes.

## Why Heta

Most RAG projects become hard to change when parsing, models, storage, indexing,
retrieval, and evaluation logic are wired directly into application code. Heta
keeps those parts explicit and replaceable:

- **Models** wrap LLM and embedding providers through LiteLLM.
- **Stores** give objects, vectors, SQL rows, and graph facts consistent APIs.
- **Steps** describe one build action and the search capability it unlocks.
- **Recipes** compose models, stores, parsers, and steps into a reusable build
  plan.
- **KnowledgeBase** runs a recipe and exposes the search modes that were built.
- **Benchmarks** evaluate a recipe by building KBs, running queries, and writing
  reports.

Heta is not a fixed RAG pipeline. A recipe is the unit of composition: choose the
components, choose the steps, build the KB, and run the same recipe against real
benchmarks.

<p align="center">
  <img src="docs/images/recipe-to-kb.png" alt="A recipe builds a Heta knowledge base" width="92%">
</p>

## Install

```bash
pip install heta
```

Optional extras:

```bash
pip install "heta[sql]"          # SQLStore and sql_text_search with SQLite or generic SQL
pip install "heta[postgres]"     # PostgreSQL driver for SQLStore and PostgreSQL text ranking
pip install "heta[mysql]"        # MySQL driver for SQLStore
pip install "heta[milvus]"       # Milvus vector store
pip install "heta[s3]"           # S3-compatible object store
pip install "heta[text-index]"   # Elasticsearch-backed full_text_search
```

Set a model key. Heta uses LiteLLM model names:

```bash
export OPENAI_API_KEY="sk-..."
```

## Quick Start

This example builds a small vector-search KB from plain text, then asks the
query engine to synthesize an answer from the retrieved evidence.

```python
import asyncio
import os
from pathlib import Path

from heta_framework.common.models import EmbeddingModel, LanguageModel
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore
from heta_framework.kb import (
    DocumentParserRegistry,
    EmbedChunks,
    IndexVectors,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    ParseDocuments,
    SplitDocuments,
    TextParser,
)


async def main() -> None:
    workspace = Path("heta-workspace")
    objects = LocalObjectStore(workspace / "objects")
    vectors = InMemoryVectorStore()

    await objects.put(
        "raw/aerospace-notes.txt",
        (
            "Heta builds knowledge bases from recipes. "
            "A recipe combines parsers, models, stores, and steps. "
            "Vector search retrieves chunks by semantic similarity."
        ).encode("utf-8"),
    )

    llm = LanguageModel(
        model_name="openai/gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    embedding = EmbeddingModel(
        model_name="openai/text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(
            documents=DocumentParserRegistry([TextParser()]),
        ),
        models=KnowledgeModels(language=llm, embedding=embedding),
        stores=KnowledgeStores(objects=objects, vector=vectors),
        steps=(
            ParseDocuments(),
            SplitDocuments(),
            EmbedChunks(),
            IndexVectors(),
        ),
    )

    kb = await KnowledgeBase.create(recipe=recipe, name="aerospace-notes")
    response = await kb.query(
        "How does Heta build a knowledge base?",
        mode="vector_search",
        options={"generate_answer": True},
    )

    print(kb.run_record.status)
    print(sorted(kb.available_queries))
    print(response.answer)
    print(response.results[0].text)

    await llm.aclose()
    await embedding.aclose()
    await vectors.aclose()
    await objects.aclose()


asyncio.run(main())
```

## Core Concepts

| Concept | Role |
| --- | --- |
| `KnowledgeRecipe` | Declares components and ordered build steps. |
| `KnowledgeBase` | Created from a recipe; exposes available query modes. |
| `ParseDocuments` | Converts raw objects into Heta `ParsedDocument` records. |
| `SplitDocuments` | Converts parsed documents into stable chunks. |
| `EmbedChunks` | Creates embeddings for chunks. |
| `IndexVectors` | Writes chunk vectors and unlocks `vector_search`. |
| `PersistChunks` | Writes chunk text to SQL and unlocks `sql_text_search`. |
| `IndexFullText` | Writes chunk text to a full-text index and unlocks `full_text_search`. |
| `HetaGraphProcedure` | Expands into entity, relation, and graph build steps. |

## Build Patterns

The examples below are small recipes, not presets. They show how adding or
removing steps changes what the resulting `KnowledgeBase` can do. Any valid
recipe can be built, queried, deleted, and evaluated through the same framework
interfaces.

### 1. Vector Search

Use this for the smallest semantic-search KB. The recipe only needs an object
store, an embedding model, a vector store, and the indexing steps.

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

kb = await KnowledgeBase.create(recipe=recipe, name="vector-kb")
response = await kb.query("knowledge base recipe", mode="vector_search")
```

Unlocked queries:

```text
available queries: vector_search
```

### 2. Vector + SQL Text Search

Add `SQLStore` and `PersistChunks` when exact terms, product codes, legal
clauses, or operational phrases matter. This is the same recipe shape with one
extra store and one extra step.

```python
from heta_framework.common.stores import SQLStore
from heta_framework.kb import PersistChunks, PersistChunksConfig

sql = SQLStore("sqlite:///heta-workspace/chunks.db")

recipe = KnowledgeRecipe(
    parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
    models=KnowledgeModels(embedding=embedding),
    stores=KnowledgeStores(objects=objects, vector=vectors, sql=sql),
    steps=(
        ParseDocuments(),
        SplitDocuments(),
        EmbedChunks(),
        IndexVectors(),
        PersistChunks(PersistChunksConfig(chunk_keys_artifact="chunk_keys")),
    ),
)

kb = await KnowledgeBase.create(recipe=recipe, name="keyword-kb")
semantic = await kb.query("safety checklist", mode="vector_search")
keyword = await kb.query("aerodynamic stall recovery", mode="sql_text_search")
```

Unlocked queries:

```text
available queries: sql_text_search, vector_search
```

### 3. Heta Graph

Add a language model, SQL store, and `HetaGraphProcedure` when the KB should
extract entities and relations, then search graph facts with evidence. The graph
procedure is still just a group of steps, so teams can replace or extend it with
their own graph-building procedure.

```python
from heta_framework.common.models import LanguageModel
from heta_framework.common.stores import SQLStore
from heta_framework.kb import HetaGraphProcedure

llm = LanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)
sql = SQLStore("sqlite:///heta-workspace/graph.db")

recipe = KnowledgeRecipe(
    parsers=KnowledgeParsers(documents=DocumentParserRegistry([TextParser()])),
    models=KnowledgeModels(language=llm, embedding=embedding),
    stores=KnowledgeStores(objects=objects, vector=vectors, sql=sql),
    steps=(
        ParseDocuments(),
        SplitDocuments(),
        EmbedChunks(),
        IndexVectors(),
        *HetaGraphProcedure.build().steps(),
    ),
)

kb = await KnowledgeBase.create(recipe=recipe, name="graph-kb")
facts = await kb.query("What entities are connected?", mode="heta_graph_search")
```

Unlocked queries:

```text
available queries: heta_graph_search, hybrid_search, vector_search
```

## Evaluate Recipes

Benchmarks are built around recipes. A benchmark can create one KB for the whole
corpus or many KBs for case-scoped documents, run the configured query modes,
and write an evaluation report. This makes a recipe easy to compare before it is
used in an application.

The benchmark runner has been exercised with real data:

| Benchmark | Scope | Verified path |
| --- | --- | --- |
| UDA-fin | 788 PDFs, 8190 cases, multi-KB by `doc_name` | Recipe build, source isolation, query, report output. |
| BEIR/SciFact | 5183 documents, 300 queries | Corpus-level KB build, vector search, metric report. |
| MultiHop-RAG | 609 documents, 2556 queries | Corpus-level KB build, vector search, evidence recall report. |

## Swappable Components

The quick examples use local stores so they run anywhere. Production recipes can
swap providers and storage backends without changing the step structure:

```python
objects = S3ObjectStore(...)
vectors = MilvusVectorStore(...)
sql = SQLStore("postgresql+psycopg://user:password@host:5432/db")
```

The naming of tables, collections, and object prefixes stays explicit. Heta does
not hide deployment boundaries from the recipe author.

## Development

```bash
git clone https://github.com/KnowledgeXLab/Heta_Framework.git
cd Heta_Framework
pip install -e ".[dev]"
pytest
```

Build docs:

```bash
mkdocs serve
```
