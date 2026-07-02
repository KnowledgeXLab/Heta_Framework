# Knowledge Recipe

`KnowledgeRecipe` is the build plan for a Heta knowledge base.

It does not execute work by itself. It declares the parsers, models, stores, and ordered steps that `KnowledgeBase.create()` will run.

```python
recipe = KnowledgeRecipe(
    parsers=KnowledgeParsers(...),
    models=KnowledgeModels(...),
    stores=KnowledgeStores(...),
    steps=(
        ParseDocuments(),
        SplitDocuments(),
        EmbedChunks(),
        IndexVectors(),
    ),
)
```

## Responsibility

A recipe answers four questions:

| Question | Field |
| --- | --- |
| Which parsers can read source files? | `parsers` |
| Which model components are available? | `models` |
| Where do artifacts, indexes, and tables live? | `stores` |
| Which build actions run, and in what order? | `steps` |

The recipe is intentionally explicit. Heta does not infer that a KB should build vectors, full-text indexes, or graph facts unless the corresponding steps are present.

## Components

`KnowledgeParsers` usually contains a `DocumentParserRegistry`:

```python
KnowledgeParsers(
    documents=DocumentParserRegistry([
        TextParser(),
        HtmlParser(...),
        SheetParser(...),
    ])
)
```

`KnowledgeModels` groups model components:

```python
KnowledgeModels(
    language=llm,
    embedding=embedding,
    reranker=reranker,
)
```

`KnowledgeStores` groups storage components:

```python
KnowledgeStores(
    objects=object_store,
    vector=vector_store,
    sql=sql_store,
    text_index=text_index_store,
)
```

Steps reference these components by stable names such as `models.embedding` or `stores.vector`. A step does not store model or store instances directly.

## Requirements

Each step declares `StepRequirements`. During build, Heta validates that the recipe can satisfy those requirements:

```text
step requirements
  -> required components
  -> required artifacts
  -> required query modes
```

This catches invalid recipes before a later step fails in a less obvious way.

For example, `IndexVectors` requires:

```text
stores.objects
stores.vector
chunk_keys
chunk_embedding_keys
```

If the recipe does not include `EmbedChunks`, `chunk_embedding_keys` will not exist and the build is invalid.

## Capabilities

Steps also declare `StepCapabilities`. Heta uses them to know what the KB supports after a build.

Examples:

| Step | Capability |
| --- | --- |
| `IndexVectors` | `vector_search` |
| `IndexFullText` | `full_text_search` |
| `PersistChunks` | `sql_text_search` |
| `BuildGraph` | `heta_graph_search` |

`KnowledgeBase.available_queries` comes from these capabilities.

## Manifest

A recipe can produce a manifest-like representation for traceability:

```python
recipe.manifest().to_dict()
```

The manifest records component names, step names, requirements, and capabilities. It is useful for runtime metadata, evaluation reports, and comparing two recipe versions.

The manifest is descriptive. It is not a portable serialization of live model clients or database connections.

## Good Practice

Keep recipes readable:

- Put infrastructure choices in `stores`.
- Put provider choices in `models`.
- Keep steps ordered and explicit.
- Prefer named components only when a recipe genuinely needs multiple components of the same kind.
- Use benchmark reports to compare recipe variants instead of relying on one manual query.

## Next

- To see how Heta executes a recipe, read [KnowledgeBaseBuilder](knowledge-base-builder.en.md).
- To see the user-facing object, read [KnowledgeBase](knowledge-base.en.md).
