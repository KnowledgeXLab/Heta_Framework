# KnowledgeBase

`KnowledgeBase` is the runtime object you use after a recipe has been built or loaded.

It gives you one place to query the KB, inspect available query modes, read build metadata, and delete derived resources.

```text
KnowledgeRecipe
  -> KnowledgeBase.create(...)
  -> KnowledgeBase
  -> kb.query(...)
```

## Create

Use `KnowledgeBase.create()` to build or resume a KB:

```python
kb = await KnowledgeBase.create(
    recipe=recipe,
    name="papers",
)
```

`create()` executes the recipe steps through `KnowledgeBaseBuilder`. If a previous run with the same name was interrupted and runtime state exists, Heta can resume from the recorded state.

The KB name is used in framework metadata paths:

```text
_heta/knowledge_bases/{name}/
```

Use stable names such as `papers`, `faa_handbook`, or `marine_biology_vector_v1`.

## Load

Use `KnowledgeBase.load()` to reopen a completed KB without running steps again:

```python
kb = await KnowledgeBase.load(
    recipe=recipe,
    name="papers",
)
```

`load()` restores metadata, the latest successful run record, capabilities, and query modes. It still uses the runtime components supplied by the recipe, so the same stores and indexes must be reachable.

Use `load()` after service restart, offline build, or benchmark completion.

If the KB did not finish building, call `create()` with the same name to continue instead.

## Query

`KnowledgeBase.query()` is the unified query entry point:

```python
response = await kb.query(
    "How does Heta build a knowledge base?",
    mode="vector_search",
    top_k=5,
    options={"generate_answer": True},
)
```

Every query mode depends on search assets produced by build steps.

Check available modes:

```python
print(sorted(kb.available_queries))
```

Examples:

```text
['vector_search']
['full_text_search', 'vector_search']
['heta_graph_search', 'hybrid_search', 'heta_rerank_search']
```

Calling a mode that the KB did not build raises an error.

## Manifest

`kb.manifest()` returns a snapshot of the built KB:

```python
manifest = kb.manifest()
data = manifest.to_dict()
```

The manifest includes:

- KB name
- recipe manifest
- run record
- available query modes
- search assets
- metadata location

This is useful for service startup checks, evaluation reports, and operational debugging.

## Delete

`KnowledgeBase.delete()` removes derived resources created by the recipe:

```python
result = await kb.delete()
```

By default, Heta keeps user inputs under `raw/`. It deletes resources declared by step cleanup plans, such as:

- parsed documents
- chunks
- embeddings
- extracted entities and relations
- SQL tables
- vector collections
- text indexes
- KB runtime metadata

Dry run first:

```python
plan = kb.delete_plan()
dry_run = await kb.delete(dry_run=True)
```

## Lifecycle

The normal lifecycle is:

```text
create
  -> build or resume
  -> query / benchmark
  -> load after restart
  -> delete when no longer needed
```

This keeps framework metadata inside the KB's `ObjectStore`, instead of depending on process memory.

## Boundary

`KnowledgeBase` does not own your web application's tenant model, authorization, scheduling, or API routing. Those belong to the application layer.

Heta owns the KB build/query/delete lifecycle and the framework metadata needed to make that lifecycle repeatable.

## Next

- To define a recipe, read [Knowledge Recipe](knowledge-recipe.en.md).
- To understand execution records, read [KnowledgeBaseBuilder](knowledge-base-builder.en.md).
- To choose a query mode, read [Query A KnowledgeBase](../../guides/query-knowledge-base.en.md).
