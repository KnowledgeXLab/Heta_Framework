# Load And Delete A KnowledgeBase

The Heta `KnowledgeBase` lifecycle has three operations:

```text
create  -> build or resume
load    -> reopen a completed KB
delete  -> remove derived resources
```

This covers local scripts, web backends, and offline evaluations.

## Create

`KnowledgeBase.create()` builds a knowledge base from a recipe:

```python
kb = await KnowledgeBase.create(recipe=recipe, name="papers")
```

When the recipe has an `ObjectStore`, Heta writes runtime metadata under a reserved prefix:

```text
_heta/knowledge_bases/{name}/
  manifest.json
  latest_run.json
  runs/
    {run_id}/
      state.json
      record.json
```

If the process is interrupted, calling `create()` again with the same name can read existing state and continue from unfinished steps.

## Load

`KnowledgeBase.load()` reopens a successfully built KB:

```python
kb = await KnowledgeBase.load(recipe=recipe, name="papers")
```

`load()` does not rerun steps and does not rewrite indexes. It restores metadata, run record, and query capabilities, then uses the runtime components provided by the recipe.

Use it when:

- A web service restarts and needs to mount an existing KB.
- An evaluation has finished and you want to query the result again.
- A KB was built offline and should be used in another process.

If the KB has not successfully finished building, continue with same-name `create()` instead of `load()`.

## Delete

`KnowledgeBase.delete()` removes derived artifacts while keeping user inputs:

```python
result = await kb.delete()
```

By default, Heta does not delete original files under `raw/`. It deletes resources declared by step cleanup plans:

- parsed documents
- chunks
- embeddings
- extracted entities and relations
- SQL tables
- vector collections
- text indexes
- `_heta/knowledge_bases/{name}/` runtime metadata

To inspect the deletion first:

```python
plan = kb.delete_plan()
dry_run = await kb.delete(dry_run=True)
```

## Naming

The KB name is used in runtime metadata paths. Use a stable, readable, unique name:

```text
papers
faa_handbook
marine_biology_vector_v1
```

Avoid temporary random strings for production KBs. Stable names make recovery, load, and delete easier.

## Next

- For the full API, read [KnowledgeBase](../core-components/knowledge-base/knowledge-base.en.md).
- For builder state records, read [KnowledgeBaseBuilder](../core-components/knowledge-base/knowledge-base-builder.en.md).
- For cleanup contracts, read [Step Protocols](../core-components/steps/step-protocols.en.md).
