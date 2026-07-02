# KnowledgeBaseBuilder

`KnowledgeBaseBuilder` executes a `KnowledgeRecipe`.

It validates step requirements, runs steps in order, records runtime state, and returns a built `KnowledgeBase`.

```text
KnowledgeRecipe
  -> KnowledgeBaseBuilder
  -> step records
  -> KnowledgeBase
```

Most users call `KnowledgeBase.create(...)`. The builder is the internal execution layer behind that API.

## What The Builder Does

The builder is responsible for:

- resolving recipe components for each step
- checking step requirements before execution
- managing step artifacts
- recording step state and final run records
- collecting capabilities and query modes
- preserving runtime metadata in the KB ObjectStore when available

It does not decide which steps should exist. That belongs to `KnowledgeRecipe`.

## Build Flow

A build runs in this order:

```text
create run state
  -> validate next step requirements
  -> mark step started
  -> run step
  -> store artifacts
  -> mark step succeeded or failed
  -> write final run record
```

This gives the framework enough information to resume interrupted builds and to load a completed KB later.

## Runtime Metadata

When the recipe has an `ObjectStore`, the builder writes metadata under:

```text
_heta/knowledge_bases/{name}/
  manifest.json
  latest_run.json
  runs/
    {run_id}/
      state.json
      record.json
```

`state.json` is for an in-progress run. `record.json` is the final record after success or failure.

This metadata is framework-owned. User input files under `raw/` are not removed or rewritten by the builder.

## Resume Behavior

If a process is interrupted, calling `KnowledgeBase.create(recipe=recipe, name=same_name)` can continue using the existing runtime state.

The builder only resumes within the same KB name and compatible recipe context. It does not guess how to migrate a KB after changing stores, table names, collection names, or step configuration.

## Step Records

Each executed step produces a record with:

```text
step name
status
started_at
finished_at
artifacts produced
capabilities produced
error, if any
```

These records make it possible to inspect which step produced a search asset and why a query mode is available.

## Capabilities

After the build, the builder merges all step capabilities:

```text
artifacts
query modes
search assets
cleanup targets
```

The resulting `KnowledgeBase` exposes only query modes that are backed by these capabilities.

## Boundary

`KnowledgeBaseBuilder` is an execution mechanism, not a business workflow engine.

It does not:

- choose parsers automatically beyond what the parser registry supports
- decide which graph strategy is best
- run benchmarks
- own provider retries beyond model/store implementations
- delete raw user input

## Next

- To define a build plan, read [Knowledge Recipe](knowledge-recipe.en.md).
- To load, query, or delete a built KB, read [KnowledgeBase](knowledge-base.en.md).
